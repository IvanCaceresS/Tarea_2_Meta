# scripts/tabu_search.py
import random
import copy
import math

# Reutilizar funciones de evaluación y búsqueda de inserción de grasp_hc
from .grasp_hc import evaluar_solucion_penalizada, _encontrar_mejor_tiempo_insercion_en_pista

def tabu_search_resolver(
    datos_del_caso, 
    solucion_inicial_dict, 
    num_pistas_ts, 
    config_ts, # Diccionario: {'tabu_tenure', 'max_iterations_ts', 'max_iter_no_improve_ts'}
    penalidad_lk, 
    penalidad_sep, 
    penalidad_no_prog
):
    """
    Resuelve el problema usando Búsqueda Tabú.

    Args:
        datos_del_caso (dict): Datos del problema.
        solucion_inicial_dict (dict): Solución inicial completa (de GD o GE).
                                     Debe tener 'secuencia_aterrizajes' y 'aviones_no_programados'.
        num_pistas_ts (int): Número de pistas.
        config_ts (dict): Parámetros para TS {'tabu_tenure', 'max_iterations_ts', 'max_iter_no_improve_ts'}.
        penalidad_lk, penalidad_sep, penalidad_no_prog (float): Valores de penalización.

    Returns:
        dict: La mejor solución encontrada por TS, con la misma estructura que las otras heurísticas.
    """
    tabu_tenure = config_ts['tabu_tenure']
    max_iterations_ts = config_ts['max_iterations_ts']
    max_iter_no_improve_ts = config_ts['max_iter_no_improve_ts']

    # Inicialización
    current_sol_dict = copy.deepcopy(solucion_inicial_dict)
    
    # Asegurarse de que la solución inicial tenga todos los aviones programados si es posible
    # Esto es más una verificación, ya que los Greedys deberían proveer soluciones completas.
    if current_sol_dict.get('aviones_no_programados'):
        # print(f"TS Warning: Solución inicial tiene aviones no programados: {current_sol_dict['aviones_no_programados']}. TS operará sobre los programados.")
        # Para TS, es mejor si opera sobre una secuencia donde todos tienen un lugar, incluso si es malo.
        # La evaluación penalizada se encargará de los costos.
        pass


    eval_current = evaluar_solucion_penalizada(
        current_sol_dict['secuencia_aterrizajes'], 
        current_sol_dict.get('aviones_no_programados', []), 
        datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog
    )
    current_costo_penalizado = eval_current['costo_penalizado']

    best_sol_overall_dict = copy.deepcopy(current_sol_dict)
    eval_best_overall = copy.deepcopy(eval_current)
    
    # La lista tabú almacenará tuplas (avion_id_movido, iteracion_expira_tabu)
    # O más simple: avion_id -> iteracion_expira_tabu
    tabu_list = {}  # avion_id -> iteracion_donde_expira

    iter_count = 0
    iters_sin_mejora_global = 0

    # print(f"  TS Inicio: CostoPenalizado={current_costo_penalizado:.2f}, CostoBase={eval_current['costo_base']:.2f}, Factible={eval_current['es_estrictamente_factible']}")

    while iter_count < max_iterations_ts and iters_sin_mejora_global < max_iter_no_improve_ts:
        iter_count += 1
        
        mejor_vecino_iter_info = {
            'costo_penalizado': float('inf'),
            'secuencia_lista': None,
            'eval_dict': None,
            'avion_id_movido_para_tabu': None
        }

        if not current_sol_dict['secuencia_aterrizajes']: # No hay aviones para mover
            # print("  TS: Secuencia actual vacía, deteniendo.")
            break

        # Explorar vecindario: mover cada avión a su mejor nueva posición
        for idx_avion_a_mover in range(len(current_sol_dict['secuencia_aterrizajes'])):
            avion_original_ater = current_sol_dict['secuencia_aterrizajes'][idx_avion_a_mover]
            avion_id_a_mover = avion_original_ater['avion_id']
            
            avion_info_a_mover = next((a for a in datos_del_caso['aviones'] if a['id'] == avion_id_a_mover), None)
            if not avion_info_a_mover: continue

            # Verificar si el *movimiento de este avión* es tabú
            es_avion_tabu_para_mover = (avion_id_a_mover in tabu_list and tabu_list[avion_id_a_mover] > iter_count)

            secuencia_temp_sin_avion = [ater for i, ater in enumerate(current_sol_dict['secuencia_aterrizajes']) if i != idx_avion_a_mover]

            # Encontrar la mejor nueva posición (pista, tiempo) para este avión
            mejor_nueva_pista_para_este_avion = -1
            mejor_nuevo_tiempo_para_este_avion = None
            menor_costo_individual_logrado = float('inf')

            for pista_destino_idx in range(num_pistas_ts):
                aterrizajes_en_pista_destino = sorted(
                    [ater for ater in secuencia_temp_sin_avion if ater['pista'] == pista_destino_idx],
                    key=lambda x: x['tiempo']
                )
                
                tiempo_opt_pista, costo_ind_opt_pista = _encontrar_mejor_tiempo_insercion_en_pista(
                    avion_info_a_mover, pista_destino_idx, aterrizajes_en_pista_destino, datos_del_caso
                )

                if tiempo_opt_pista is not None:
                    # Construir la secuencia vecina potencial
                    vecino_seq_lista = copy.deepcopy(secuencia_temp_sin_avion)
                    vecino_seq_lista.append({
                        'avion_id': avion_id_a_mover,
                        'pista': pista_destino_idx,
                        'tiempo': tiempo_opt_pista,
                        'costo_individual': costo_ind_opt_pista 
                    })
                    vecino_seq_lista.sort(key=lambda x: x['tiempo'])
                    
                    eval_vecino = evaluar_solucion_penalizada(
                        vecino_seq_lista, 
                        current_sol_dict.get('aviones_no_programados', []), # Aviones no programados no cambian con este movimiento
                        datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog
                    )

                    # Aplicar lógica Tabú y Aspiración
                    acepta_movimiento = False
                    if not es_avion_tabu_para_mover:
                        acepta_movimiento = True
                    elif eval_vecino['costo_penalizado'] < eval_best_overall['costo_penalizado']: # Criterio de Aspiración
                        acepta_movimiento = True
                        # print(f"    TS Aspiración: Movimiento tabú de avión {avion_id_a_mover} aceptado. Nuevo costo {eval_vecino['costo_penalizado']:.2f} < MejorGlobal {eval_best_overall['costo_penalizado']:.2f}")
                    
                    if acepta_movimiento:
                        if eval_vecino['costo_penalizado'] < mejor_vecino_iter_info['costo_penalizado']:
                            mejor_vecino_iter_info['costo_penalizado'] = eval_vecino['costo_penalizado']
                            mejor_vecino_iter_info['secuencia_lista'] = vecino_seq_lista
                            mejor_vecino_iter_info['eval_dict'] = eval_vecino
                            mejor_vecino_iter_info['avion_id_movido_para_tabu'] = avion_id_a_mover
        
        # Moverse al mejor vecino encontrado en esta iteración (si existe)
        if mejor_vecino_iter_info['secuencia_lista'] is not None:
            current_sol_dict['secuencia_aterrizajes'] = mejor_vecino_iter_info['secuencia_lista']
            eval_current = mejor_vecino_iter_info['eval_dict']
            current_costo_penalizado = eval_current['costo_penalizado']
            
            # Actualizar lista tabú: hacer tabú el avión que se movió
            avion_movido_id = mejor_vecino_iter_info['avion_id_movido_para_tabu']
            tabu_list[avion_movido_id] = iter_count + tabu_tenure
            
            # Limpiar entradas expiradas de la lista tabú (opcional, pero bueno para la memoria)
            # O simplemente se sobrescriben o se ignoran si tenure > iter_actual
            # No es estrictamente necesario si se chequea `tabu_list[avion_id] > iter_count`

            # Actualizar la mejor solución global
            if current_costo_penalizado < eval_best_overall['costo_penalizado']:
                best_sol_overall_dict = copy.deepcopy(current_sol_dict)
                eval_best_overall = copy.deepcopy(eval_current)
                iters_sin_mejora_global = 0
                # print(f"    TS Iter {iter_count}: Nueva mejor global. Penalizado={eval_best_overall['costo_penalizado']:.2f}, Base={eval_best_overall['costo_base']:.2f}, Factible={eval_best_overall['es_estrictamente_factible']}")
            else:
                iters_sin_mejora_global += 1
        else:
            # No se encontró ningún movimiento válido (quizás todos eran tabú y no cumplían aspiración)
            iters_sin_mejora_global += 1
            # print(f"    TS Iter {iter_count}: No se encontró movimiento válido o mejora.")


        if iters_sin_mejora_global >= max_iter_no_improve_ts:
            # print(f"  TS: Detenido por {max_iter_no_improve_ts} iteraciones sin mejora global.")
            break
    
    # print(f"  TS Finalizado. Iter: {iter_count}. Mejor Penalizado={eval_best_overall['costo_penalizado']:.2f}, Mejor Base={eval_best_overall['costo_base']:.2f}, Factible={eval_best_overall['es_estrictamente_factible']}")
    
    # Devolver la mejor solución global encontrada
    return {
        'secuencia_aterrizajes': best_sol_overall_dict['secuencia_aterrizajes'],
        'costo_total': eval_best_overall['costo_base'], 
        'costo_penalizado': eval_best_overall['costo_penalizado'], 
        'es_factible': eval_best_overall['es_estrictamente_factible'], 
        'aviones_no_programados': best_sol_overall_dict.get('aviones_no_programados', []), # Heredado de la inicial
        'violaciones_lk_count': eval_best_overall['violaciones_lk_count'],
        'violaciones_sep_count': eval_best_overall['violaciones_sep_count'],
        'violaciones_no_prog_count': eval_best_overall['violaciones_no_prog_count']
    }

if __name__ == '__main__':
    import sys
    import os
    directorio_base_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(directorio_base_proyecto)
    
    from scripts.lector_cases import read_case
    from scripts.greedy_deterministic import resolver as construir_solucion_determinista

    nombre_caso_prueba = 'case1.txt' 
    num_pistas_prueba = 1       
    
    ts_configs_test = [
        {'id_config': 1, 'tabu_tenure': 5, 'max_iterations_ts': 50, 'max_iter_no_improve_ts': 10},
        {'id_config': 2, 'tabu_tenure': 10, 'max_iterations_ts': 50, 'max_iter_no_improve_ts': 10},
    ]
    
    pen_lk_test = 1000.0
    pen_sep_test = 5000.0
    pen_no_prog_test = 100000.0

    ruta_completa_caso = os.path.join(directorio_base_proyecto, 'cases', nombre_caso_prueba)
    print(f"Cargando datos desde: {ruta_completa_caso}")
    datos_caso_ts = read_case(ruta_completa_caso)

    if datos_caso_ts:
        print(f"\n--- Prueba Tabu Search para: {nombre_caso_prueba}, {num_pistas_prueba} pista(s) ---")
        
        sol_gd_inicial_ts = construir_solucion_determinista(datos_caso_ts, num_pistas_prueba)
        eval_gd_ts = evaluar_solucion_penalizada(sol_gd_inicial_ts['secuencia_aterrizajes'], sol_gd_inicial_ts['aviones_no_programados'], datos_caso_ts, pen_lk_test, pen_sep_test, pen_no_prog_test)
        print(f"  Solución Inicial (GD): CostoBase={eval_gd_ts['costo_base']:.2f}, Penalizado={eval_gd_ts['costo_penalizado']:.2f}, Factible={eval_gd_ts['es_estrictamente_factible']}")

        for config_ts_actual in ts_configs_test:
            print(f"\n  Ejecutando TS con config ID {config_ts_actual['id_config']}: {config_ts_actual}")
            solucion_final_ts = tabu_search_resolver(
                datos_caso_ts,
                sol_gd_inicial_ts,
                num_pistas_prueba,
                config_ts_actual,
                pen_lk_test, pen_sep_test, pen_no_prog_test
            )
            print(f"    Resultado TS (Cfg {config_ts_actual['id_config']}):")
            print(f"      Costo Base Final: {solucion_final_ts['costo_total']:.2f}")
            print(f"      Costo Penalizado Final: {solucion_final_ts['costo_penalizado']:.2f}")
            print(f"      Es Estrictamente Factible: {solucion_final_ts['es_factible']}")
            if solucion_final_ts['aviones_no_programados']:
                 print(f"      Aviones No Programados: {solucion_final_ts['aviones_no_programados']}")
            print(f"      Violaciones Lk: {solucion_final_ts['violaciones_lk_count']}, Sep: {solucion_final_ts['violaciones_sep_count']}, NoProg: {solucion_final_ts['violaciones_no_prog_count']}")
    else:
        print(f"No se pudieron cargar los datos para {nombre_caso_prueba}.")
