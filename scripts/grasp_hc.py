# scripts/grasp_hc.py
import random
import copy
import math

# Asumimos que greedy_stochastic está en el mismo directorio o accesible vía PYTHONPATH
from .greedy_stochastic import resolver as construir_solucion_stochastic
# También necesitaremos la función de cálculo de costo individual
from .greedy_stochastic import calcular_costo_aterrizaje as calcular_costo_individual_avion

def evaluar_solucion_penalizada(secuencia_aterrizajes_lista, aviones_no_programados_ids, datos_del_caso, penalidad_lk_por_unidad=1000, penalidad_separacion_fija=5000, penalidad_no_programado_fija=100000):
    """
    Evalúa una solución, calculando su costo base y un costo penalizado.
    El costo penalizado incluye multas por violaciones de L_k, separación y aviones no programados.

    Args:
        secuencia_aterrizajes_lista (list): Lista de diccionarios de aterrizajes.
        aviones_no_programados_ids (list): Lista de IDs de aviones no programados.
        datos_del_caso (dict): Datos del problema.
        penalidad_lk_por_unidad (float): Penalización por cada unidad de tiempo que se excede L_k.
        penalidad_separacion_fija (float): Penalización fija por cada violación de separación.
        penalidad_no_programado_fija (float): Penalización fija por cada avión no programado.

    Returns:
        dict: {
            'costo_penalizado': float,
            'costo_base': float,      # Suma de costos individuales sin penalizaciones de HC
            'es_estrictamente_factible': bool, # True si no hay violaciones de Lk ni separación y todos programados
            'violaciones_lk_count': int,
            'violaciones_sep_count': int,
            'violaciones_no_prog_count': int
        }
    """
    aviones_data_dict = {avion['id']: avion for avion in datos_del_caso['aviones']}
    tiempos_separacion_matriz = datos_del_caso['tiempos_separacion']
    num_pistas = 0
    if secuencia_aterrizajes_lista:
        num_pistas = max(ater['pista'] for ater in secuencia_aterrizajes_lista) + 1
    elif datos_del_caso.get('num_pistas_original', 0) > 0 : # Si main.py pasa esta info
        num_pistas = datos_del_caso['num_pistas_original']


    costo_base_calculado = 0
    costo_penalizado_total = 0
    
    violaciones_lk_count = 0
    violaciones_sep_count = 0
    
    # Penalización por aviones no programados
    violaciones_no_prog_count = len(aviones_no_programados_ids)
    costo_penalizado_total += violaciones_no_prog_count * penalidad_no_programado_fija

    # Ordenar copia para no modificar la original externamente si no se desea
    secuencia_ordenada = sorted(copy.deepcopy(secuencia_aterrizajes_lista), key=lambda x: x['tiempo'])

    # Calcular costo base y penalizaciones L_k
    for ater in secuencia_ordenada:
        avion_info = aviones_data_dict[ater['avion_id']]
        costo_ind = calcular_costo_individual_avion(ater['tiempo'], avion_info)
        ater['costo_individual_recalculado'] = costo_ind # Guardar para posible análisis
        costo_base_calculado += costo_ind

        if ater['tiempo'] > avion_info['L']:
            violaciones_lk_count += 1
            costo_penalizado_total += (ater['tiempo'] - avion_info['L']) * penalidad_lk_por_unidad
    
    costo_penalizado_total += costo_base_calculado # Sumar el costo base al penalizado

    # Verificar y penalizar separaciones
    # Crear listas de aterrizajes por pista
    aterrizajes_por_pista = [[] for _ in range(num_pistas if num_pistas > 0 else 1)] # Evitar error si no hay pistas
    if num_pistas > 0:
        for ater in secuencia_ordenada:
            if 0 <= ater['pista'] < num_pistas:
                 aterrizajes_por_pista[ater['pista']].append(ater)
            # else: # Esto indicaría un error en la generación de la solución
            # print(f"Error: Pista {ater['pista']} fuera de rango para avión {ater['avion_id']}")


    for pista_idx in range(num_pistas):
        # Los aterrizajes en 'aterrizajes_por_pista[pista_idx]' ya están ordenados por tiempo
        # porque 'secuencia_ordenada' lo estaba.
        for i in range(len(aterrizajes_por_pista[pista_idx]) - 1):
            avion_i_ater = aterrizajes_por_pista[pista_idx][i]
            avion_j_ater = aterrizajes_por_pista[pista_idx][i+1]

            id_i = avion_i_ater['avion_id']
            id_j = avion_j_ater['avion_id']
            
            tiempo_sep_req = tiempos_separacion_matriz[id_i][id_j]
            
            if avion_j_ater['tiempo'] < avion_i_ater['tiempo'] + tiempo_sep_req:
                violaciones_sep_count += 1
                costo_penalizado_total += penalidad_separacion_fija
                # print(f"Debug: Violación Sep Pista {pista_idx}: Av {id_i}@{avion_i_ater['tiempo']} -> Av {id_j}@{avion_j_ater['tiempo']}. Req: {tiempo_sep_req}")


    es_estrictamente_factible = (violaciones_lk_count == 0 and violaciones_sep_count == 0 and violaciones_no_prog_count == 0)

    return {
        'costo_penalizado': costo_penalizado_total,
        'costo_base': costo_base_calculado,
        'es_estrictamente_factible': es_estrictamente_factible,
        'violaciones_lk_count': violaciones_lk_count,
        'violaciones_sep_count': violaciones_sep_count,
        'violaciones_no_prog_count': violaciones_no_prog_count
    }


def _encontrar_mejor_tiempo_insercion_en_pista(avion_info, pista_id_objetivo, otros_aterrizajes_en_pista, datos_del_caso):
    """
    Encuentra el mejor tiempo de aterrizaje para 'avion_info' en 'pista_id_objetivo',
    considerando los 'otros_aterrizajes_en_pista'.
    "Mejor" significa que minimiza el costo individual del avión, respetando E_k, L_k y separaciones.

    Args:
        avion_info (dict): Información del avión a insertar.
        pista_id_objetivo (int): ID de la pista donde se intentará insertar.
        otros_aterrizajes_en_pista (list): Lista de aterrizajes ya existentes en esa pista, ordenados por tiempo.
        datos_del_caso (dict): Datos del problema.

    Returns:
        tuple: (mejor_tiempo_encontrado, costo_individual_asociado) o (None, float('inf')) si no es posible.
    """
    tiempos_separacion_matriz = datos_del_caso['tiempos_separacion']
    
    mejor_tiempo_hallado = None
    menor_costo_individual = float('inf')

    # Caso 1: Insertar al principio de la pista
    tiempo_min_ater_actual = avion_info['E']
    tiempo_max_ater_actual = avion_info['L']

    if not otros_aterrizajes_en_pista: # Pista vacía
        tiempo_ater_candidato = max(avion_info['E'], avion_info['P'])
        if tiempo_ater_candidato > avion_info['L']: # P_k es muy tarde, aterrizar en E_k si es posible
            tiempo_ater_candidato = avion_info['E']
        
        if tiempo_ater_candidato <= avion_info['L']: # Asegurar que E_k no sea > L_k
            costo_cand = calcular_costo_individual_avion(tiempo_ater_candidato, avion_info)
            if costo_cand < menor_costo_individual:
                menor_costo_individual = costo_cand
                mejor_tiempo_hallado = tiempo_ater_candidato
    else:
        # Intentar insertar antes del primer avión actual en la pista
        primer_avion_actual = otros_aterrizajes_en_pista[0]
        tiempo_max_permisible_antes_primero = primer_avion_actual['tiempo'] - tiempos_separacion_matriz[avion_info['id']][primer_avion_actual['avion_id']]
        
        tiempo_min_valido_gap = tiempo_min_ater_actual
        tiempo_max_valido_gap = min(tiempo_max_ater_actual, tiempo_max_permisible_antes_primero)

        if tiempo_min_valido_gap <= tiempo_max_valido_gap:
            tiempo_ater_candidato = max(tiempo_min_valido_gap, avion_info['P'])
            if tiempo_ater_candidato > tiempo_max_valido_gap: # P_k es muy tarde para este gap
                 tiempo_ater_candidato = tiempo_min_valido_gap # Intentar lo más temprano posible en el gap
            
            if tiempo_ater_candidato <= tiempo_max_valido_gap : # Re-validar
                costo_cand = calcular_costo_individual_avion(tiempo_ater_candidato, avion_info)
                if costo_cand < menor_costo_individual:
                    menor_costo_individual = costo_cand
                    mejor_tiempo_hallado = tiempo_ater_candidato
                elif costo_cand == menor_costo_individual and (mejor_tiempo_hallado is None or tiempo_ater_candidato < mejor_tiempo_hallado):
                    mejor_tiempo_hallado = tiempo_ater_candidato


        # Intentar insertar entre aviones existentes o al final
        for i in range(len(otros_aterrizajes_en_pista)):
            avion_prev = otros_aterrizajes_en_pista[i]
            
            tiempo_min_despues_prev = avion_prev['tiempo'] + tiempos_separacion_matriz[avion_prev['avion_id']][avion_info['id']]
            
            if i + 1 < len(otros_aterrizajes_en_pista): # Hay un avión siguiente
                avion_sig = otros_aterrizajes_en_pista[i+1]
                tiempo_max_antes_sig = avion_sig['tiempo'] - tiempos_separacion_matriz[avion_info['id']][avion_sig['avion_id']]
            else: # Insertar al final
                tiempo_max_antes_sig = tiempo_max_ater_actual # L_k del avión a insertar

            tiempo_min_valido_gap = max(tiempo_min_ater_actual, tiempo_min_despues_prev)
            tiempo_max_valido_gap = min(tiempo_max_ater_actual, tiempo_max_antes_sig)

            if tiempo_min_valido_gap <= tiempo_max_valido_gap:
                tiempo_ater_candidato = max(tiempo_min_valido_gap, avion_info['P'])
                if tiempo_ater_candidato > tiempo_max_valido_gap:
                    tiempo_ater_candidato = tiempo_min_valido_gap
                
                if tiempo_ater_candidato <= tiempo_max_valido_gap: # Re-validar
                    costo_cand = calcular_costo_individual_avion(tiempo_ater_candidato, avion_info)
                    if costo_cand < menor_costo_individual:
                        menor_costo_individual = costo_cand
                        mejor_tiempo_hallado = tiempo_ater_candidato
                    elif costo_cand == menor_costo_individual and (mejor_tiempo_hallado is None or tiempo_ater_candidato < mejor_tiempo_hallado):
                         mejor_tiempo_hallado = tiempo_ater_candidato
                         
    return mejor_tiempo_hallado, menor_costo_individual


def hill_climbing_mejor_mejora(solucion_inicial_dict, datos_del_caso, num_pistas_hc, max_iter_sin_mejora_hc, penalidad_lk, penalidad_sep, penalidad_no_prog):
    """
    Aplica Hill Climbing (Mejor Mejora) a una solución inicial.
    Vecindario: Mover un avión a su mejor posible nueva (pista, tiempo).
    """
    mejor_solucion_hc_lista = copy.deepcopy(solucion_inicial_dict['secuencia_aterrizajes'])
    # Asumimos que la solución inicial ya fue evaluada si viene de GRASP,
    # pero la reevaluamos aquí para tener un costo penalizado consistente.
    # La solución inicial de greedy_stochastic tiene 'aviones_no_programados' y 'es_factible' (basado en Lk y completitud).
    
    eval_actual = evaluar_solucion_penalizada(mejor_solucion_hc_lista, solucion_inicial_dict.get('aviones_no_programados', []), datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog)
    costo_actual_penalizado = eval_actual['costo_penalizado']

    iter_sin_mejora_actual = 0
    num_aviones_en_sol = len(mejor_solucion_hc_lista)

    while iter_sin_mejora_actual < max_iter_sin_mejora_hc:
        mejor_vecino_en_iteracion_lista = None
        eval_mejor_vecino_en_iteracion = None
        costo_mejor_vecino_penalizado_iteracion = costo_actual_penalizado
        
        # Para cada avión en la secuencia actual, intentar moverlo
        for idx_avion_a_mover in range(num_aviones_en_sol):
            avion_movido_id = mejor_solucion_hc_lista[idx_avion_a_mover]['avion_id']
            avion_movido_info = datos_del_caso['aviones'][avion_movido_id] # Acceder por ID original

            # Crear una secuencia temporal sin el avión a mover
            secuencia_temp_sin_avion = [ater for i, ater in enumerate(mejor_solucion_hc_lista) if i != idx_avion_a_mover]
            
            # Encontrar la mejor nueva posición (pista, tiempo) para este avión
            mejor_nueva_pista_para_avion = -1
            mejor_nuevo_tiempo_para_avion = None
            menor_costo_individual_para_avion = float('inf')

            for pista_destino_idx in range(num_pistas_hc):
                # Filtrar aterrizajes en la pista_destino_idx de la secuencia_temp_sin_avion
                aterrizajes_en_pista_destino = sorted(
                    [ater for ater in secuencia_temp_sin_avion if ater['pista'] == pista_destino_idx],
                    key=lambda x: x['tiempo']
                )
                
                tiempo_opt_pista, costo_ind_opt_pista = _encontrar_mejor_tiempo_insercion_en_pista(
                    avion_movido_info, pista_destino_idx, aterrizajes_en_pista_destino, datos_del_caso
                )

                if tiempo_opt_pista is not None:
                    if costo_ind_opt_pista < menor_costo_individual_para_avion:
                        menor_costo_individual_para_avion = costo_ind_opt_pista
                        mejor_nuevo_tiempo_para_avion = tiempo_opt_pista
                        mejor_nueva_pista_para_avion = pista_destino_idx
                    elif costo_ind_opt_pista == menor_costo_individual_para_avion:
                        if mejor_nuevo_tiempo_para_avion is None or tiempo_opt_pista < mejor_nuevo_tiempo_para_avion : # Preferir tiempo más temprano
                            mejor_nuevo_tiempo_para_avion = tiempo_opt_pista
                            mejor_nueva_pista_para_avion = pista_destino_idx

            if mejor_nuevo_tiempo_para_avion is not None: # Si se encontró una nueva posición válida
                # Construir la secuencia vecina
                nueva_secuencia_vecina_lista = copy.deepcopy(secuencia_temp_sin_avion)
                nueva_secuencia_vecina_lista.append({
                    'avion_id': avion_movido_id,
                    'pista': mejor_nueva_pista_para_avion,
                    'tiempo': mejor_nuevo_tiempo_para_avion,
                    'costo_individual': menor_costo_individual_para_avion # Este es el costo individual del avión movido
                })
                # La lista debe reordenarse por tiempo para la evaluación
                nueva_secuencia_vecina_lista.sort(key=lambda x: x['tiempo'])
                
                # Evaluar esta secuencia vecina
                # Aviones no programados es vacío porque estamos moviendo uno que ya estaba programado.
                eval_vecina = evaluar_solucion_penalizada(nueva_secuencia_vecina_lista, [], datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog)
                
                if eval_vecina['costo_penalizado'] < costo_mejor_vecino_penalizado_iteracion:
                    costo_mejor_vecino_penalizado_iteracion = eval_vecina['costo_penalizado']
                    mejor_vecino_en_iteracion_lista = nueva_secuencia_vecina_lista
                    eval_mejor_vecino_en_iteracion = eval_vecina
        
        # Fin de la exploración del vecindario para la iteración actual de HC
        if mejor_vecino_en_iteracion_lista is not None and costo_mejor_vecino_penalizado_iteracion < costo_actual_penalizado:
            mejor_solucion_hc_lista = mejor_vecino_en_iteracion_lista
            costo_actual_penalizado = costo_mejor_vecino_penalizado_iteracion
            eval_actual = eval_mejor_vecino_en_iteracion
            iter_sin_mejora_actual = 0 # Hubo mejora
        else:
            iter_sin_mejora_actual += 1 # No hubo mejora

    # Devolver la mejor solución encontrada por HC
    return {
        'secuencia_aterrizajes': mejor_solucion_hc_lista,
        'costo_total': eval_actual['costo_base'], # Costo base de la solución final de HC
        'costo_penalizado': costo_actual_penalizado, # Costo penalizado de la solución final de HC
        'es_factible': eval_actual['es_estrictamente_factible'], # Factibilidad estricta
        'aviones_no_programados': [] # Asumimos que HC trabaja con soluciones completas
    }


def grasp_resolver(datos_del_caso, num_pistas_grasp, num_iteraciones_grasp, semilla_inicial_grasp, parametro_rcl_ge, max_iter_sin_mejora_hc, penalidad_lk, penalidad_sep, penalidad_no_prog):
    """
    Resuelve el problema usando GRASP con Hill Climbing (Mejor Mejora).
    """
    mejor_solucion_global_grasp = None
    mejor_costo_penalizado_global = float('inf')
    mejor_costo_base_global_si_factible = float('inf')


    for i_grasp in range(num_iteraciones_grasp):
        semilla_ge_actual = semilla_inicial_grasp + i_grasp
        # print(f"  GRASP Iteración {i_grasp + 1}/{num_iteraciones_grasp} (Semilla GE: {semilla_ge_actual})")

        # 1. Fase de Construcción (usando greedy estocástico)
        solucion_construida_dict = construir_solucion_stochastic(
            datos_del_caso, 
            num_pistas_grasp, 
            semilla_ge_actual, 
            parametro_rcl_alpha=parametro_rcl_ge
        )
        
        # La solución construida ya tiene 'secuencia_aterrizajes', 'costo_total' (base), 'aviones_no_programados', 'es_factible'

        # 2. Fase de Búsqueda Local (Hill Climbing Mejor Mejora)
        solucion_mejorada_hc_dict = hill_climbing_mejor_mejora(
            solucion_construida_dict, # Pasamos el dict completo
            datos_del_caso, 
            num_pistas_grasp, 
            max_iter_sin_mejora_hc,
            penalidad_lk,
            penalidad_sep,
            penalidad_no_prog
        )

        costo_hc_penalizado = solucion_mejorada_hc_dict['costo_penalizado']
        
        # print(f"    Costo Construido (base): {solucion_construida_dict['costo_total']:.2f} (Factible GE: {solucion_construida_dict['es_factible']})")
        # eval_construida = evaluar_solucion_penalizada(solucion_construida_dict['secuencia_aterrizajes'], solucion_construida_dict['aviones_no_programados'], datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog)
        # print(f"    Costo Construido (penalizado): {eval_construida['costo_penalizado']:.2f}")
        # print(f"    Costo Mejorado HC (penalizado): {costo_hc_penalizado:.2f}, (base): {solucion_mejorada_hc_dict['costo_total']:.2f} (Factible HC: {solucion_mejorada_hc_dict['es_factible']})")


        if mejor_solucion_global_grasp is None or costo_hc_penalizado < mejor_costo_penalizado_global:
            mejor_costo_penalizado_global = costo_hc_penalizado
            mejor_solucion_global_grasp = copy.deepcopy(solucion_mejorada_hc_dict)
            mejor_costo_base_global_si_factible = solucion_mejorada_hc_dict['costo_total'] if solucion_mejorada_hc_dict['es_factible'] else float('inf')
            # print(f"    *** Nueva mejor solución GRASP encontrada. Costo Penalizado: {mejor_costo_penalizado_global:.2f} ***")
        elif costo_hc_penalizado == mejor_costo_penalizado_global:
            # Si los costos penalizados son iguales, preferir la que tenga mejor costo base (si es factible)
            if solucion_mejorada_hc_dict['es_factible'] and solucion_mejorada_hc_dict['costo_total'] < mejor_costo_base_global_si_factible:
                mejor_solucion_global_grasp = copy.deepcopy(solucion_mejorada_hc_dict)
                mejor_costo_base_global_si_factible = solucion_mejorada_hc_dict['costo_total']
                # print(f"    *** Nueva mejor solución GRASP (mismo penalizado, mejor base factible). Costo Base: {mejor_costo_base_global_si_factible:.2f} ***")


    if mejor_solucion_global_grasp is None: # No se pudo generar ninguna solución
        return {
             'secuencia_aterrizajes': [], 
             'costo_total': float('inf'), 
             'costo_penalizado': float('inf'), 
             'es_factible': False, 
             'aviones_no_programados': [i for i in range(datos_del_caso['num_aviones'])] # Todos no programados
        }
        
    return mejor_solucion_global_grasp


if __name__ == '__main__':
    # --- Bloque de prueba ---
    # Añadir la carpeta raíz del proyecto al path para importar módulos del mismo nivel
    import sys
    import os
    directorio_base_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(directorio_base_proyecto)
    
    from scripts.lector_cases import read_case # Importar lector_cases desde la raíz de scripts

    # Configuración de prueba
    nombre_caso_prueba = 'case1.txt' # Puedes cambiar esto a 'case2.txt', 'case3.txt', 'case4.txt'
    num_pistas_prueba = 1       # Puedes cambiar a 2
    
    # Parámetros GRASP
    iteraciones_grasp_test = [5, 10] # Probar con pocas iteraciones para el testeo rápido
    semilla_grasp_inicial_test = 0
    rcl_param_test = 3 # k fijo para RCL
    
    # Parámetros Hill Climbing
    max_iter_hc_test = 20 # Iteraciones máximas sin mejora para HC
    
    # Penalizaciones (ajusta según sea necesario)
    pen_lk = 1000.0
    pen_sep = 5000.0
    pen_no_prog = 100000.0

    # Cargar datos del caso
    ruta_completa_caso = os.path.join(directorio_base_proyecto, 'cases', nombre_caso_prueba)
    print(f"Cargando datos desde: {ruta_completa_caso}")
    datos_caso = read_case(ruta_completa_caso)

    if datos_caso:
        print(f"\n--- Iniciando Prueba GRASP + Hill Climbing para: {nombre_caso_prueba}, {num_pistas_prueba} pista(s) ---")
        
        for num_iter_grasp_actual in iteraciones_grasp_test:
            print(f"\n  Ejecutando GRASP con {num_iter_grasp_actual} iteraciones (restarts):")
            
            solucion_final_grasp = grasp_resolver(
                datos_del_caso,
                num_pistas_prueba,
                num_iter_grasp_actual,
                semilla_grasp_inicial_test,
                rcl_param_test,
                max_iter_hc_test,
                pen_lk,
                pen_sep,
                pen_no_prog
            )

            print(f"    Resultado GRASP ({num_iter_grasp_actual} iters):")
            print(f"      Costo Base Final (si factible): {solucion_final_grasp['costo_total']:.2f}")
            print(f"      Costo Penalizado Final: {solucion_final_grasp['costo_penalizado']:.2f}")
            print(f"      Es Estrictamente Factible: {solucion_final_grasp['es_factible']}")
            if solucion_final_grasp['aviones_no_programados']:
                 print(f"      Aviones No Programados: {solucion_final_grasp['aviones_no_programados']}")
            
            # Para ver la secuencia (opcional, puede ser muy larga)
            # print("      Secuencia Final:")
            # for ater in solucion_final_grasp['secuencia_aterrizajes']:
            #     print(f"        Avión {ater['avion_id']} en Pista {ater['pista']} a Tiempo {ater['tiempo']} (Costo Ind: {ater.get('costo_individual', 0):.2f})")

        # Prueba de Hill Climbing sobre solución determinista (para cubrir el requisito)
        print(f"\n--- Prueba de Hill Climbing sobre Greedy Determinista para: {nombre_caso_prueba}, {num_pistas_prueba} pista(s) ---")
        from scripts.greedy_deterministic import resolver as construir_solucion_determinista
        
        sol_gd = construir_solucion_determinista(datos_caso, num_pistas_prueba)
        print(f"  Resultado Greedy Determinista (GD):")
        eval_gd_inicial = evaluar_solucion_penalizada(sol_gd['secuencia_aterrizajes'], sol_gd['aviones_no_programados'], datos_caso, pen_lk, pen_sep, pen_no_prog)
        print(f"    Costo Base GD: {eval_gd_inicial['costo_base']:.2f}")
        print(f"    Costo Penalizado GD: {eval_gd_inicial['costo_penalizado']:.2f}")
        print(f"    Factible Estricto GD: {eval_gd_inicial['es_estrictamente_factible']}")
        if sol_gd['aviones_no_programados']:
            print(f"    Aviones No Programados GD: {sol_gd['aviones_no_programados']}")

        sol_hc_sobre_gd = hill_climbing_mejor_mejora(
            sol_gd, # Pasar el dict completo de la solución GD
            datos_caso,
            num_pistas_prueba,
            max_iter_hc_test * 2, # Más iteraciones para HC sobre GD si se desea
            pen_lk,
            pen_sep,
            pen_no_prog
        )
        print(f"  Resultado de HC sobre GD:")
        print(f"    Costo Base Final HC(GD): {sol_hc_sobre_gd['costo_total']:.2f}")
        print(f"    Costo Penalizado Final HC(GD): {sol_hc_sobre_gd['costo_penalizado']:.2f}")
        print(f"    Es Estrictamente Factible HC(GD): {sol_hc_sobre_gd['es_factible']}")


    else:
        print(f"No se pudieron cargar los datos para {nombre_caso_prueba}. No se puede ejecutar la prueba de GRASP.")

