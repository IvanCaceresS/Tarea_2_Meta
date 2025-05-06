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
    if secuencia_aterrizajes_lista: # Intentar determinar num_pistas desde la secuencia
        pistas_usadas = {ater['pista'] for ater in secuencia_aterrizajes_lista}
        if pistas_usadas:
            num_pistas = max(pistas_usadas) + 1
    
    if num_pistas == 0 and datos_del_caso.get('num_pistas_original', 0) > 0 : # Fallback si main.py pasa esta info
        num_pistas = datos_del_caso['num_pistas_original']
    elif num_pistas == 0 : # Si aún no se pudo determinar y no hay info, asumir 1 para evitar errores
        num_pistas = 1


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
    aterrizajes_por_pista = [[] for _ in range(num_pistas)]
    if num_pistas > 0:
        for ater in secuencia_ordenada:
            if 0 <= ater['pista'] < num_pistas:
                 aterrizajes_por_pista[ater['pista']].append(ater)

    for pista_idx in range(num_pistas):
        for i in range(len(aterrizajes_por_pista[pista_idx]) - 1):
            avion_i_ater = aterrizajes_por_pista[pista_idx][i]
            avion_j_ater = aterrizajes_por_pista[pista_idx][i+1]

            id_i = avion_i_ater['avion_id']
            id_j = avion_j_ater['avion_id']
            
            tiempo_sep_req = tiempos_separacion_matriz[id_i][id_j]
            
            if avion_j_ater['tiempo'] < avion_i_ater['tiempo'] + tiempo_sep_req:
                violaciones_sep_count += 1
                costo_penalizado_total += penalidad_separacion_fija

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
    tiempos_separacion_matriz = datos_del_caso['tiempos_separacion']
    
    mejor_tiempo_hallado = None
    menor_costo_individual = float('inf')

    tiempo_min_ater_actual = avion_info['E']
    tiempo_max_ater_actual = avion_info['L']

    posibles_tiempos_aterrizaje = []

    # Considerar aterrizar en P_k si es posible
    tiempo_objetivo_pk = avion_info['P']

    # Generar puntos de prueba: E_k, P_k, L_k y tiempos alrededor de otros aviones
    puntos_criticos_propios = {avion_info['E'], avion_info['P'], avion_info['L']}
    
    # Para cada gap (incluyendo antes del primero y después del último)
    # Gap antes del primer avión
    tiempo_inicio_gap = avion_info['E']
    tiempo_fin_gap = avion_info['L']
    if otros_aterrizajes_en_pista:
        tiempo_fin_gap = min(tiempo_fin_gap, otros_aterrizajes_en_pista[0]['tiempo'] - tiempos_separacion_matriz[avion_info['id']][otros_aterrizajes_en_pista[0]['avion_id']])
    
    if tiempo_inicio_gap <= tiempo_fin_gap:
        for t_critico_propio in puntos_criticos_propios:
            if tiempo_inicio_gap <= t_critico_propio <= tiempo_fin_gap:
                posibles_tiempos_aterrizaje.append(t_critico_propio)
        posibles_tiempos_aterrizaje.append(tiempo_inicio_gap) # El más temprano posible en el gap

    # Gaps entre aviones y después del último
    for i in range(len(otros_aterrizajes_en_pista) + 1):
        avion_prev = otros_aterrizajes_en_pista[i-1] if i > 0 else None
        avion_sig = otros_aterrizajes_en_pista[i] if i < len(otros_aterrizajes_en_pista) else None

        tiempo_inicio_gap = avion_info['E']
        if avion_prev:
            tiempo_inicio_gap = max(tiempo_inicio_gap, avion_prev['tiempo'] + tiempos_separacion_matriz[avion_prev['avion_id']][avion_info['id']])

        tiempo_fin_gap = avion_info['L']
        if avion_sig:
            tiempo_fin_gap = min(tiempo_fin_gap, avion_sig['tiempo'] - tiempos_separacion_matriz[avion_info['id']][avion_sig['avion_id']])
        
        if tiempo_inicio_gap <= tiempo_fin_gap:
            for t_critico_propio in puntos_criticos_propios:
                 if tiempo_inicio_gap <= t_critico_propio <= tiempo_fin_gap:
                    posibles_tiempos_aterrizaje.append(t_critico_propio)
            posibles_tiempos_aterrizaje.append(tiempo_inicio_gap)


    # Eliminar duplicados y ordenar
    posibles_tiempos_aterrizaje = sorted(list(set(pt for pt in posibles_tiempos_aterrizaje if avion_info['E'] <= pt <= avion_info['L'])))

    for t_cand in posibles_tiempos_aterrizaje:
        # Validar separaciones para este t_cand (esto ya debería estar implícito en cómo se calcularon los gaps)
        # pero una doble comprobación no hace daño.
        valido_con_prev = True
        if otros_aterrizajes_en_pista:
            # Chequear con el avión que aterrizaría inmediatamente antes en la pista ordenada
            idx_prev = -1
            for k, ater_existente in enumerate(otros_aterrizajes_en_pista):
                if ater_existente['tiempo'] < t_cand:
                    idx_prev = k
                else:
                    break
            if idx_prev != -1:
                avion_prev_en_pista = otros_aterrizajes_en_pista[idx_prev]
                if t_cand < avion_prev_en_pista['tiempo'] + tiempos_separacion_matriz[avion_prev_en_pista['avion_id']][avion_info['id']]:
                    valido_con_prev = False
            
            # Chequear con el avión que aterrizaría inmediatamente después
            idx_sig = -1
            for k, ater_existente in enumerate(otros_aterrizajes_en_pista):
                if ater_existente['tiempo'] > t_cand:
                    idx_sig = k
                    break
            if idx_sig != -1:
                avion_sig_en_pista = otros_aterrizajes_en_pista[idx_sig]
                if avion_sig_en_pista['tiempo'] < t_cand + tiempos_separacion_matriz[avion_info['id']][avion_sig_en_pista['avion_id']]:
                     valido_con_prev = False # Reutilizo la variable, debería ser valido_con_sig

        if valido_con_prev: # Asumiendo que la validación de separación es completa
            costo_cand = calcular_costo_individual_avion(t_cand, avion_info)
            if costo_cand < menor_costo_individual:
                menor_costo_individual = costo_cand
                mejor_tiempo_hallado = t_cand
            elif costo_cand == menor_costo_individual:
                if mejor_tiempo_hallado is None or t_cand < mejor_tiempo_hallado: # Preferir más temprano en caso de empate
                    mejor_tiempo_hallado = t_cand
                         
    return mejor_tiempo_hallado, menor_costo_individual


def hill_climbing_mejor_mejora(solucion_inicial_dict, datos_del_caso, num_pistas_hc, max_iter_sin_mejora_hc, penalidad_lk, penalidad_sep, penalidad_no_prog):
    mejor_solucion_hc_actual = copy.deepcopy(solucion_inicial_dict) # Trabajar con el dict completo

    eval_actual = evaluar_solucion_penalizada(
        mejor_solucion_hc_actual['secuencia_aterrizajes'], 
        mejor_solucion_hc_actual.get('aviones_no_programados', []), 
        datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog
    )
    costo_actual_penalizado = eval_actual['costo_penalizado']

    iter_sin_mejora_actual = 0
    
    # Asegurarse de que HC solo opera sobre aviones programados
    if not mejor_solucion_hc_actual['secuencia_aterrizajes']:
        # print("      HC: Solución inicial sin aviones programados. No se puede aplicar HC.")
        # Devolver la evaluación de la solución inicial (que será muy penalizada si hay no programados)
        return {
            'secuencia_aterrizajes': [],
            'costo_total': eval_actual['costo_base'], 
            'costo_penalizado': eval_actual['costo_penalizado'], 
            'es_factible': eval_actual['es_estrictamente_factible'], 
            'aviones_no_programados': mejor_solucion_hc_actual.get('aviones_no_programados', []),
            'violaciones_lk_count': eval_actual['violaciones_lk_count'],
            'violaciones_sep_count': eval_actual['violaciones_sep_count'],
            'violaciones_no_prog_count': eval_actual['violaciones_no_prog_count']
        }

    num_aviones_en_sol = len(mejor_solucion_hc_actual['secuencia_aterrizajes'])

    while iter_sin_mejora_actual < max_iter_sin_mejora_hc:
        mejor_movimiento_info = {
            'costo_penalizado_vecino': costo_actual_penalizado,
            'secuencia_vecina_lista': None,
            'eval_vecina': None
        }
        
        for idx_avion_a_mover in range(num_aviones_en_sol):
            avion_movido_original_ater = mejor_solucion_hc_actual['secuencia_aterrizajes'][idx_avion_a_mover]
            avion_movido_id = avion_movido_original_ater['avion_id']
            
            # Buscar el dict completo del avión en datos_del_caso
            avion_movido_info = next((avion for avion in datos_del_caso['aviones'] if avion['id'] == avion_movido_id), None)
            if not avion_movido_info: continue # Salvaguarda

            secuencia_temp_sin_avion = [ater for i, ater in enumerate(mejor_solucion_hc_actual['secuencia_aterrizajes']) if i != idx_avion_a_mover]
            
            for pista_destino_idx in range(num_pistas_hc):
                aterrizajes_en_pista_destino = sorted(
                    [ater for ater in secuencia_temp_sin_avion if ater['pista'] == pista_destino_idx],
                    key=lambda x: x['tiempo']
                )
                
                tiempo_opt_pista, costo_ind_opt_pista = _encontrar_mejor_tiempo_insercion_en_pista(
                    avion_movido_info, pista_destino_idx, aterrizajes_en_pista_destino, datos_del_caso
                )

                if tiempo_opt_pista is not None:
                    nueva_secuencia_vecina_lista = copy.deepcopy(secuencia_temp_sin_avion)
                    nueva_secuencia_vecina_lista.append({
                        'avion_id': avion_movido_id,
                        'pista': pista_destino_idx,
                        'tiempo': tiempo_opt_pista,
                        'costo_individual': costo_ind_opt_pista 
                    })
                    nueva_secuencia_vecina_lista.sort(key=lambda x: x['tiempo'])
                    
                    eval_vecina = evaluar_solucion_penalizada(
                        nueva_secuencia_vecina_lista, 
                        mejor_solucion_hc_actual.get('aviones_no_programados', []), # HC no cambia los no programados
                        datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog
                    )
                    
                    if eval_vecina['costo_penalizado'] < mejor_movimiento_info['costo_penalizado_vecino']:
                        mejor_movimiento_info['costo_penalizado_vecino'] = eval_vecina['costo_penalizado']
                        mejor_movimiento_info['secuencia_vecina_lista'] = nueva_secuencia_vecina_lista
                        mejor_movimiento_info['eval_vecina'] = eval_vecina
        
        if mejor_movimiento_info['secuencia_vecina_lista'] is not None and mejor_movimiento_info['costo_penalizado_vecino'] < costo_actual_penalizado:
            mejor_solucion_hc_actual['secuencia_aterrizajes'] = mejor_movimiento_info['secuencia_vecina_lista']
            eval_actual = mejor_movimiento_info['eval_vecina']
            costo_actual_penalizado = eval_actual['costo_penalizado']
            # Los aviones no programados no cambian por HC, se mantienen de la solución inicial
            # La factibilidad y costos sí cambian.
            iter_sin_mejora_actual = 0 
        else:
            iter_sin_mejora_actual += 1 

    return {
        'secuencia_aterrizajes': mejor_solucion_hc_actual['secuencia_aterrizajes'],
        'costo_total': eval_actual['costo_base'], 
        'costo_penalizado': costo_actual_penalizado, 
        'es_factible': eval_actual['es_estrictamente_factible'], 
        'aviones_no_programados': mejor_solucion_hc_actual.get('aviones_no_programados', []),
        'violaciones_lk_count': eval_actual['violaciones_lk_count'],
        'violaciones_sep_count': eval_actual['violaciones_sep_count'],
        'violaciones_no_prog_count': eval_actual['violaciones_no_prog_count']
    }


def grasp_resolver(datos_del_caso, num_pistas_grasp, num_iteraciones_grasp, 
                   semilla_inicial_grasp, parametro_rcl_ge, 
                   max_iter_sin_mejora_hc, 
                   penalidad_lk, penalidad_sep, penalidad_no_prog,
                   solucion_inicial_para_primera_iter=None): # NUEVO PARÁMETRO
    """
    Resuelve el problema usando GRASP con Hill Climbing (Mejor Mejora).
    Si se provee 'solucion_inicial_para_primera_iter', se usa en la primera iteración de GRASP.
    """
    mejor_solucion_global_grasp_dict = {
        'secuencia_aterrizajes': [], 
        'costo_total': float('inf'), 
        'costo_penalizado': float('inf'), 
        'es_factible': False, 
        'aviones_no_programados': [i for i in range(datos_del_caso['num_aviones'])],
        'violaciones_lk_count': datos_del_caso['num_aviones'], # Peor caso inicial
        'violaciones_sep_count': datos_del_caso['num_aviones'] * datos_del_caso['num_aviones'], # Peor caso
        'violaciones_no_prog_count': datos_del_caso['num_aviones']
    }
    
    for i_grasp in range(num_iteraciones_grasp):
        semilla_ge_actual = semilla_inicial_grasp + i_grasp 
        
        solucion_construida_dict = None
        # print(f"  GRASP Iter {i_grasp + 1}/{num_iteraciones_grasp}. Semilla GE actual: {semilla_ge_actual}")

        if i_grasp == 0 and solucion_inicial_para_primera_iter is not None:
            # print(f"    Usando solución inicial proporcionada para la primera iteración de GRASP.")
            solucion_construida_dict = copy.deepcopy(solucion_inicial_para_primera_iter)
        else:
            # print(f"    Construyendo nueva solución con Greedy Estocástico.")
            solucion_construida_dict = construir_solucion_stochastic(
                datos_del_caso, 
                num_pistas_grasp, 
                semilla_ge_actual, 
                parametro_rcl_alpha=parametro_rcl_ge
            )
        
        if not solucion_construida_dict or not solucion_construida_dict.get('secuencia_aterrizajes'):
            # print(f"    GRASP Iter {i_grasp + 1}: Solución construida inválida o vacía. Saltando HC.")
            # Evaluar la solución construida (probablemente muy penalizada)
            current_eval = evaluar_solucion_penalizada(
                solucion_construida_dict.get('secuencia_aterrizajes', []),
                solucion_construida_dict.get('aviones_no_programados', list(range(datos_del_caso['num_aviones']))),
                datos_del_caso, penalidad_lk, penalidad_sep, penalidad_no_prog
            )
            current_costo_penalizado = current_eval['costo_penalizado']
            solucion_mejorada_hc_dict = {
                'secuencia_aterrizajes': solucion_construida_dict.get('secuencia_aterrizajes', []),
                'costo_total': current_eval['costo_base'],
                'costo_penalizado': current_costo_penalizado,
                'es_factible': current_eval['es_estrictamente_factible'],
                'aviones_no_programados': solucion_construida_dict.get('aviones_no_programados', list(range(datos_del_caso['num_aviones']))),
                'violaciones_lk_count': current_eval['violaciones_lk_count'],
                'violaciones_sep_count': current_eval['violaciones_sep_count'],
                'violaciones_no_prog_count': current_eval['violaciones_no_prog_count']

            }
        else:
            # print(f"    Aplicando Hill Climbing a la solución construida.")
            solucion_mejorada_hc_dict = hill_climbing_mejor_mejora(
                solucion_construida_dict, 
                datos_del_caso, 
                num_pistas_grasp, 
                max_iter_sin_mejora_hc,
                penalidad_lk,
                penalidad_sep,
                penalidad_no_prog
            )
        
        costo_hc_penalizado = solucion_mejorada_hc_dict['costo_penalizado']
        
        # Actualizar la mejor solución global de GRASP
        if costo_hc_penalizado < mejor_solucion_global_grasp_dict['costo_penalizado']:
            mejor_solucion_global_grasp_dict = copy.deepcopy(solucion_mejorada_hc_dict)
            # print(f"    GRASP Iter {i_grasp + 1}: Nueva mejor solución global. Penalizado: {costo_hc_penalizado:.2f}, Base: {mejor_solucion_global_grasp_dict['costo_total']:.2f}, Factible: {mejor_solucion_global_grasp_dict['es_factible']}")
        elif costo_hc_penalizado == mejor_solucion_global_grasp_dict['costo_penalizado']:
            # Si costos penalizados son iguales, preferir la que sea estrictamente factible
            if solucion_mejorada_hc_dict['es_factible'] and not mejor_solucion_global_grasp_dict['es_factible']:
                mejor_solucion_global_grasp_dict = copy.deepcopy(solucion_mejorada_hc_dict)
            # Si ambas son factibles (o ambas infactibles), preferir la de menor costo base
            elif solucion_mejorada_hc_dict['es_factible'] == mejor_solucion_global_grasp_dict['es_factible'] and \
                 solucion_mejorada_hc_dict['costo_total'] < mejor_solucion_global_grasp_dict['costo_total']:
                mejor_solucion_global_grasp_dict = copy.deepcopy(solucion_mejorada_hc_dict)
        
    return mejor_solucion_global_grasp_dict


if __name__ == '__main__':
    import sys
    import os
    directorio_base_proyecto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.append(directorio_base_proyecto)
    
    from scripts.lector_cases import read_case 

    nombre_caso_prueba = 'case1.txt' 
    num_pistas_prueba = 1       
    
    iteraciones_grasp_test_config = [1, 5] # [0 restarts efectivos, 4 restarts efectivos]
    semilla_grasp_inicial_test = 0
    rcl_param_test = 3 
    max_iter_hc_test = 10 
    
    pen_lk = 1000.0
    pen_sep = 5000.0
    pen_no_prog = 100000.0

    ruta_completa_caso = os.path.join(directorio_base_proyecto, 'cases', nombre_caso_prueba)
    print(f"Cargando datos desde: {ruta_completa_caso}")
    datos_caso = read_case(ruta_completa_caso)

    if datos_caso:
        print(f"\n--- Prueba GRASP + HC para: {nombre_caso_prueba}, {num_pistas_prueba} pista(s) ---")
        
        # Prueba 1: GRASP con solución inicial determinista (0 restarts efectivos)
        print(f"\n  GRASP usando GD como inicio (0 restarts efectivos):")
        from scripts.greedy_deterministic import resolver as construir_solucion_determinista
        sol_gd_inicial = construir_solucion_determinista(datos_caso, num_pistas_prueba)
        
        solucion_grasp_gd_inicio = grasp_resolver(
            datos_del_caso=datos_caso,
            num_pistas_grasp=num_pistas_prueba,
            num_iteraciones_grasp=1, # Solo procesa la solución inicial
            semilla_inicial_grasp=semilla_grasp_inicial_test, # No se usará para construir si se da sol_inicial
            parametro_rcl_ge=rcl_param_test, # No se usará para construir si se da sol_inicial
            max_iter_sin_mejora_hc=max_iter_hc_test,
            penalidad_lk=pen_lk, penalidad_sep=pen_sep, penalidad_no_prog=pen_no_prog,
            solucion_inicial_para_primera_iter=sol_gd_inicial
        )
        print(f"    Resultado GRASP (inicio GD):")
        print(f"      Costo Base: {solucion_grasp_gd_inicio['costo_total']:.2f}, Penalizado: {solucion_grasp_gd_inicio['costo_penalizado']:.2f}, Factible: {solucion_grasp_gd_inicio['es_factible']}")

        # Prueba 2: GRASP con solución inicial estocástica y algunos restarts
        print(f"\n  GRASP con inicio Estocástico (ej. 5 iteraciones GRASP):")
        sol_ge_inicial_para_grasp = construir_solucion_stochastic(datos_caso, num_pistas_prueba, semilla=99, parametro_rcl_alpha=rcl_param_test)

        for num_iter_cfg in iteraciones_grasp_test_config:
            print(f"    GRASP con {num_iter_cfg} iteraciones (inicio GE semilla 99):")
            solucion_grasp_ge_inicio = grasp_resolver(
                datos_del_caso=datos_caso,
                num_pistas_grasp=num_pistas_prueba,
                num_iteraciones_grasp=num_iter_cfg, 
                semilla_inicial_grasp=semilla_grasp_inicial_test, # Semilla para las iteraciones > 1
                parametro_rcl_ge=rcl_param_test, 
                max_iter_sin_mejora_hc=max_iter_hc_test,
                penalidad_lk=pen_lk, penalidad_sep=pen_sep, penalidad_no_prog=pen_no_prog,
                solucion_inicial_para_primera_iter=sol_ge_inicial_para_grasp if num_iter_cfg > 0 else None # Pasa la sol GE si es la primera iter de un GRASP mayor
            )
            print(f"      Costo Base: {solucion_grasp_ge_inicio['costo_total']:.2f}, Penalizado: {solucion_grasp_ge_inicio['costo_penalizado']:.2f}, Factible: {solucion_grasp_ge_inicio['es_factible']}")

    else:
        print(f"No se pudieron cargar datos para {nombre_caso_prueba}.")

