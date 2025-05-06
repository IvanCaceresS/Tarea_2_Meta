# scripts/greedy_estocastico.py
import random
import copy

def calcular_costo_aterrizaje(tiempo_aterrizaje, avion_info):
    costo = 0
    if tiempo_aterrizaje < avion_info['P']:
        costo = (avion_info['P'] - tiempo_aterrizaje) * avion_info['cost_temprano']
    elif tiempo_aterrizaje > avion_info['P']:
        costo = (tiempo_aterrizaje - avion_info['P']) * avion_info['cost_tardio']
    return costo

def resolver(datos_del_caso, num_pistas, semilla, parametro_rcl_alpha=0.1):
    random.seed(semilla) 

    aviones_originales = datos_del_caso['aviones']
    tiempos_separacion = datos_del_caso['tiempos_separacion']
    num_aviones_total = datos_del_caso['num_aviones']

    solucion = {
        'secuencia_aterrizajes': [],
        'costo_total': 0,
        'aviones_no_programados': [], # Debería quedar vacía
        'es_factible': True 
    }
    
    estado_pistas = [{'ultimo_avion_id': None, 'ultimo_tiempo_aterrizaje': -float('inf')} for _ in range(num_pistas)]
    aviones_a_programar = [copy.deepcopy(avion) for avion in aviones_originales] # Lista de los que faltan
    
    # El bucle principal ahora se asegura de que todos los aviones se procesen una vez
    # El orden de procesamiento vendrá de la RCL
    for _ in range(num_aviones_total): # Iterar D veces para asegurar que cada avión se intente programar
        if not aviones_a_programar: # Si ya no quedan aviones por alguna razón
            break
            
        # --- Construcción de la RCL ---
        candidatos_evaluados = []
        for avion_info_iter in aviones_a_programar: # Solo considerar los que aún no se han programado
            candidatos_evaluados.append({
                'avion_info': avion_info_iter,
                'P': avion_info_iter['P'], 
                'E': avion_info_iter['E'],
                'id': avion_info_iter['id'] 
            })

        if not candidatos_evaluados: # No debería pasar si aviones_a_programar no está vacío
            break 

        candidatos_evaluados.sort(key=lambda x: (x['P'], x['E'], x['id']))
        
        rcl_size = max(1, int(len(candidatos_evaluados) * parametro_rcl_alpha)) if parametro_rcl_alpha <= 1 and isinstance(parametro_rcl_alpha, float) else int(parametro_rcl_alpha)
        rcl_size = min(rcl_size, len(candidatos_evaluados))
        
        rcl = [c['avion_info'] for c in candidatos_evaluados[:rcl_size]]

        if not rcl: 
            if candidatos_evaluados: # Fallback si rcl_size es 0 por alguna razón
                 rcl = [candidatos_evaluados[0]['avion_info']] 
            else: # No hay más candidatos
                 break 

        avion_actual_info = random.choice(rcl)
        avion_id_actual = avion_actual_info['id']
        
        mejor_opcion_para_avion_actual = {
            'pista_asignada': -1,
            'tiempo_aterrizaje_final': float('inf'),
            'costo_penalizacion': float('inf')
        }

        # Forzar la programación: encontrar la pista que permita el aterrizaje más temprano legal
        for pista_idx in range(num_pistas):
            tiempo_min_legal_en_pista = avion_actual_info['E']
            ultimo_avion_en_pista_id = estado_pistas[pista_idx]['ultimo_avion_id']
            if ultimo_avion_en_pista_id is not None:
                tiempo_separacion_requerido = tiempos_separacion[ultimo_avion_en_pista_id][avion_id_actual]
                tiempo_min_legal_en_pista = max(
                    tiempo_min_legal_en_pista,
                    estado_pistas[pista_idx]['ultimo_tiempo_aterrizaje'] + tiempo_separacion_requerido
                )
            
            tiempo_aterrizaje_forzado_pista = tiempo_min_legal_en_pista
            costo_actual_en_pista = calcular_costo_aterrizaje(tiempo_aterrizaje_forzado_pista, avion_actual_info)

            if tiempo_aterrizaje_forzado_pista < mejor_opcion_para_avion_actual['tiempo_aterrizaje_final']:
                mejor_opcion_para_avion_actual['tiempo_aterrizaje_final'] = tiempo_aterrizaje_forzado_pista
                mejor_opcion_para_avion_actual['pista_asignada'] = pista_idx
                mejor_opcion_para_avion_actual['costo_penalizacion'] = costo_actual_en_pista
            elif tiempo_aterrizaje_forzado_pista == mejor_opcion_para_avion_actual['tiempo_aterrizaje_final']:
                if costo_actual_en_pista < mejor_opcion_para_avion_actual['costo_penalizacion']:
                    mejor_opcion_para_avion_actual['pista_asignada'] = pista_idx
                    mejor_opcion_para_avion_actual['costo_penalizacion'] = costo_actual_en_pista
                elif costo_actual_en_pista == mejor_opcion_para_avion_actual['costo_penalizacion']:
                    if random.choice([True, False]): # Desempate aleatorio de pista
                         mejor_opcion_para_avion_actual['pista_asignada'] = pista_idx
        
        p_asignada = mejor_opcion_para_avion_actual['pista_asignada']
        t_final = mejor_opcion_para_avion_actual['tiempo_aterrizaje_final']
        c_individual = mejor_opcion_para_avion_actual['costo_penalizacion']

        if p_asignada == -1: # Salvaguarda
            print(f"ERROR GE: Avión {avion_id_actual} no pudo ser forzado. Se añade a no programados.")
            solucion['aviones_no_programados'].append(avion_id_actual)
            solucion['es_factible'] = False 
            aviones_a_programar = [avion for avion in aviones_a_programar if avion['id'] != avion_id_actual]
            continue

        if t_final > avion_actual_info['L']:
            solucion['es_factible'] = False 

        solucion['secuencia_aterrizajes'].append({
            'avion_id': avion_id_actual,
            'pista': p_asignada,
            'tiempo': t_final,
            'costo_individual': c_individual
        })
        solucion['costo_total'] += c_individual

        estado_pistas[p_asignada]['ultimo_avion_id'] = avion_id_actual
        estado_pistas[p_asignada]['ultimo_tiempo_aterrizaje'] = t_final
        
        aviones_a_programar = [avion for avion in aviones_a_programar if avion['id'] != avion_id_actual]
        # aviones_programados_count ya no es necesario aquí, el bucle for _ in range(num_aviones_total) lo controla

    # --- POST-PROCESAMIENTO ---
    if len(solucion['secuencia_aterrizajes']) != num_aviones_total:
        solucion['es_factible'] = False
        ids_programados = {ater['avion_id'] for ater in solucion['secuencia_aterrizajes']}
        solucion['aviones_no_programados'] = sorted(list(set(avion['id'] for avion in aviones_originales) - ids_programados))
        if solucion['aviones_no_programados']:
             print(f"Advertencia GE: No todos los aviones fueron programados al final. Faltantes: {solucion['aviones_no_programados']}")
    else:
        solucion['aviones_no_programados'] = []

    solucion['secuencia_aterrizajes'].sort(key=lambda x: x['tiempo'])
    return solucion

# --- Bloque de prueba (if __name__ == '__main__':) ---
# ... (tu bloque de prueba se mantiene igual, pero ahora la 'solucion_actual' tendrá 'es_factible') ...
# ... y aviones_no_programados debería estar vacío siempre, a menos que haya un error crítico ...
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from scripts.lector_cases import read_case 
    # from scripts.verificador import verificar_solucion # Comentado para esta prueba

    ruta_case1 = os.path.join('..', 'cases', 'case1.txt') 
    datos_case1 = read_case(ruta_case1)

    if datos_case1:
        print("\n--- Greedy Estocástico (Forzando Programación): case1.txt ---")
        
        num_ejecuciones = 10
        k_rcl = 3 

        for num_pista_actual in [1, 2]:
            print(f"\n  Resultados para {num_pista_actual} pista(s):")
            costos_ejecuciones = []
            
            for i in range(num_ejecuciones):
                semilla = i 
                sol_actual = resolver(datos_case1, num_pistas=num_pista_actual, semilla=semilla, parametro_rcl_alpha=k_rcl)
                
                costo_display_val = sol_actual['costo_total']
                costo_display_str = f"{sol_actual['costo_total']:.2f}"
                if not sol_actual.get('es_factible', True): 
                    costo_display_str += " (INFACTIBLE - Lk violada o no todos programados)"
                
                print(f"    Ejecución {i+1} (semilla {semilla}): Costo = {costo_display_str}", end="")
                
                if sol_actual['aviones_no_programados']: 
                    print(f", No Programados (ERROR): {sol_actual['aviones_no_programados']}", end="")
                print() 

                # Usar el costo real para estadísticas si es factible, sino un valor muy alto
                costos_ejecuciones.append(costo_display_val if sol_actual.get('es_factible', True) else float('inf'))
            
            if costos_ejecuciones:
                costos_para_stats = [c for c in costos_ejecuciones if c != float('inf')] 
                if costos_para_stats:
                    print(f"    Mejor costo (de soluciones factibles): {min(costos_para_stats):.2f}")
                    print(f"    Costo promedio (de soluciones factibles): {sum(costos_para_stats)/len(costos_para_stats):.2f}")
                    print(f"    Peor costo (de soluciones factibles): {max(costos_para_stats):.2f}")
                else:
                    print(f"    No se encontraron soluciones factibles en {num_ejecuciones} ejecuciones.")
    else:
        print("No se pudieron cargar datos para case1.txt para la prueba del greedy estocástico.")