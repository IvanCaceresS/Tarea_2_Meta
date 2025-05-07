# greedy_determinista.py
import copy # Para copiar listas de aviones sin modificar la original

def calcular_costo_aterrizaje(tiempo_aterrizaje, avion_info):
    """
    Calcula el costo de penalización para un avión aterrizando en un tiempo específico.
    """
    costo = 0
    if tiempo_aterrizaje < avion_info['P']:
        costo = (avion_info['P'] - tiempo_aterrizaje) * avion_info['cost_temprano']
    elif tiempo_aterrizaje > avion_info['P']:
        costo = (tiempo_aterrizaje - avion_info['P']) * avion_info['cost_tardio']
    return costo

def resolver(datos_del_caso, num_pistas):
    """
    Resuelve el problema de asignación de aterrizajes usando un algoritmo greedy determinista.

    Args:
        datos_del_caso (dict): Los datos parseados del archivo de caso.
        num_pistas (int): El número de pistas disponibles (1 o 2).

    Returns:
        dict: Un diccionario con la solución, e.g.:
            {
                'secuencia_aterrizajes': [
                    {'avion_id': id, 'pista': p, 'tiempo': t, 'costo_individual': c}, ...
                ],
                'costo_total': costo_total_solucion,
                'aviones_no_programados': [...] # Lista de IDs de aviones que no se pudieron programar
            }
    """
    # Crear una copia profunda para no modificar los datos originales fuera de esta función
    aviones = copy.deepcopy(datos_del_caso['aviones']) 
    tiempos_separacion = datos_del_caso['tiempos_separacion']
    num_aviones_total = datos_del_caso['num_aviones']

    # --- Criterio Greedy: Ordenar aviones ---
    # Ordenar por P_k, luego E_k, luego id para desempates.
    # Esto define el orden en que intentaremos programar los aviones.
    aviones_ordenados = sorted(aviones, key=lambda x: (x['P'], x['E'], x['id']))

    solucion = {
        'secuencia_aterrizajes': [],
        'costo_total': 0,
        'aviones_no_programados': []
    }

    # Estado de las pistas:
    # Para cada pista, guardamos el ID del último avión y su tiempo de aterrizaje.
    estado_pistas = [{'ultimo_avion_id': None, 'ultimo_tiempo_aterrizaje': -float('inf')} for _ in range(num_pistas)]

    aviones_programados_ids = set()

    for avion_actual_info in aviones_ordenados:
        avion_id_actual = avion_actual_info['id']

        mejor_opcion_para_avion_actual = {
            'pista_asignada': -1,
            'tiempo_aterrizaje_final': -1,
            'costo_penalizacion': float('inf'),
            'valida': False
        }

        # Intentar programar avion_actual en la mejor pista posible
        for pista_idx in range(num_pistas):
            # 1. Calcular el tiempo más temprano posible en esta pista
            tiempo_aterrizaje_min_pista = avion_actual_info['E']
            ultimo_avion_en_pista_id = estado_pistas[pista_idx]['ultimo_avion_id']
            if ultimo_avion_en_pista_id is not None:
                tiempo_separacion_requerido = tiempos_separacion[ultimo_avion_en_pista_id][avion_id_actual]
                tiempo_aterrizaje_min_pista = max(
                    tiempo_aterrizaje_min_pista,
                    estado_pistas[pista_idx]['ultimo_tiempo_aterrizaje'] + tiempo_separacion_requerido
                )
            
            # 2. Determinar el tiempo de aterrizaje objetivo (intentar P_k)
            tiempo_aterrizaje_propuesto = max(tiempo_aterrizaje_min_pista, avion_actual_info['P'])

            # 3. Ajustar si supera L_k o si P_k era muy temprano
            if tiempo_aterrizaje_propuesto > avion_actual_info['L']:
                if tiempo_aterrizaje_min_pista <= avion_actual_info['L']:
                    tiempo_aterrizaje_final_pista = tiempo_aterrizaje_min_pista
                else:
                    continue # No se puede programar en esta pista
            else:
                tiempo_aterrizaje_final_pista = tiempo_aterrizaje_propuesto

            # Calcular costo para esta opción válida
            costo_actual = calcular_costo_aterrizaje(tiempo_aterrizaje_final_pista, avion_actual_info)

            # Comparar con la mejor opción encontrada hasta ahora PARA ESTE AVIÓN
            if costo_actual < mejor_opcion_para_avion_actual['costo_penalizacion']:
                mejor_opcion_para_avion_actual['pista_asignada'] = pista_idx
                mejor_opcion_para_avion_actual['tiempo_aterrizaje_final'] = tiempo_aterrizaje_final_pista
                mejor_opcion_para_avion_actual['costo_penalizacion'] = costo_actual
                mejor_opcion_para_avion_actual['valida'] = True
            elif costo_actual == mejor_opcion_para_avion_actual['costo_penalizacion']:
                # Desempate: preferir la pista con el índice menor
                if mejor_opcion_para_avion_actual['valida'] and pista_idx < mejor_opcion_para_avion_actual['pista_asignada']:
                    mejor_opcion_para_avion_actual['pista_asignada'] = pista_idx
                    mejor_opcion_para_avion_actual['tiempo_aterrizaje_final'] = tiempo_aterrizaje_final_pista
                    # El costo es el mismo

        # Fin del bucle de pistas para avion_actual
        if mejor_opcion_para_avion_actual['valida']:
            p_asignada = mejor_opcion_para_avion_actual['pista_asignada']
            t_final = mejor_opcion_para_avion_actual['tiempo_aterrizaje_final']
            c_individual = mejor_opcion_para_avion_actual['costo_penalizacion']

            solucion['secuencia_aterrizajes'].append({
                'avion_id': avion_id_actual,
                'pista': p_asignada,
                'tiempo': t_final,
                'costo_individual': c_individual
            })
            solucion['costo_total'] += c_individual

            # Actualizar estado de la pista
            estado_pistas[p_asignada]['ultimo_avion_id'] = avion_id_actual
            estado_pistas[p_asignada]['ultimo_tiempo_aterrizaje'] = t_final
            
            aviones_programados_ids.add(avion_id_actual)
        else:
            # El avión no pudo ser programado en ninguna pista
            solucion['aviones_no_programados'].append(avion_id_actual)

    # Verificar si todos los aviones fueron programados
    if len(aviones_programados_ids) != num_aviones_total:
        # CORRECCIÓN: Usar 'aviones' (la copia local) en lugar de 'aviones_originales'
        aviones_faltantes_check = [avion['id'] for avion in aviones if avion['id'] not in aviones_programados_ids] 
        
        # Si la lista aviones_no_programados está vacía pero la verificación encuentra faltantes, actualizarla.
        if not solucion['aviones_no_programados'] and aviones_faltantes_check:
            solucion['aviones_no_programados'] = aviones_faltantes_check
        
        # Solo imprimir advertencia si realmente hay aviones no programados
        if solucion['aviones_no_programados']: 
            print(f"Advertencia GD: No todos los aviones pudieron ser programados. Faltantes: {solucion['aviones_no_programados']}")
    
    # Ordenar la secuencia final por tiempo de aterrizaje (opcional, buena práctica)
    solucion['secuencia_aterrizajes'].sort(key=lambda x: x['tiempo'])
    
    # Añadir clave 'es_factible' para consistencia con otros resolvedores
    # Una solución GD se considera factible si todos fueron programados y la lógica interna respetó E_k/L_k
    solucion['es_factible'] = (len(solucion['aviones_no_programados']) == 0)

    return solucion

if __name__ == '__main__':
    import sys
    import os
    # Añadir la carpeta raíz del proyecto al path para importar lector_cases
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
    
    from scripts.lector_cases import read_case 

    # --- Prueba con case1.txt y 1 pista ---
    ruta_case1 = os.path.join(os.path.dirname(__file__), '..', 'cases', 'case1.txt') 
    datos_case1 = read_case(ruta_case1)
    if datos_case1:
        print("\n--- Greedy Determinista: case1.txt, 1 Pista ---")
        solucion_gd_c1_p1 = resolver(datos_case1, num_pistas=1)
        print(f"Costo Total: {solucion_gd_c1_p1['costo_total']:.2f}") # Formateado
        print(f"Es Factible (según GD): {solucion_gd_c1_p1['es_factible']}")
        if solucion_gd_c1_p1['aviones_no_programados']:
            print(f"Aviones no programados: {solucion_gd_c1_p1['aviones_no_programados']}")

        # --- Prueba con case1.txt y 2 pistas ---
        print("\n--- Greedy Determinista: case1.txt, 2 Pistas ---")
        solucion_gd_c1_p2 = resolver(datos_case1, num_pistas=2)
        print(f"Costo Total: {solucion_gd_c1_p2['costo_total']:.2f}") # Formateado
        print(f"Es Factible (según GD): {solucion_gd_c1_p2['es_factible']}")
        if solucion_gd_c1_p2['aviones_no_programados']:
            print(f"Aviones no programados: {solucion_gd_c1_p2['aviones_no_programados']}")
    else:
        print("No se pudieron cargar datos para case1.txt para la prueba del greedy.")

