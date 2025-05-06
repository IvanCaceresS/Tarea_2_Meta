# scripts/verificador.py

def calcular_costo_aterrizaje(tiempo_aterrizaje, avion_info):
    """
    Calcula el costo de penalización para un avión aterrizando en un tiempo específico.
    (Duplicada de greedy_determinista.py para que el verificador sea autocontenido,
     o podrías importarla si prefieres no duplicar).
    """
    costo = 0
    if tiempo_aterrizaje < avion_info['P']:
        costo = (avion_info['P'] - tiempo_aterrizaje) * avion_info['cost_temprano']
    elif tiempo_aterrizaje > avion_info['P']:
        costo = (tiempo_aterrizaje - avion_info['P']) * avion_info['cost_tardio']
    return costo

def verificar_solucion(solucion_propuesta, datos_del_caso, num_pistas_usadas):
    """
    Verifica si una solución de aterrizaje es válida y consistente.

    Args:
        solucion_propuesta (dict): La solución generada por un algoritmo.
                                   Debe contener 'secuencia_aterrizajes' y 'costo_total'.
        datos_del_caso (dict): Los datos originales del problema.
        num_pistas_usadas (int): El número de pistas que se asumieron para generar la solución.

    Returns:
        bool: True si la solución es válida, False en caso contrario.
    """
    if not solucion_propuesta or 'secuencia_aterrizajes' not in solucion_propuesta or 'costo_total' not in solucion_propuesta:
        print("Error de verificación: Solución propuesta está vacía o mal formada (faltan claves 'secuencia_aterrizajes' o 'costo_total').")
        return False

    # Ordenar por tiempo para facilitar la verificación de separaciones en pista
    # Es importante que la secuencia ya venga ordenada o se ordene aquí de la misma manera
    # que el algoritmo la procesaría para actualizar el estado de las pistas.
    # Si el algoritmo ya la devuelve ordenada por tiempo, mejor. Sino, ordenar aquí.
    try:
        secuencia = sorted(solucion_propuesta['secuencia_aterrizajes'], key=lambda x: x['tiempo'])
    except TypeError as e:
        print(f"Error de verificación: Problema al ordenar la secuencia de aterrizajes. ¿Faltan tiempos? Detalle: {e}")
        # Imprimir algunos elementos de la secuencia para depurar
        # for item in solucion_propuesta.get('secuencia_aterrizajes', [])[:5]:
        #     print(item)
        return False


    aviones_data_dict = {avion['id']: avion for avion in datos_del_caso['aviones']}
    tiempos_separacion = datos_del_caso['tiempos_separacion']
    num_aviones_total = datos_del_caso['num_aviones']

    # 1. Verificar que todos los aviones del caso estén en la solución (si aplica, o los que se pudieron programar)
    ids_aviones_en_solucion = {ater['avion_id'] for ater in secuencia}
    
    if 'aviones_no_programados' in solucion_propuesta:
        num_programados_esperados = num_aviones_total - len(solucion_propuesta['aviones_no_programados'])
    else: # Si la clave no existe, asumimos que todos debieron ser programados
        num_programados_esperados = num_aviones_total 

    if len(ids_aviones_en_solucion) != num_programados_esperados:
        print(f"Error de verificación: Número de aviones únicos en secuencia ({len(ids_aviones_en_solucion)}) "
              f"no coincide con los esperados a programar ({num_programados_esperados}).")
        # Considerar si los aviones no programados están bien gestionados.
        # return False # Esto depende de si es un error fatal o no.

    # Estado de las pistas para verificación
    estado_pistas_verif = [{'ultimo_avion_id': None, 'ultimo_tiempo_aterrizaje': -float('inf')} for _ in range(num_pistas_usadas)]
    costo_total_recalculado = 0.0

    for ater_actual in secuencia:
        avion_id_actual = ater_actual['avion_id']
        tiempo_actual = ater_actual['tiempo']
        pista_actual = ater_actual['pista']
        costo_individual_solucion = ater_actual.get('costo_individual', 0.0) # Obtener con default

        if pista_actual < 0 or pista_actual >= num_pistas_usadas:
            print(f"Error de verificación (Avión {avion_id_actual}): Pista asignada {pista_actual} fuera de rango [0, {num_pistas_usadas-1}]")
            return False

        if avion_id_actual not in aviones_data_dict:
            print(f"Error de verificación: Avión ID {avion_id_actual} en solución no existe en datos del caso.")
            return False
        
        info_avion_actual = aviones_data_dict[avion_id_actual]

        # 2. Verificar ventana de tiempo E_k y L_k
        if not (info_avion_actual['E'] <= tiempo_actual <= info_avion_actual['L']):
            print(f"Error de verificación (Avión {avion_id_actual}): Tiempo de aterrizaje {tiempo_actual} "
                  f"fuera de ventana [{info_avion_actual['E']}, {info_avion_actual['L']}].")
            return False

        # 3. Verificar separación con el avión anterior EN LA MISMA PISTA
        ultimo_avion_en_pista_id = estado_pistas_verif[pista_actual]['ultimo_avion_id']
        ultimo_tiempo_en_pista = estado_pistas_verif[pista_actual]['ultimo_tiempo_aterrizaje']

        if ultimo_avion_en_pista_id is not None:
            # Asegurarse que los IDs son válidos para la matriz de separación
            if not (0 <= ultimo_avion_en_pista_id < num_aviones_total and 0 <= avion_id_actual < num_aviones_total):
                 print(f"Error de verificación: IDs de avión ({ultimo_avion_en_pista_id}, {avion_id_actual}) fuera de rango para la matriz de separación.")
                 return False

            sep_requerida = tiempos_separacion[ultimo_avion_en_pista_id][avion_id_actual]
            if tiempo_actual < ultimo_tiempo_en_pista + sep_requerida:
                print(f"Error de verificación (Avión {avion_id_actual} en pista {pista_actual}): Violación de separación con avión {ultimo_avion_en_pista_id}.")
                print(f"  T_actual={tiempo_actual}, T_prev={ultimo_tiempo_en_pista}, Sep_Requerida={sep_requerida}. "
                      f"Debe ser T_actual >= {ultimo_tiempo_en_pista + sep_requerida}")
                return False
        
        estado_pistas_verif[pista_actual]['ultimo_avion_id'] = avion_id_actual
        estado_pistas_verif[pista_actual]['ultimo_tiempo_aterrizaje'] = tiempo_actual

        # 4. Recalcular costo individual y sumarlo
        costo_individual_recalculado = calcular_costo_aterrizaje(tiempo_actual, info_avion_actual)
        
        if abs(costo_individual_recalculado - costo_individual_solucion) > 1e-5 : # Comparar floats con tolerancia
             print(f"Advertencia de verificación (Avión {avion_id_actual}): Costo individual no coincide exactamente. "
                   f"Calculado: {costo_individual_recalculado:.2f}, En Solución: {costo_individual_solucion:.2f}")
             # Decidir si esto es un error fatal o solo una advertencia de precisión.
             # Para la tarea, es bueno que coincidan lo más posible.

        costo_total_recalculado += costo_individual_recalculado

    # 5. Verificar costo total
    if abs(costo_total_recalculado - solucion_propuesta['costo_total']) > 1e-2: # Tolerancia un poco mayor para sumas acumuladas
        print(f"Error de verificación: Costo total no coincide. "
              f"Recalculado: {costo_total_recalculado:.2f}, En Solución: {solucion_propuesta['costo_total']:.2f}")
        return False
        
    print(f"Verificación de la solución ({num_pistas_usadas} pista(s)): OK. Restricciones cumplidas y costos consistentes.")
    return True