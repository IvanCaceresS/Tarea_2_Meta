import random
from scripts.calculate_cost import calculate_total_cost

# --- Constantes ---
INFINITO_COSTO = float('inf')
# Asegúrate que este valor coincide con cómo lees/usas T[i][i]
# Si tu read_case_data pone los valores originales (90, 68), necesitarás
# o bien usar 99999 aquí y modificar read_case_data para que ponga 99999 en T[i][i]
# o usar un número muy grande aquí que sepas que nunca será una separación válida.
# Usaremos 99999 asumiendo que la diagonal T ya está manejada correctamente (o se ignora T[i][i]).
MARCADOR_INFINITO_ENTRADA = 99999

def solve_greedy_stochastic(D, planes_data, separations, num_runways, seed, rcl_size=3):
    """
    Implementa un algoritmo Greedy Estocástico paso a paso (CORREGIDO v3).

    Respeta TODAS las separaciones Tij anteriores y recalcula el tiempo
    óptimo DESPUÉS de seleccionar aleatoriamente de la RCL.

    Args:
        D (int): Número de aviones.
        planes_data (list): Lista de listas [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        seed (int): Semilla para el generador de números aleatorios.
        rcl_size (int): Tamaño máximo de la Lista Restringida de Candidatos (RCL).

    Returns:
        tuple: (schedule, landing_times, total_cost)
               - schedule (list): Secuencia de aterrizaje determinada.
               - landing_times (dict): Diccionario {plane_id: landing_time}.
               - total_cost (float): Costo total. float('inf') si falla.
    """
    random.seed(seed) # Fijar la semilla para esta ejecución

    # --- Inicialización ---
    landing_times = {} # {plane_id: tiempo_aterrizaje}
    runway_assignments = {} # {plane_id: pista_asignada}
    schedule = [] # Orden en que se seleccionan los aviones

    unscheduled = set(range(D))
    tiempo_libre_pista = [0.0] * num_runways
    # Historial de aterrizajes: (id_avion, tiempo_aterrizaje)
    # Es crucial para calcular el EAT correctamente
    historial_aterrizajes = []

    while unscheduled:
        # --- Paso 1 y 2: Identificar candidatos factibles y evaluar su 'bondad' (Pk) ---
        candidate_evaluations = [] # Lista de (Pk, plane_id) para ordenar y crear RCL

        for plane_id in unscheduled:
            # Datos del avión candidato
            E, P, L, Ce, Cl = planes_data[plane_id]

            # Calcular EAT Global para este candidato basado en TODOS los aviones YA programados
            eat_global = E
            is_feasible_based_on_separation = True # Bandera inicial
            for id_previo, tiempo_previo in historial_aterrizajes:
                try:
                    # Ignorar T[i][i] si está en el historial (no debería pasar)
                    if id_previo == plane_id: continue

                    separacion_req = separations[id_previo][plane_id]

                    # Si T[i][j] es el marcador infinito, este avión NO PUEDE ir después de id_previo
                    if separacion_req == MARCADOR_INFINITO_ENTRADA:
                         is_feasible_based_on_separation = False
                         # print(f"DEBUG Seed {seed}: Avión {plane_id} infactible por T[{id_previo}][{plane_id}]==INF")
                         break # No necesita chequear más separaciones para este candidato

                    # La restricción es T_j >= T_i + T_ij
                    eat_global = max(eat_global, tiempo_previo + separacion_req)

                except IndexError:
                    print(f"Error Fatal Seed {seed}: Índice fuera de rango separations[{id_previo}][{plane_id}]")
                    return [], {}, INFINITO_COSTO # Fallo crítico

            # Si una separación T[i][j]=INF lo hizo infactible, saltar al siguiente candidato
            if not is_feasible_based_on_separation:
                continue

            # Verificar si existe al menos una pista donde pueda aterrizar antes de L_k
            can_land_somewhere_before_L = False
            for pista_id in range(num_runways):
                # Tiempo si aterriza en esta pista (considerando EAT global y disponibilidad)
                tiempo_potencial_en_pista = max(eat_global, tiempo_libre_pista[pista_id])
                # ¿Es factible respecto a L_k?
                if tiempo_potencial_en_pista <= L:
                    can_land_somewhere_before_L = True
                    break # Suficiente con encontrar una pista factible

            # Si el avión es factible (respecto a separaciones T y ventana L en al menos una pista)
            if can_land_somewhere_before_L:
                # Añadir a la lista de candidatos con su criterio de bondad (usaremos P_k)
                candidate_evaluations.append( (P, plane_id) ) # (Bondad=Pk, ID)

        # --- Si no hay candidatos factibles, terminar ---
        if not candidate_evaluations:
            # Si aún quedan aviones no programados, es un fallo
            if unscheduled:
                 # print(f"  Advertencia (Seed {seed}): No se encontraron candidatos factibles. Restantes: {unscheduled}. Fallo.")
                 return schedule, landing_times, INFINITO_COSTO
            else:
                 # Todos programados, salimos del bucle (aunque no debería llegar aquí si D>0)
                 break

        # --- Paso 3: Construir RCL ---
        # Ordenar por P_k (menor es mejor), luego por ID para desempate
        candidate_evaluations.sort()
        # Tomar los 'rcl_size' mejores (o menos si no hay tantos)
        rcl = candidate_evaluations[:rcl_size]

        # --- Paso 4: Elegir aleatoriamente de la RCL ---
        if not rcl: # Seguridad
             print(f"Error Fatal Seed {seed}: RCL vacía inesperadamente (después de encontrar candidatos).")
             return schedule, landing_times, INFINITO_COSTO

        # >>> SOLO SE ELIGE EL ID del avión de la RCL <<<
        chosen_pk, chosen_plane_id = random.choice(rcl)

        # --- Paso 5: RECALCULAR tiempo/pista óptimos para el avión ELEGIDO ---
        # Necesitamos los datos del avión elegido
        E_chosen, P_chosen, L_chosen, Ce_chosen, Cl_chosen = planes_data[chosen_plane_id]

        # Recalcular EAT Global para el elegido basado en el historial ACTUAL
        # (Es importante hacerlo de nuevo por si el historial cambió implícitamente)
        eat_global_chosen = E_chosen
        for id_previo, tiempo_previo in historial_aterrizajes:
             # No necesitamos chequear T[i][j]==INF aquí porque ya se filtró antes
             # si el elegido estuviera en candidate_evaluations
             separacion_req = separations[id_previo][chosen_plane_id]
             eat_global_chosen = max(eat_global_chosen, tiempo_previo + separacion_req)

        # Encontrar el mejor tiempo y pista AHORA para el avión elegido
        final_landing_time = INFINITO_COSTO # Reiniciar para este cálculo
        chosen_runway = -1              # Reiniciar para este cálculo
        for pista_id in range(num_runways):
            tiempo_potencial = max(eat_global_chosen, tiempo_libre_pista[pista_id])
            # Chequear factibilidad L_k del elegido
            if tiempo_potencial <= L_chosen:
                 # Si es estrictamente mejor O igual pero pista con índice menor (para determinismo en empates)
                 if tiempo_potencial < final_landing_time or \
                    (tiempo_potencial == final_landing_time and pista_id < chosen_runway):
                     final_landing_time = tiempo_potencial
                     chosen_runway = pista_id

        # Validar si se encontró un tiempo (debería si pasó la factibilidad antes)
        if chosen_runway == -1:
            # Si esto ocurre, significa que un avión considerado factible antes,
            # ya no lo es. Podría pasar si las separaciones con nuevos aviones lo impiden.
            print(f"Error Fatal Seed {seed}: Avión {chosen_plane_id} elegido de RCL ({P_chosen=}) se volvió infactible al recalcular tiempo final. EAT={eat_global_chosen}, L={L_chosen}, Tiempos Pista={tiempo_libre_pista}")
            # Investigar por qué ocurrió esto. Puede ser un error lógico o una condición extrema del caso.
            # Devolver fallo para esta ejecución.
            return schedule, landing_times, INFINITO_COSTO

        # --- Paso 6: Actualizar estado ---
        # Usar los valores RECALCULADOS: final_landing_time y chosen_runway
        landing_times[chosen_plane_id] = final_landing_time
        runway_assignments[chosen_plane_id] = chosen_runway # Guardar pista asignada
        schedule.append(chosen_plane_id)                 # Añadir a la secuencia
        unscheduled.remove(chosen_plane_id)              # Quitar de pendientes

        # Actualizar tiempo libre de la pista ELEGIDA con el tiempo RECALCULADO
        tiempo_libre_pista[chosen_runway] = final_landing_time

        # Añadir al historial (id, tiempo_recalculado) para el siguiente paso
        historial_aterrizajes.append((chosen_plane_id, final_landing_time))

    # --- Fin del bucle while ---
    # Si el bucle terminó y 'unscheduled' está vacío, se programaron todos

    # Calcular costo final
    # Asegurarse que el schedule contiene todos los aviones si unscheduled está vacío
    if len(schedule) == D:
        total_cost = calculate_total_cost(planes_data, schedule, landing_times)
        # Verificar si el costo calculado es infinito (no debería si schedule está completo)
        if total_cost == INFINITO_COSTO:
             # print(f"Advertencia Seed {seed}: Schedule completo pero costo infinito.")
             return schedule, landing_times, INFINITO_COSTO
        else:
             return schedule, landing_times, total_cost
    else:
        # Si el bucle terminó pero no se programaron todos (raro, manejado antes)
        # print(f"Advertencia Seed {seed}: Bucle terminado pero no todos programados.")
        return schedule, landing_times, INFINITO_COSTO