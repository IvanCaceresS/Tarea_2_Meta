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
    Implementa un algoritmo Greedy Estocástico usando RCL.

    Calcula el EAT Global basado en todos los aterrizajes previos,
    luego asigna a la mejor pista disponible. Retorna float('inf') si falla.

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
    runway_assignments = {} # {plane_id: pista_asignada} (Solo relevante para 2P)
    schedule = [] # Orden en que se seleccionan los aviones para aterrizar

    unscheduled = set(range(D))
    # Tiempo en que cada pista estará disponible (inicialmente 0)
    tiempo_libre_pista = [0.0] * num_runways
    # Historial de aterrizajes: lista de tuplas (id_avion, tiempo_aterrizaje)
    # Crucial para calcular el EAT Global correctamente
    historial_aterrizajes = []

    while unscheduled:
        # --- Paso 1 y 2: Identificar candidatos factibles y evaluar su 'bondad' (Pk) ---
        candidate_evaluations = [] # Lista de (Pk, plane_id) para ordenar y crear RCL

        for plane_id in unscheduled:
            # Datos del avión candidato
            E, P, L, Ce, Cl = planes_data[plane_id]

            # Calcular EAT Global para este candidato basado en TODOS los aviones YA programados
            eat_global = E # El tiempo más temprano es su propia ventana E
            is_feasible_based_on_separation = True # Asumir factible inicialmente

            # Verificar separaciones con TODOS los aviones ya en el historial
            for id_previo, tiempo_previo in historial_aterrizajes:
                # No necesitamos separación consigo mismo
                if id_previo == plane_id: continue

                try:
                    separacion_req = separations[id_previo][plane_id]
                except IndexError:
                    # Esto no debería ocurrir si D, planes_data y separations son consistentes
                    print(f"ERROR Fatal Seed {seed}: Índice fuera de rango al acceder separations[{id_previo}][{plane_id}]")
                    return [], {}, INFINITO_COSTO

                # Si la separación es imposible, este avión no puede ir después de id_previo
                if separacion_req == MARCADOR_INFINITO_ENTRADA:
                     is_feasible_based_on_separation = False
                     break # No puede ser candidato en esta iteración

                # Actualizar el EAT Global: T_j >= T_i + T_ij
                eat_global = max(eat_global, tiempo_previo + separacion_req)

            # Si una separación T[i][j]==INF lo hizo infactible, pasar al siguiente candidato
            if not is_feasible_based_on_separation:
                continue

            # Ahora, verificar si, dado el EAT Global, existe AL MENOS UNA pista
            # donde pueda aterrizar ANTES de su ventana L
            can_land_somewhere_before_L = False
            for pista_id in range(num_runways):
                # El tiempo potencial en esta pista es el máximo entre el EAT global
                # y el tiempo en que la pista específica está libre
                tiempo_potencial_en_pista = max(eat_global, tiempo_libre_pista[pista_id])

                # ¿Es este tiempo potencial menor o igual a la ventana L del avión?
                if tiempo_potencial_en_pista <= L:
                    can_land_somewhere_before_L = True
                    break # Suficiente, sabemos que es factible en al menos una pista

            # Si el avión es factible globalmente (separaciones) y localmente (ventana L en alguna pista)
            if can_land_somewhere_before_L:
                # Añadir a la lista para la RCL, usando Pk como criterio de "bondad"
                candidate_evaluations.append( (P, plane_id) ) # (Bondad=Pk, ID)

        # --- Fin del bucle 'for plane_id in unscheduled' ---

        # --- Verificar si hay candidatos factibles ---
        if not candidate_evaluations:
            if unscheduled: # Si aún quedan aviones, falló la planificación
                 print(f"ERROR (Estocástico {num_runways}P, Seed {seed}): ¡No se encontró candidato factible! No planificados: {unscheduled}")
                 return schedule, landing_times, INFINITO_COSTO # Retornar fallo
            else: # Si no quedan no planificados, terminamos exitosamente
                 break # Salir del bucle while

        # --- Paso 3: Construir RCL ---
        candidate_evaluations.sort() # Ordenar por Pk (menor P es mejor), luego por ID
        current_rcl_size = min(rcl_size, len(candidate_evaluations))
        rcl = candidate_evaluations[:current_rcl_size]

        # --- Paso 4: Elegir aleatoriamente de la RCL ---
        if not rcl: # Chequeo de seguridad
             print(f"Error Fatal Seed {seed}: RCL vacía inesperadamente.")
             return schedule, landing_times, INFINITO_COSTO
        chosen_pk, chosen_plane_id = random.choice(rcl) # SOLO se elige el ID

        # --- Paso 5: Recalcular tiempo/pista óptimos para el avión ELEGIDO ---
        E_chosen, P_chosen, L_chosen, Ce_chosen, Cl_chosen = planes_data[chosen_plane_id]

        # Recalcular EAT Global para el elegido basado en el historial ACTUAL
        eat_global_chosen = E_chosen
        for id_previo, tiempo_previo in historial_aterrizajes:
             # Ya sabemos que las separaciones != INF son posibles por el chequeo anterior
             separacion_req = separations[id_previo][chosen_plane_id]
             eat_global_chosen = max(eat_global_chosen, tiempo_previo + separacion_req)

        # Encontrar la mejor pista y tiempo para el avión elegido AHORA
        final_landing_time = INFINITO_COSTO
        chosen_runway = -1
        for pista_id in range(num_runways):
            tiempo_potencial = max(eat_global_chosen, tiempo_libre_pista[pista_id])
            # Verificar si es factible respecto a L
            if tiempo_potencial <= L_chosen:
                 # Si es un tiempo estrictamente mejor O es igual pero en una pista de índice menor
                 if tiempo_potencial < final_landing_time or \
                    (tiempo_potencial == final_landing_time and pista_id < chosen_runway):
                     final_landing_time = tiempo_potencial
                     chosen_runway = pista_id

        # Validar si se encontró una asignación (debería si estaba en candidate_evaluations)
        if chosen_runway == -1:
            # Esto indicaría un posible error lógico o condición extrema
            print(f"ERROR Lógico Fatal Seed {seed}: Avión {chosen_plane_id} (P={P_chosen}) elegido de RCL se volvió infactible al asignar pista. EAT={eat_global_chosen}, L={L_chosen}, Tiempos Pista={tiempo_libre_pista}")
            return schedule, landing_times, INFINITO_COSTO

        # --- Paso 6: Actualizar estado ---
        landing_times[chosen_plane_id] = final_landing_time
        runway_assignments[chosen_plane_id] = chosen_runway # Guardar asignación (útil para debug o análisis)
        schedule.append(chosen_plane_id)                 # Añadir a la secuencia de selección
        unscheduled.remove(chosen_plane_id)              # Quitar de pendientes

        # Actualizar el tiempo libre de la pista ELEGIDA
        tiempo_libre_pista[chosen_runway] = final_landing_time

        # Añadir al historial para calcular EAT en la siguiente iteración
        historial_aterrizajes.append((chosen_plane_id, final_landing_time))

    # --- Fin del bucle while ---

    # Calcular costo final SÓLO si se planificaron todos los aviones
    if len(schedule) == D:
        total_cost = calculate_total_cost(planes_data, schedule, landing_times)
        # El costo calculado podría ser infinito si los tiempos son enormes, aunque raro
        if total_cost == INFINITO_COSTO:
            return schedule, landing_times, INFINITO_COSTO
        else:
            return schedule, landing_times, total_cost
    else:
        # Si el bucle terminó pero no se programaron todos (ya manejado antes, pero por seguridad)
        return schedule, landing_times, INFINITO_COSTO