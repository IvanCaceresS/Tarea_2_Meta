import copy # Para copiar objetos y evitar modificar los originales
from scripts.calculate_cost import calculate_total_cost

# --- Constantes ---
INFINITO_COSTO = float('inf')
MARCADOR_INFINITO_ENTRADA = 99999 # Asegúrate que coincida con el usado en greedy_stochastic

def recalculate_landing_times(sequence, D, planes_data, separations, num_runways):
    """
    Recalcula los tiempos de aterrizaje más tempranos posibles para una secuencia dada.

    Args:
        sequence (list): Una lista de IDs de avión en el orden de aterrizaje propuesto.
        D (int): Número de aviones.
        planes_data (list): Datos originales [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).

    Returns:
        tuple: (landing_times, cost, is_feasible)
               - landing_times (dict): {plane_id: tiempo} si es factible, None si no.
               - cost (float): Costo total si es factible, INFINITO_COSTO si no.
               - is_feasible (bool): True si la secuencia es factible, False si no.
    """
    landing_times = {}
    if not sequence: # Secuencia vacía es factible con costo 0
        return {}, 0.0, True

    is_feasible = True

    if num_runways == 1:
        last_scheduled_id = None
        last_landing_time = -1
        for plane_id in sequence:
            E, P, L, Ce, Cl = planes_data[plane_id]
            min_start_time = E
            if last_scheduled_id is not None:
                # Verificar si la separación lo permite
                separation_needed = separations[last_scheduled_id][plane_id]
                if separation_needed == MARCADOR_INFINITO_ENTRADA:
                     # print(f"DEBUG HC Recalc 1P: Infeasible T[{last_scheduled_id}][{plane_id}]==INF")
                     is_feasible = False
                     break # Esta secuencia es infactible
                min_start_time = max(min_start_time, last_landing_time + separation_needed)

            # Verificar ventana L
            if min_start_time > L:
                 # print(f"DEBUG HC Recalc 1P: Infeasible {plane_id=}, {min_start_time=} > {L=}")
                 is_feasible = False
                 break # Esta secuencia es infactible

            # Asignar tiempo y actualizar estado para el siguiente
            landing_times[plane_id] = min_start_time
            last_scheduled_id = plane_id
            last_landing_time = min_start_time

    elif num_runways == 2:
        # Requiere mantener el estado de ambas pistas durante la recalculación
        runway_last_id = [None, None]
        runway_last_time = [-1, -1]
        # Nota: No necesitamos runway_assignments aquí, solo calculamos tiempos

        historial_aterrizajes = [] # Mantenemos (id, tiempo) para calcular EAT global

        for plane_id in sequence:
            E, P, L, Ce, Cl = planes_data[plane_id]

            # 1. Calcular EAT Global basado en separaciones con TODOS los previos
            eat_global = E
            for id_previo, tiempo_previo in historial_aterrizajes:
                 separation_needed = separations[id_previo][plane_id]
                 if separation_needed == MARCADOR_INFINITO_ENTRADA:
                      # print(f"DEBUG HC Recalc 2P: Infeasible T[{id_previo}][{plane_id}]==INF")
                      is_feasible = False
                      break # Secuencia infactible
                 eat_global = max(eat_global, tiempo_previo + separation_needed)
            if not is_feasible: break # Salir del bucle principal si ya es infactible

            # 2. Encontrar la mejor pista y tiempo para este avión
            best_time_for_plane = INFINITO_COSTO
            best_runway = -1
            for pista_id in range(num_runways):
                 tiempo_disponible_pista = runway_last_time[pista_id] # En 2P, la disponibilidad es solo el T último en ESA pista
                 tiempo_potencial = max(eat_global, tiempo_disponible_pista)

                 # Factibilidad L
                 if tiempo_potencial <= L:
                      # Mejor tiempo encontrado hasta ahora?
                      if tiempo_potencial < best_time_for_plane or \
                         (tiempo_potencial == best_time_for_plane and pista_id < best_runway):
                          best_time_for_plane = tiempo_potencial
                          best_runway = pista_id

            # 3. Verificar si se encontró una asignación factible
            if best_runway == -1:
                 # print(f"DEBUG HC Recalc 2P: Infeasible {plane_id=}, no se encontró pista/tiempo <= L ({L=})")
                 is_feasible = False
                 break # Esta secuencia es infactible

            # 4. Asignar y actualizar estado
            landing_times[plane_id] = best_time_for_plane
            runway_last_id[best_runway] = plane_id       # Actualiza el último en la pista asignada
            runway_last_time[best_runway] = best_time_for_plane # Actualiza tiempo libre de esa pista
            historial_aterrizajes.append((plane_id, best_time_for_plane)) # Añadir al historial global

    else:
        raise ValueError("Num pistas debe ser 1 o 2")

    # Calcular costo final si fue factible
    if is_feasible:
        final_cost = calculate_total_cost(planes_data, sequence, landing_times)
        return landing_times, final_cost, True
    else:
        # Si no fue factible, retornar None y costo infinito
        return None, INFINITO_COSTO, False


def hill_climbing_first_improvement(initial_schedule, initial_times, initial_cost,
                                     D, planes_data, separations, num_runways, max_iterations=1000):
    """
    Realiza Hill Climbing (Alguna Mejora) usando intercambio 2-opt.

    Args:
        initial_schedule (list): Secuencia inicial de IDs de avión.
        initial_times (dict): Tiempos de aterrizaje iniciales.
        initial_cost (float): Costo inicial.
        D (int): Número de aviones.
        planes_data (list): Datos originales [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        max_iterations (int): Límite de iteraciones para evitar ciclos infinitos (raro pero posible).


    Returns:
        tuple: (best_schedule, best_times, best_cost)
               La mejor solución encontrada (puede ser la inicial si no hubo mejora).
    """
    if D < 2: # No se pueden hacer intercambios
        return initial_schedule, initial_times, initial_cost

    current_schedule = list(initial_schedule) # Copiar para no modificar original
    current_times = dict(initial_times)     # Copiar
    current_cost = initial_cost

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        improved = False
        # Iterar sobre todos los pares posibles para intercambiar (i, j)
        for i in range(D):
            for j in range(i + 1, D):
                # Crear vecino intercambiando posiciones i y j
                neighbor_schedule = list(current_schedule)
                neighbor_schedule[i], neighbor_schedule[j] = neighbor_schedule[j], neighbor_schedule[i]

                # Recalcular tiempos, costo y factibilidad para el vecino
                neighbor_times, neighbor_cost, is_feasible = recalculate_landing_times(
                    neighbor_schedule, D, planes_data, separations, num_runways
                )

                # Si el vecino es factible Y mejora el costo actual
                if is_feasible and neighbor_cost < current_cost:
                    # Moverse a la primera mejora encontrada
                    current_schedule = neighbor_schedule
                    current_times = neighbor_times
                    current_cost = neighbor_cost
                    improved = True
                    # print(f"  HC Mejora encontrada en iter {iterations}, nuevo costo: {current_cost:.2f}") # Debug opcional
                    break # Salir del bucle j (First Improvement)
            if improved:
                break # Salir del bucle i y reiniciar búsqueda de vecinos desde la nueva solución

        # Si se completó un ciclo por todos los vecinos sin mejora, estamos en un óptimo local
        if not improved:
            # print(f"  HC Óptimo local alcanzado después de {iterations} iteraciones.") # Debug opcional
            break

    # Retornar la mejor solución encontrada
    return current_schedule, current_times, current_cost