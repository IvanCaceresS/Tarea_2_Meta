# hill_climbing.py (recalculate_landing_times Corregido)
import copy
try:
    # Adjust path if necessary
    from scripts.calculate_cost import calculate_total_cost
except ImportError:
    try:
        from calculate_cost import calculate_total_cost
    except ImportError:
        print("ERROR FATAL HC: calculate_cost no encontrado.")
        def calculate_total_cost(*args, **kwargs): return float('inf')

# --- Constantes ---
INFINITO_COSTO = float('inf')

def recalculate_landing_times(sequence, D, planes_data, separations, num_runways):
    """
    Recalcula los tiempos de aterrizaje más tempranos posibles para una secuencia dada,
    respetando E_k, L_k y las separaciones tau_ij EN LA MISMA PISTA.

    Args:
        sequence (list): Una lista de IDs de avión en el orden de aterrizaje propuesto.
        D (int): Número de aviones.
        planes_data (list): Datos originales [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).

    Returns:
        tuple: (landing_times, cost, is_feasible)
               - landing_times (dict): {plane_id: tiempo} si es factible, None si no.
               - cost (float): Costo total (calculado con costo original) si es factible, INFINITO_COSTO si no.
               - is_feasible (bool): True si la secuencia es factible, False si no.
    """
    landing_times = {}
    if not sequence:
        return {}, 0.0, True

    is_feasible = True

    if num_runways == 1:
        last_scheduled_id = None
        last_landing_time = -1.0 # Use float for time
        for plane_id in sequence:
            # Validate plane_id
            if not (0 <= plane_id < len(planes_data)):
                print(f"ERROR Recalc 1P: ID inválido {plane_id}.")
                return None, INFINITO_COSTO, False

            E, P, L, Ce, Cl = planes_data[plane_id]
            min_start_time = float(E) # Ensure float

            if last_scheduled_id is not None:
                 # Validate indices for separations
                if not (0 <= last_scheduled_id < D and 0 <= plane_id < D):
                    print(f"ERROR Recalc 1P: Índices inválidos para separación ({last_scheduled_id}, {plane_id}).")
                    return None, INFINITO_COSTO, False
                if last_scheduled_id >= len(separations) or plane_id >= len(separations[last_scheduled_id]):
                     print(f"ERROR Recalc 1P: Índice fuera de rango para separations[{last_scheduled_id}][{plane_id}].")
                     return None, INFINITO_COSTO, False

                separation_needed = float(separations[last_scheduled_id][plane_id])
                min_start_time = max(min_start_time, last_landing_time + separation_needed)

            if min_start_time > L:
                 is_feasible = False
                 # print(f"DEBUG Recalc 1P: Infeasible Lk for {plane_id}. T={min_start_time} > L={L}") # Debug
                 break

            landing_times[plane_id] = min_start_time
            last_scheduled_id = plane_id
            last_landing_time = min_start_time

    elif num_runways == 2:
        runway_last_id = [None, None]
        runway_last_time = [-1.0, -1.0] # Use float

        for plane_id in sequence:
            # Validate plane_id
            if not (0 <= plane_id < len(planes_data)):
                print(f"ERROR Recalc 2P: ID inválido {plane_id}.")
                return None, INFINITO_COSTO, False

            E, P, L, Ce, Cl = planes_data[plane_id]
            earliest_possible_on_runway = [float(E), float(E)] # Earliest start based on E_k

            # Calculate earliest start time considering separation ON EACH RUNWAY
            for r_idx in range(num_runways):
                last_id_on_runway = runway_last_id[r_idx]
                if last_id_on_runway is not None:
                    # Validate indices for separations
                    if not (0 <= last_id_on_runway < D and 0 <= plane_id < D):
                        print(f"ERROR Recalc 2P: Índices inválidos para separación ({last_id_on_runway}, {plane_id}).")
                        return None, INFINITO_COSTO, False
                    if last_id_on_runway >= len(separations) or plane_id >= len(separations[last_id_on_runway]):
                         print(f"ERROR Recalc 2P: Índice fuera de rango para separations[{last_id_on_runway}][{plane_id}].")
                         return None, INFINITO_COSTO, False

                    separation_needed = float(separations[last_id_on_runway][plane_id])
                    earliest_possible_on_runway[r_idx] = max(
                        earliest_possible_on_runway[r_idx],
                        runway_last_time[r_idx] + separation_needed
                    )

            # Find the best feasible assignment (earliest time respecting L)
            best_time_for_plane = INFINITO_COSTO
            best_runway = -1

            for r_idx in range(num_runways):
                potential_time = earliest_possible_on_runway[r_idx]
                if potential_time <= L: # Check feasibility regarding L
                    if potential_time < best_time_for_plane:
                        best_time_for_plane = potential_time
                        best_runway = r_idx
                    # Tie-breaking: prefer runway 0 if times are equal
                    elif potential_time == best_time_for_plane and r_idx == 0:
                        best_runway = 0 # Explicitly prefer runway 0

            # Check if a feasible assignment was found
            if best_runway == -1:
                is_feasible = False
                # print(f"DEBUG Recalc 2P: Infeasible Lk for {plane_id}. R0_T={earliest_possible_on_runway[0]}, R1_T={earliest_possible_on_runway[1]} > L={L}") # Debug
                break

            # Assign and update state for the chosen runway
            landing_times[plane_id] = best_time_for_plane
            runway_last_id[best_runway] = plane_id
            runway_last_time[best_runway] = best_time_for_plane

    else:
        raise ValueError("Num pistas debe ser 1 o 2")

    # Calculate final cost only if the entire sequence was feasible
    if is_feasible:
        # Ensure all planes in sequence have assigned times before calculating cost
        if len(landing_times) != len(sequence):
             print(f"ERROR Recalc Final: Mismatch len(landing_times)={len(landing_times)} vs len(sequence)={len(sequence)}")
             return None, INFINITO_COSTO, False # Should not happen if is_feasible is True

        final_cost = calculate_total_cost(planes_data, sequence, landing_times)
        # Double check cost calculation didn't return INF
        if final_cost == INFINITO_COSTO:
             print(f"ERROR Recalc Final: calculate_total_cost returned INF for a supposedly feasible sequence.")
             return landing_times, INFINITO_COSTO, False # Mark as infeasible if cost calc failed
        return landing_times, final_cost, True
    else:
        # Return None for times if infeasible
        return None, INFINITO_COSTO, False


# --- Hill Climbing (Best Improvement - Sin cambios en su lógica interna) ---
def hill_climbing_best_improvement(initial_schedule, initial_times, initial_cost,
                                    D, planes_data, separations, num_runways, max_iterations=1000):
    """
    Realiza Hill Climbing (Mejor Mejora) usando intercambio 2-opt.
    Explora todos los vecinos en cada iteración y se mueve al mejor si mejora la solución actual.
    USA LA VERSIÓN CORREGIDA DE recalculate_landing_times.

    Args:
        initial_schedule (list): Secuencia inicial de IDs de avión.
        initial_times (dict): Tiempos de aterrizaje iniciales.
        initial_cost (float): Costo inicial.
        D (int): Número de aviones.
        planes_data (list): Datos originales [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        max_iterations (int): Límite de iteraciones para evitar ciclos o ejecución muy larga.

    Returns:
        tuple: (best_schedule, best_times, best_cost)
               La mejor solución encontrada (puede ser la inicial si no hubo mejora).
    """
    if D < 2: # No se pueden hacer intercambios
        return initial_schedule, initial_times, initial_cost
    if initial_cost == INFINITO_COSTO or not initial_schedule:
         return initial_schedule, initial_times, initial_cost # Return invalid input as is

    current_schedule = list(initial_schedule) # Copiar para no modificar original
    current_times = dict(initial_times)     # Copiar
    current_cost = initial_cost

    iterations = 0
    while iterations < max_iterations:
        iterations += 1
        best_neighbor_schedule = None
        best_neighbor_times = None
        best_neighbor_cost = current_cost # Inicializar con el costo actual
        found_improving_neighbor = False

        # Iterar sobre TODOS los pares posibles para intercambiar (i, j)
        for i in range(D):
            for j in range(i + 1, D):
                # Crear vecino intercambiando posiciones i y j
                neighbor_schedule_try = list(current_schedule)
                neighbor_schedule_try[i], neighbor_schedule_try[j] = neighbor_schedule_try[j], neighbor_schedule_try[i]

                # Recalcular tiempos, costo y factibilidad para el vecino (USA VERSIÓN CORREGIDA)
                neighbor_times_try, neighbor_cost_try, is_feasible = recalculate_landing_times(
                    neighbor_schedule_try, D, planes_data, separations, num_runways
                )

                # Si el vecino es factible Y es MEJOR que el MEJOR vecino encontrado HASTA AHORA
                if is_feasible and neighbor_cost_try < best_neighbor_cost:
                    # Actualizar el mejor vecino encontrado en ESTA iteración
                    best_neighbor_schedule = neighbor_schedule_try
                    best_neighbor_times = neighbor_times_try
                    best_neighbor_cost = neighbor_cost_try
                    found_improving_neighbor = True # Marcamos que al menos un vecino mejora

        # --- Fin de la exploración de vecinos ---

        # Si se encontró un vecino que mejora el costo actual, moverse a ESE MEJOR vecino
        if found_improving_neighbor:
            # print(f"  HC Mejor Mejora encontrada en iter {iterations}, nuevo costo: {best_neighbor_cost:.2f}") # Debug opcional
            current_schedule = best_neighbor_schedule
            current_times = best_neighbor_times
            current_cost = best_neighbor_cost
            # Continuar al siguiente ciclo while para buscar mejoras desde la nueva solución
        else:
            # Si NINGÚN vecino explorado mejoró el costo actual, hemos alcanzado un óptimo local
            # print(f"  HC Óptimo local (Mejor Mejora) alcanzado después de {iterations} iteraciones.") # Debug opcional
            break # Salir del bucle while

    # Retornar la mejor solución encontrada al final del proceso
    return current_schedule, current_times, current_cost

