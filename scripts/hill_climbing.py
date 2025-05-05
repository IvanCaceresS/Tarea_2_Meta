import copy # Para copiar objetos y evitar modificar los originales
from scripts.calculate_cost import calculate_total_cost

# --- Constantes ---
INFINITO_COSTO = float('inf')
# Asegúrate que coincida con el usado en otros módulos si es relevante
# MARCADOR_INFINITO_ENTRADA = 99999

def recalculate_landing_times(sequence, D, planes_data, separations, num_runways):
    """
    Recalcula los tiempos de aterrizaje más tempranos posibles para una secuencia dada.
    (Esta función no cambia entre First y Best Improvement)

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

    is_feasible = True # Asumir factibilidad inicial

    if num_runways == 1:
        last_scheduled_id = None
        last_landing_time = -1
        for plane_id in sequence:
            E, P, L, Ce, Cl = planes_data[plane_id]
            min_start_time = E
            if last_scheduled_id is not None:
                separation_needed = separations[last_scheduled_id][plane_id]
                # Manejo opcional de separaciones infinitas si existe ese marcador
                # if separation_needed == MARCADOR_INFINITO_ENTRADA:
                #     is_feasible = False
                #     break
                min_start_time = max(min_start_time, last_landing_time + separation_needed)

            # Verificar ventana L
            if min_start_time > L:
                 is_feasible = False
                 break # Esta secuencia es infactible

            # Asignar tiempo y actualizar estado para el siguiente
            landing_times[plane_id] = min_start_time
            last_scheduled_id = plane_id
            last_landing_time = min_start_time

    elif num_runways == 2:
        runway_last_id = [None, None]
        runway_last_time = [-1, -1]
        historial_aterrizajes = [] # Mantenemos (id, tiempo) para calcular EAT global

        for plane_id in sequence:
            E, P, L, Ce, Cl = planes_data[plane_id]

            # 1. Calcular EAT Global (Earliest Arrival Time) basado en separaciones
            #    con TODOS los aviones previamente aterrizados en CUALQUIER pista.
            eat_global = E
            for id_previo, tiempo_previo in historial_aterrizajes:
                 separation_needed = separations[id_previo][plane_id]
                 # Manejo opcional de separaciones infinitas
                 # if separation_needed == MARCADOR_INFINITO_ENTRADA:
                 #     is_feasible = False
                 #     break
                 eat_global = max(eat_global, tiempo_previo + separation_needed)
            if not is_feasible: break # Salir si ya se detectó infactibilidad

            # 2. Encontrar la mejor pista y tiempo factible para este avión
            best_time_for_plane = INFINITO_COSTO
            best_runway = -1
            for pista_id in range(num_runways):
                 # El tiempo mínimo en esta pista depende de la disponibilidad global (eat_global)
                 # y del último aterrizaje EN ESA PISTA específica.
                 # Nota: En la implementación original de recalculate, la lógica podría necesitar revisión
                 # para asegurar que considera la disponibilidad de la pista correctamente.
                 # Una interpretación común es que el avión puede aterrizar en la pista `pista_id`
                 # en el tiempo `max(eat_global, runway_last_time[pista_id])`, siempre que respete L.
                 # Sin embargo, la implementación original parece usar solo runway_last_time para calcular
                 # el tiempo potencial, lo cual podría ser incorrecto si eat_global es mayor.
                 # Vamos a usar la lógica más segura: max(eat_global, runway_last_time[pista_id])

                 tiempo_disponible_pista = runway_last_time[pista_id]
                 tiempo_potencial = max(eat_global, tiempo_disponible_pista) # Debe respetar ambas restricciones

                 # Factibilidad L
                 if tiempo_potencial <= L:
                      # ¿Es el mejor tiempo factible encontrado hasta ahora para este avión?
                      if tiempo_potencial < best_time_for_plane or \
                         (tiempo_potencial == best_time_for_plane and pista_id < best_runway): # Preferir pista 0 en empates
                          best_time_for_plane = tiempo_potencial
                          best_runway = pista_id

            # 3. Verificar si se encontró una asignación factible
            if best_runway == -1: # No se encontró pista/tiempo <= L
                 is_feasible = False
                 break # Esta secuencia es infactible

            # 4. Asignar y actualizar estado
            landing_times[plane_id] = best_time_for_plane
            runway_last_id[best_runway] = plane_id       # Actualiza el último en la pista asignada
            runway_last_time[best_runway] = best_time_for_plane # Actualiza tiempo libre de esa pista
            historial_aterrizajes.append((plane_id, best_time_for_plane)) # Añadir al historial global

    else:
        raise ValueError("Num pistas debe ser 1 o 2")

    # Calcular costo final solo si la secuencia completa fue factible
    if is_feasible:
        final_cost = calculate_total_cost(planes_data, sequence, landing_times)
        return landing_times, final_cost, True
    else:
        # Si no fue factible en algún punto, retornar fallo
        return None, INFINITO_COSTO, False


def hill_climbing_best_improvement(initial_schedule, initial_times, initial_cost,
                                    D, planes_data, separations, num_runways, max_iterations=1000):
    """
    Realiza Hill Climbing (Mejor Mejora) usando intercambio 2-opt.
    Explora todos los vecinos en cada iteración y se mueve al mejor si mejora la solución actual.

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

                # Recalcular tiempos, costo y factibilidad para el vecino
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