from scripts.calculate_cost import calculate_total_cost

def solve_greedy_deterministic(D, planes_data, separations, num_runways):
    """
    Implementa el algoritmo greedy determinista (selecciona por P más temprano).
    Retorna float('inf') como costo si no puede completar la planificación.
    """
    unscheduled = set(range(D))
    schedule = []
    landing_times = {}

    if num_runways == 1:
        last_scheduled_id = None
        last_landing_time = -1
        while unscheduled:
            feasible_candidates = []
            for plane_id in unscheduled:
                E, P, L, Ce, Cl = planes_data[plane_id]
                min_start_time = E
                if last_scheduled_id is not None:
                    separation_needed = separations[last_scheduled_id][plane_id]
                    min_start_time = max(min_start_time, last_landing_time + separation_needed)
                if min_start_time <= L:
                    feasible_candidates.append((plane_id, min_start_time))

            if not feasible_candidates:
                print(f"ERROR (Determinista 1P): ¡No se encontró candidato factible! No planificados: {unscheduled}")
                # Retornar indicando fallo completo
                return schedule, landing_times, float('inf') # <-- CAMBIO

            feasible_candidates.sort(key=lambda x: (planes_data[x[0]][1], x[0]))
            chosen_plane_id, chosen_time = feasible_candidates[0]
            schedule.append(chosen_plane_id)
            landing_times[chosen_plane_id] = chosen_time
            unscheduled.remove(chosen_plane_id)
            last_scheduled_id = chosen_plane_id
            last_landing_time = chosen_time

    elif num_runways == 2:
        runway_last_id = [None, None]
        runway_last_time = [-1, -1]
        runway_assignments = {}
        while unscheduled:
            candidate_options = []
            for plane_id in unscheduled:
                E, P, L, Ce, Cl = planes_data[plane_id]
                feasible_times_on_runways = []
                min_time_r0 = E
                if runway_last_id[0] is not None:
                     sep0 = separations[runway_last_id[0]][plane_id]
                     min_time_r0 = max(min_time_r0, runway_last_time[0] + sep0)
                feasible_times_on_runways.append(min_time_r0 if min_time_r0 <= L else float('inf'))
                min_time_r1 = E
                if runway_last_id[1] is not None:
                     sep1 = separations[runway_last_id[1]][plane_id]
                     min_time_r1 = max(min_time_r1, runway_last_time[1] + sep1)
                feasible_times_on_runways.append(min_time_r1 if min_time_r1 <= L else float('inf'))
                best_time_for_plane = min(feasible_times_on_runways)
                if best_time_for_plane != float('inf'):
                    candidate_options.append((plane_id, best_time_for_plane))

            if not candidate_options:
                print(f"ERROR (Determinista 2P): ¡No se encontró candidato factible! No planificados: {unscheduled}")
                # Retornar indicando fallo completo
                return schedule, landing_times, float('inf') # <-- CAMBIO

            candidate_options.sort(key=lambda x: (planes_data[x[0]][1], x[0]))
            chosen_plane_id, _ = candidate_options[0]
            E, P, L, Ce, Cl = planes_data[chosen_plane_id]
            time_r0 = E
            if runway_last_id[0] is not None:
                 time_r0 = max(time_r0, runway_last_time[0] + separations[runway_last_id[0]][chosen_plane_id])
            feasible_r0 = (time_r0 <= L)
            time_r1 = E
            if runway_last_id[1] is not None:
                 time_r1 = max(time_r1, runway_last_time[1] + separations[runway_last_id[1]][chosen_plane_id])
            feasible_r1 = (time_r1 <= L)
            chosen_time = float('inf')
            chosen_runway = -1
            if feasible_r0 and (not feasible_r1 or time_r0 <= time_r1): # Preferir pista 0 en empates
                chosen_time = time_r0
                chosen_runway = 0
            elif feasible_r1:
                chosen_time = time_r1
                chosen_runway = 1
            # La condición if not candidate_options previene que chosen_time siga siendo inf aquí

            schedule.append(chosen_plane_id)
            landing_times[chosen_plane_id] = chosen_time
            runway_assignments[chosen_plane_id] = chosen_runway
            runway_last_id[chosen_runway] = chosen_plane_id
            runway_last_time[chosen_runway] = chosen_time
            unscheduled.remove(chosen_plane_id)
    else:
        raise ValueError("El número de pistas debe ser 1 o 2")

    # Calcular costo final solo si se completó (no retornó inf antes)
    total_cost = calculate_total_cost(planes_data, schedule, landing_times)
    return schedule, landing_times, total_cost