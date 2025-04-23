# greedy_stochastic.py
import random
import math
import copy
from scripts.calculate_cost import calculate_total_cost

def solve_greedy_stochastic(D, planes_data, separations, num_runways, seed, rcl_size=3):
    """
    Implementa el algoritmo greedy estocástico usando RCL.
    Retorna float('inf') como costo si no puede completar la planificación.
    """
    random.seed(seed)
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
                print(f"ERROR (Estocástico 1P, Seed {seed}): ¡No se encontró candidato factible! No planificados: {unscheduled}")
                # Retornar indicando fallo completo
                return schedule, landing_times, float('inf') # <-- CAMBIO

            feasible_candidates.sort(key=lambda x: (planes_data[x[0]][1], x[0]))
            current_rcl_size = min(rcl_size, len(feasible_candidates))
            rcl = feasible_candidates[:current_rcl_size]
            chosen_plane_id, chosen_time = random.choice(rcl)
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
                    candidate_options.append({'id': plane_id, 'P': planes_data[plane_id][1], 'best_time': best_time_for_plane})

            if not candidate_options:
                print(f"ERROR (Estocástico 2P, Seed {seed}): ¡No se encontró candidato factible! No planificados: {unscheduled}")
                # Retornar indicando fallo completo
                return schedule, landing_times, float('inf') # <-- CAMBIO

            candidate_options.sort(key=lambda x: (x['P'], x['id']))
            current_rcl_size = min(rcl_size, len(candidate_options))
            rcl = candidate_options[:current_rcl_size]
            chosen_candidate = random.choice(rcl)
            chosen_plane_id = chosen_candidate['id']
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
            if feasible_r0 and (not feasible_r1 or time_r0 <= time_r1):
                chosen_time = time_r0
                chosen_runway = 0
            elif feasible_r1:
                chosen_time = time_r1
                chosen_runway = 1
            else:
                print(f"ERROR Lógico (Estocástico 2P, Seed {seed}): Avión elegido {chosen_plane_id} se volvió infactible.")
                # Retornar indicando fallo completo también aquí
                return schedule, landing_times, float('inf') # <-- CAMBIO ADICIONAL

            schedule.append(chosen_plane_id)
            landing_times[chosen_plane_id] = chosen_time
            runway_assignments[chosen_plane_id] = chosen_runway
            runway_last_id[chosen_runway] = chosen_plane_id
            runway_last_time[chosen_runway] = chosen_time
            unscheduled.remove(chosen_plane_id)
    else:
        raise ValueError("El número de pistas debe ser 1 o 2")

    # Calcular costo final solo si se completó
    total_cost = calculate_total_cost(planes_data, schedule, landing_times)
    return schedule, landing_times,total_cost