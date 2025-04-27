# greedy_stochastic_weighted_rcl.py
import random
import math
import copy
# Asegúrate que la ruta a calculate_cost sea correcta
try:
    from scripts.calculate_cost import calculate_total_cost
except ImportError:
    from calculate_cost import calculate_total_cost
    print("Advertencia: Se usó importación local para calculate_cost.")

# --- Constantes ---
INFINITO_COSTO = float('inf')
EPSILON = 1e-6 # Pequeño valor para evitar división por cero en pesos

def solve_greedy_stochastic(D, planes_data, separations, num_runways, seed, rcl_size=3):
    """
    Implementa el algoritmo greedy estocástico usando RCL con SELECCIÓN PONDERADA.
    Los candidatos en la RCL tienen mayor probabilidad de ser elegidos si
    tienen menor 'slack' (L - min_time).
    Retorna float('inf') como costo si no puede completar la planificación.
    """
    random.seed(seed)
    unscheduled = set(range(D))
    schedule = []
    landing_times = {}

    # --- Lógica para 1 Pista ---
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
                    slack = L - min_start_time
                    feasible_candidates.append({
                        'id': plane_id,
                        'min_time': min_start_time,
                        'L': L,
                        'P': P,
                        'slack': slack # Guardar slack para pesos
                    })

            if not feasible_candidates:
                return [], {}, INFINITO_COSTO

            # Ordenar por Slack (menor primero) para crear RCL base
            feasible_candidates.sort(key=lambda x: (x['slack'], x['P'], x['id']))

            # --- Construcción RCL ---
            current_rcl_size = min(rcl_size, len(feasible_candidates))
            rcl = feasible_candidates[:current_rcl_size]

            # --- Selección PONDERADA de la RCL ---
            if not rcl: # Si RCL está vacía (aunque no debería pasar si feasible_candidates no lo estaba)
                 return [], {}, INFINITO_COSTO

            # Calcular pesos inversamente proporcionales al slack
            weights = []
            for candidate in rcl:
                # Peso mayor para slack menor. Añadir EPSILON para evitar 1/0.
                weight = 1.0 / (candidate['slack'] + EPSILON)
                weights.append(weight)

            # Normalizar pesos para que sumen 1 (convertir a probabilidades)
            total_weight = sum(weights)
            if total_weight > 0:
                probabilities = [w / total_weight for w in weights]
            else:
                # Si todos los pesos son 0 (improbable con EPSILON), usar probabilidad uniforme
                probabilities = [1.0 / len(rcl)] * len(rcl)

            # Elegir usando las probabilidades calculadas
            # random.choices devuelve una lista, tomamos el primer elemento [0]
            chosen_candidate = random.choices(rcl, weights=probabilities, k=1)[0]

            # --- Asignación y Actualización ---
            chosen_plane_id = chosen_candidate['id']
            chosen_time = chosen_candidate['min_time']

            schedule.append(chosen_plane_id)
            landing_times[chosen_plane_id] = chosen_time
            unscheduled.remove(chosen_plane_id)
            last_scheduled_id = chosen_plane_id
            last_landing_time = chosen_time

    # --- Lógica para 2 Pistas ---
    elif num_runways == 2:
        runway_last_id = [None, None]
        runway_last_time = [-1, -1]
        while unscheduled:
            candidate_options = []
            for plane_id in unscheduled:
                E, P, L, Ce, Cl = planes_data[plane_id]
                possible_times_on_runways = []
                for r_idx in range(num_runways):
                    current_min_time = E
                    if runway_last_id[r_idx] is not None:
                         sep = separations[runway_last_id[r_idx]][plane_id]
                         current_min_time = max(current_min_time, runway_last_time[r_idx] + sep)
                    possible_times_on_runways.append((current_min_time, r_idx))

                possible_times_on_runways.sort(key=lambda x: (x[0], x[1]))
                best_possible_time, assigned_runway_idx = possible_times_on_runways[0]

                if best_possible_time <= L:
                    slack = L - best_possible_time
                    candidate_options.append({
                        'id': plane_id,
                        'min_time': best_possible_time,
                        'assigned_runway': assigned_runway_idx,
                        'L': L,
                        'P': P,
                        'slack': slack # Guardar slack
                    })

            if not candidate_options:
                return [], {}, INFINITO_COSTO

            # Ordenar por Slack para crear RCL base
            candidate_options.sort(key=lambda x: (x['slack'], x['P'], x['id']))

            # --- Construcción RCL ---
            current_rcl_size = min(rcl_size, len(candidate_options))
            rcl = candidate_options[:current_rcl_size]

            # --- Selección PONDERADA de la RCL ---
            if not rcl:
                return [], {}, INFINITO_COSTO

            weights = [1.0 / (c['slack'] + EPSILON) for c in rcl]
            total_weight = sum(weights)
            if total_weight > 0:
                 probabilities = [w / total_weight for w in weights]
            else:
                 probabilities = [1.0 / len(rcl)] * len(rcl)

            chosen_candidate = random.choices(rcl, weights=probabilities, k=1)[0]

            # --- Asignación y Actualización ---
            chosen_plane_id = chosen_candidate['id']
            chosen_time = chosen_candidate['min_time']
            chosen_runway = chosen_candidate['assigned_runway']

            schedule.append(chosen_plane_id)
            landing_times[chosen_plane_id] = chosen_time
            unscheduled.remove(chosen_plane_id)
            runway_last_id[chosen_runway] = chosen_plane_id
            runway_last_time[chosen_runway] = chosen_time

    else:
        raise ValueError("El número de pistas debe ser 1 o 2")

    # --- Verificación Final y Cálculo de Costo ---
    if len(schedule) == D:
        total_cost = calculate_total_cost(planes_data, schedule, landing_times)
        return schedule, landing_times, total_cost
    else:
        return [], {}, INFINITO_COSTO
