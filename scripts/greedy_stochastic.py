# greedy_stochastic.py (Versión Corregida con RCL basada en Costo)
import random
import math
import copy
# Asegúrate que la ruta a calculate_cost sea correcta
try:
    # Intenta importar desde 'scripts' primero
    from scripts.calculate_cost import calculate_total_cost
except ImportError:
    # Si falla, intenta importar localmente
    try:
        from calculate_cost import calculate_total_cost
        print("Advertencia: Se usó importación local para calculate_cost.")
    except ImportError:
        print("ERROR: No se pudo encontrar 'calculate_cost.py' en 'scripts/' ni localmente.")
        # Define una función dummy para evitar errores posteriores
        def calculate_total_cost(*args, **kwargs):
             print("ERROR FATAL: calculate_total_cost no disponible.")
             return float('inf')


# --- Constantes ---
INFINITO_COSTO = float('inf')
# EPSILON = 1e-6 # Ya no se necesita para la selección ponderada

def solve_greedy_stochastic(D, planes_data, separations, num_runways, seed, rcl_size=3):
    """
    Implementa el algoritmo greedy estocástico priorizando la factibilidad.
    La Lista Restringida de Candidatos (RCL) se construye con los 'rcl_size'
    candidatos factibles que generan el menor costo individual inmediato.
    Si no se encuentran candidatos factibles en algún paso, la construcción falla
    y retorna costo infinito para esa ejecución.

    Args:
        D (int): Número de aviones.
        planes_data (list): Lista de [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        seed (int): Semilla para el generador de números aleatorios.
        rcl_size (int): Tamaño de la Lista Restringida de Candidatos (RCL).

    Returns:
        tuple: (schedule, landing_times, cost)
               - schedule (list): Lista de IDs de aviones planificados.
               - landing_times (dict): Tiempos de aterrizaje asignados.
               - cost (float): Costo total si se encontró una solución factible completa,
                               INFINITO_COSTO si la construcción falló.
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

                # Solo considerar candidatos factibles respecto a L
                if min_start_time <= L:
                    # *** NUEVO: Calcular costo individual ***
                    costo_individual = Ce * max(0, P - min_start_time) + Cl * max(0, min_start_time - P)
                    feasible_candidates.append({
                        'id': plane_id,
                        'min_time': min_start_time,
                        'cost': costo_individual, # Guardar costo individual
                        # 'L': L, # Ya no son necesarios para ordenar/seleccionar
                        # 'P': P,
                        # 'slack': L - min_start_time
                    })
                # else: Se ignora el candidato por ser infactible (min_start_time > L)

            # --- Selección de la RCL y Candidato ---
            if not feasible_candidates:
                # print(f"WARN (Stoch 1P, Seed {seed}): No se encontraron candidatos factibles. No planificados: {unscheduled}. Construcción fallida.")
                return schedule, landing_times, INFINITO_COSTO # Retorna fallo

            # *** NUEVO: Ordenar candidatos factibles por costo individual (menor primero) ***
            feasible_candidates.sort(key=lambda x: x['cost'])

            # *** NUEVO: Construir RCL con los mejores 'rcl_size' por costo ***
            current_rcl_size = min(rcl_size, len(feasible_candidates))
            rcl = feasible_candidates[:current_rcl_size]

            if not rcl:
                 print(f"ERROR LÓGICO (Stoch 1P, Seed {seed}): RCL vacía inesperadamente.")
                 return schedule, landing_times, INFINITO_COSTO # Retorna fallo

            # *** NUEVO: Selección uniforme de la RCL (ya no ponderada por slack) ***
            chosen_candidate = random.choice(rcl)

            # --- Asignación y Actualización ---
            chosen_plane_id = chosen_candidate['id']
            chosen_time = chosen_candidate['min_time'] # Tiempo factible

            schedule.append(chosen_plane_id)
            landing_times[chosen_plane_id] = chosen_time
            unscheduled.remove(chosen_plane_id)
            last_scheduled_id = chosen_plane_id
            last_landing_time = chosen_time

    # --- Lógica para 2 Pistas ---
    elif num_runways == 2:
        runway_last_id = [None, None]
        runway_last_time = [-1, -1]
        # runway_assignments = {} # No parece usarse fuera de la función, se puede quitar si no es necesario para el futuro
        while unscheduled:
            feasible_candidates = []
            for plane_id in unscheduled:
                E, P, L, Ce, Cl = planes_data[plane_id]
                best_feasible_time = INFINITO_COSTO
                best_runway_for_plane = -1

                # Evaluar ambas pistas para encontrar el *mejor tiempo factible*
                for r_idx in range(num_runways):
                    current_min_time = E
                    if runway_last_id[r_idx] is not None:
                        sep = separations[runway_last_id[r_idx]][plane_id]
                        current_min_time = max(current_min_time, runway_last_time[r_idx] + sep)

                    # Solo considerar si es factible respecto a L
                    if current_min_time <= L:
                        # Si es mejor que el mejor tiempo factible encontrado hasta ahora para este avión
                        if current_min_time < best_feasible_time or \
                           (current_min_time == best_feasible_time and r_idx < best_runway_for_plane): # Preferir pista 0 en empates
                            best_feasible_time = current_min_time
                            best_runway_for_plane = r_idx

                # Si se encontró al menos una opción factible para este avión
                if best_runway_for_plane != -1:
                    # *** NUEVO: Calcular costo individual para el mejor tiempo/pista encontrado ***
                    costo_individual = Ce * max(0, P - best_feasible_time) + Cl * max(0, best_feasible_time - P)
                    feasible_candidates.append({
                        'id': plane_id,
                        'min_time': best_feasible_time, # El mejor tiempo factible encontrado
                        'assigned_runway': best_runway_for_plane, # La pista para ese mejor tiempo
                        'cost': costo_individual # Guardar costo individual
                        # 'L': L, # Ya no son necesarios
                        # 'P': P,
                        # 'slack': L - best_feasible_time
                    })
                # else: Se ignora, no hay forma factible de aterrizar este avión ahora

            # --- Selección de la RCL y Candidato ---
            if not feasible_candidates:
                # print(f"WARN (Stoch 2P, Seed {seed}): No se encontraron candidatos factibles. No planificados: {unscheduled}. Construcción fallida.")
                return schedule, landing_times, INFINITO_COSTO # Retorna fallo

            # *** NUEVO: Ordenar candidatos factibles por costo individual (menor primero) ***
            feasible_candidates.sort(key=lambda x: x['cost'])

            # *** NUEVO: Construir RCL con los mejores 'rcl_size' por costo ***
            current_rcl_size = min(rcl_size, len(feasible_candidates))
            rcl = feasible_candidates[:current_rcl_size]

            if not rcl:
                print(f"ERROR LÓGICO (Stoch 2P, Seed {seed}): RCL vacía inesperadamente.")
                return schedule, landing_times, INFINITO_COSTO # Retorna fallo

            # *** NUEVO: Selección uniforme de la RCL ***
            chosen_candidate = random.choice(rcl)

            # --- Asignación y Actualización ---
            chosen_plane_id = chosen_candidate['id']
            chosen_time = chosen_candidate['min_time']       # Tiempo factible
            chosen_runway = chosen_candidate['assigned_runway'] # Pista asignada

            schedule.append(chosen_plane_id)
            landing_times[chosen_plane_id] = chosen_time
            unscheduled.remove(chosen_plane_id)
            runway_last_id[chosen_runway] = chosen_plane_id
            runway_last_time[chosen_runway] = chosen_time
            # runway_assignments[chosen_plane_id] = chosen_runway # Quitado si no se usa

    else:
        raise ValueError("El número de pistas debe ser 1 o 2")

    # --- Cálculo de Costo Final ---
    # Si llegamos aquí, se planificaron todos los aviones de forma factible.
    total_cost = calculate_total_cost(planes_data, schedule, landing_times)
    return schedule, landing_times, total_cost