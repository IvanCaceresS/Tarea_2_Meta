# greedy_stochastic.py (Híbrido Slack/Costo/P + Fallback Temprano)
import random
import math
import copy

# Asegúrate que la ruta a calculate_cost sea correcta y apunte a la versión ORIGINAL
try:
    from scripts.calculate_cost import calculate_total_cost
except ImportError:
    try:
        from calculate_cost import calculate_total_cost
    except ImportError:
        print("ERROR FATAL: No se pudo encontrar 'calculate_cost.py' (ORIGINAL).")
        def calculate_total_cost(*args, **kwargs):
            print("ERROR FATAL: calculate_total_cost no disponible.")
            return float('inf')

# --- Constantes ---
INFINITO_COSTO = float('inf')

def solve_greedy_stochastic(D, planes_data, separations, num_runways, seed, rcl_size=3):
    """
    Implementa el algoritmo greedy estocástico con heurística híbrida y fallback modificado.

    - RCL basada en: 1. Slack (menor), 2. Costo P (menor), 3. Tiempo P (menor).
    - Selecciona aleatoriamente de la RCL si hay candidatos factibles (T <= L).
    - Si NO hay candidatos factibles, elige DETERMINÍSTICAMENTE aquel
      que puede empezar más temprano (min_start_time), ignorando Lk.
    - Retorna una secuencia completa y el costo calculado con la función ORIGINAL.

    Args:
        D (int): Número de aviones.
        planes_data (list): Lista de [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        seed (int): Semilla para el generador de números aleatorios.
        rcl_size (int): Tamaño de la Lista Restringida de Candidatos (RCL).

    Returns:
        tuple: (schedule, landing_times, cost)
               - schedule (list): Lista COMPLETA de IDs.
               - landing_times (dict): Tiempos asignados (pueden violar L).
               - cost (float): Costo original (sin penalización Lk explícita).
                               INFINITO_COSTO si hubo error grave.
    """
    random.seed(seed)
    unscheduled = set(range(D))
    schedule = []
    landing_times = {}

    # --- Lógica para 1 Pista ---
    if num_runways == 1:
        last_scheduled_id = None
        last_landing_time = -1.0

        while unscheduled:
            feasible_candidates = []
            infeasible_potential_candidates = [] # Guardar todos los T > L para el fallback

            for plane_id in unscheduled:
                if not (0 <= plane_id < len(planes_data)): continue
                E, P, L, Ce, Cl = planes_data[plane_id]
                min_start_time = float(E)
                if last_scheduled_id is not None:
                    if not (0 <= last_scheduled_id < D and 0 <= plane_id < D): return [], {}, INFINITO_COSTO
                    if last_scheduled_id >= len(separations) or plane_id >= len(separations[last_scheduled_id]): return [], {}, INFINITO_COSTO
                    separation_needed = float(separations[last_scheduled_id][plane_id])
                    min_start_time = max(min_start_time, last_landing_time + separation_needed)

                if min_start_time <= L:
                    # FACTIBLE: Calcular Slack, Costo y P
                    slack = L - min_start_time
                    costo_individual = Ce * max(0, P - min_start_time) + Cl * max(0, min_start_time - P)
                    feasible_candidates.append({
                        'id': plane_id,
                        'min_time': min_start_time,
                        'slack': slack,
                        'cost': costo_individual,
                        'P': P # Necesario para desempate
                    })
                else:
                    # INFACTIBLE: Guardar para fallback
                    infeasible_potential_candidates.append({
                         'id': plane_id,
                         'min_time': min_start_time, # El tiempo calculado (aunque > L)
                         # No necesitamos violation aquí para el nuevo fallback
                    })


            # --- Selección de Candidato ---
            chosen_candidate = None

            if feasible_candidates:
                # CASO 1: Usar RCL Híbrida
                # Ordenar por: 1. Slack (asc), 2. Costo (asc), 3. P (asc)
                feasible_candidates.sort(key=lambda x: (x['slack'], x['cost'], x['P']))
                current_rcl_size = min(rcl_size, len(feasible_candidates))
                rcl = feasible_candidates[:current_rcl_size]
                if rcl:
                     chosen_candidate = random.choice(rcl) # Selección uniforme
                else:
                     print(f"ERROR LÓGICO (Stoch 1P Hybrid, Seed {seed}): RCL vacía.")
                     if feasible_candidates: chosen_candidate = feasible_candidates[0]
                     else: return [], {}, INFINITO_COSTO * 0.98

            elif infeasible_potential_candidates:
                # CASO 2: Fallback -> Elegir el que puede empezar MÁS TEMPRANO (ignorando L)
                # print(f"  DEBUG Stoch 1P Hybrid (Seed {seed}): Sin candidatos factibles. Fallback: eligiendo el de menor min_time...") # Debug
                infeasible_potential_candidates.sort(key=lambda x: x['min_time'])
                chosen_candidate = infeasible_potential_candidates[0] # Determinista

            else:
                 # CASO 3: Error o fin
                 if not unscheduled: break
                 else:
                      print(f"ERROR FATAL (Stoch 1P Hybrid, Seed {seed}): Ni factibles ni infactibles.")
                      return [], {}, INFINITO_COSTO * 0.99

            # --- Asignación y Actualización ---
            if chosen_candidate:
                 chosen_plane_id = chosen_candidate['id']
                 chosen_time = chosen_candidate['min_time']
                 schedule.append(chosen_plane_id)
                 landing_times[chosen_plane_id] = chosen_time
                 unscheduled.remove(chosen_plane_id)
                 last_scheduled_id = chosen_plane_id
                 last_landing_time = chosen_time
            else:
                 print(f"ERROR FATAL (Stoch 1P Hybrid, Seed {seed}): chosen_candidate es None.")
                 return [], {}, INFINITO_COSTO * 0.97


    # --- Lógica para 2 Pistas ---
    elif num_runways == 2:
        runway_last_id = [None, None]
        runway_last_time = [-1.0, -1.0]

        while unscheduled:
            feasible_candidates = []
            infeasible_potential_candidates = []

            for plane_id in unscheduled:
                if not (0 <= plane_id < len(planes_data)): continue
                E, P, L, Ce, Cl = planes_data[plane_id]
                best_time_overall = INFINITO_COSTO
                best_runway_overall = -1
                is_best_time_feasible = False

                # Guardar tiempos potenciales en cada pista para el fallback
                potential_times_for_fallback = {}

                for r_idx in range(num_runways):
                    current_time = float(E)
                    last_id_on_runway = runway_last_id[r_idx]
                    if last_id_on_runway is not None:
                        if not (0 <= last_id_on_runway < D and 0 <= plane_id < D): continue
                        if last_id_on_runway >= len(separations) or plane_id >= len(separations[last_id_on_runway]): continue
                        sep = float(separations[last_id_on_runway][plane_id])
                        current_time = max(current_time, runway_last_time[r_idx] + sep)

                    # Guardar el tiempo potencial para el fallback
                    potential_times_for_fallback[r_idx] = current_time

                    # Evaluar si es el mejor hasta ahora (considerando L para factibilidad)
                    if current_time <= L: # Solo considerar si es factible respecto a L
                        if current_time < best_time_overall:
                            best_time_overall = current_time
                            best_runway_overall = r_idx
                            is_best_time_feasible = True
                        elif current_time == best_time_overall and r_idx == 0: # Preferir pista 0
                            best_runway_overall = r_idx
                            is_best_time_feasible = True
                    # Si no es factible (current_time > L), solo lo guardamos para el fallback

                # Clasificar al candidato
                if best_runway_overall != -1 and is_best_time_feasible:
                     # FACTIBLE: Calcular Slack, Costo y P para la mejor opción factible
                     slack = L - best_time_overall
                     costo_individual = Ce * max(0, P - best_time_overall) + Cl * max(0, best_time_overall - P)
                     feasible_candidates.append({
                         'id': plane_id,
                         'min_time': best_time_overall,
                         'assigned_runway': best_runway_overall,
                         'slack': slack,
                         'cost': costo_individual,
                         'P': P
                     })
                elif potential_times_for_fallback: # Si hubo al menos un tiempo calculado (aunque > L)
                     # INFACTIBLE o sin opción factible: Preparar para fallback
                     # Encontrar el tiempo mínimo absoluto entre las pistas (ignorando L)
                     min_infeasible_time = min(potential_times_for_fallback.values())
                     # Encontrar la pista correspondiente (prefiriendo 0 en empate)
                     chosen_infeasible_runway = -1
                     if potential_times_for_fallback.get(0) == min_infeasible_time:
                          chosen_infeasible_runway = 0
                     elif potential_times_for_fallback.get(1) == min_infeasible_time:
                          chosen_infeasible_runway = 1

                     if chosen_infeasible_runway != -1:
                          infeasible_potential_candidates.append({
                              'id': plane_id,
                              'min_time': min_infeasible_time, # El tiempo más temprano posible
                              'assigned_runway': chosen_infeasible_runway
                              # No necesitamos violation para este fallback
                          })


            # --- Selección de Candidato ---
            chosen_candidate = None

            if feasible_candidates:
                # CASO 1: Usar RCL Híbrida
                feasible_candidates.sort(key=lambda x: (x['slack'], x['cost'], x['P']))
                current_rcl_size = min(rcl_size, len(feasible_candidates))
                rcl = feasible_candidates[:current_rcl_size]
                if rcl:
                    chosen_candidate = random.choice(rcl)
                else:
                    print(f"ERROR LÓGICO (Stoch 2P Hybrid, Seed {seed}): RCL vacía.")
                    if feasible_candidates: chosen_candidate = feasible_candidates[0]
                    else: return [], {}, INFINITO_COSTO * 0.98

            elif infeasible_potential_candidates:
                # CASO 2: Fallback -> Elegir el que puede empezar MÁS TEMPRANO
                # print(f"  DEBUG Stoch 2P Hybrid (Seed {seed}): Sin candidatos factibles. Fallback: eligiendo el de menor min_time...") # Debug
                infeasible_potential_candidates.sort(key=lambda x: x['min_time'])
                chosen_candidate = infeasible_potential_candidates[0] # Determinista

            else:
                 # CASO 3: Error o fin
                 if not unscheduled: break
                 else:
                      print(f"ERROR FATAL (Stoch 2P Hybrid, Seed {seed}): Ni factibles ni infactibles.")
                      return [], {}, INFINITO_COSTO * 0.99

            # --- Asignación y Actualización ---
            if chosen_candidate:
                 chosen_plane_id = chosen_candidate['id']
                 chosen_time = chosen_candidate['min_time']
                 chosen_runway = chosen_candidate['assigned_runway']
                 schedule.append(chosen_plane_id)
                 landing_times[chosen_plane_id] = chosen_time
                 unscheduled.remove(chosen_plane_id)
                 runway_last_id[chosen_runway] = chosen_plane_id
                 runway_last_time[chosen_runway] = chosen_time
            else:
                 print(f"ERROR FATAL (Stoch 2P Hybrid, Seed {seed}): chosen_candidate es None.")
                 return [], {}, INFINITO_COSTO * 0.97

    else:
        raise ValueError("El número de pistas debe ser 1 o 2")

    # --- Cálculo de Costo Final ---
    final_cost = calculate_total_cost(planes_data, schedule, landing_times)

    return schedule, landing_times, final_cost
