# main.py (Modificado para usar Reparación)
import os
import math
import time
import statistics
import csv
import json
import random
import copy
import sys
import traceback

# --- Importar Scripts ---
try:
    from scripts.read_case_data import read_case_data
    from scripts.greedy_deterministic import solve_greedy_deterministic
    from scripts.greedy_stochastic import solve_greedy_stochastic # Versión que puede violar Lk
    from scripts.hill_climbing import hill_climbing_best_improvement, recalculate_landing_times
    from scripts.simulated_annealing import solve_simulated_annealing
    # Importar GRASP que ahora llama a reparación internamente
    from scripts.grasp import solve_grasp, solve_hc_from_deterministic # Importar ambas
    # Importar la función de reparación directamente para usar antes de SA
    from scripts.repair_schedule import repair_schedule_simple_swap
    # Importar la función de costo ORIGINAL
    from scripts.calculate_cost import calculate_total_cost
except ImportError as e:
     print(f"ERROR FATAL en main.py: No se pudo importar un módulo necesario: {e}")
     print("Asegúrate que la estructura de directorios ('scripts/') es correcta y los archivos existen.")
     sys.exit(1)


# --- Constantes y Parámetros ---
INFINITO_COSTO = float('inf')
NUM_STOCHASTIC_RUNS = 10
RCL_SIZE = 3
HC_MAX_ITER = 500 # Iteraciones para HC dentro de GRASP y standalone
GRASP_RESTARTS_LIST = [10, 25, 50] # Iteraciones internas de GRASP
NUM_GRASP_EXECUTIONS = 10 # Veces que se ejecuta GRASP completo
SA_INITIAL_TEMPS = [10000, 5000, 1000, 500, 100]
SA_T_MIN = 0.1
SA_ALPHA = 0.95
SA_ITER_PER_TEMP = 100 # Iteraciones por nivel de temperatura en SA
SA_MAX_NEIGHBOR_ATTEMPTS = 50 # Intentos para generar vecino factible en SA
MAX_REPAIR_ATTEMPTS = None # Usará el default (2*D) en la función de reparación

# --- Funciones Auxiliares de Formato y Cálculo ---
def format_cost(cost):
    if cost == INFINITO_COSTO or cost is None: return "INF"
    try: return f"{float(cost):.2f}"
    except (ValueError, TypeError): return "ERR_FMT"

def format_time(exec_time):
    if exec_time is None: return "N/A"
    try: return f"{float(exec_time):.4f}s"
    except (ValueError, TypeError): return "ERR_FMT"

def calculate_improvement(initial_cost, final_cost):
    if initial_cost in [INFINITO_COSTO, None] or final_cost in [INFINITO_COSTO, None]: return "N/A"
    # Handle case where initial cost is 0 or very close to 0
    try:
        init_c = float(initial_cost)
        final_c = float(final_cost)
        # Avoid division by zero or near-zero
        if abs(init_c) < 1e-9:
             # If initial is zero, improvement is infinite if final is positive, 0 if final is zero
             return "-INF%" if final_c > 1e-9 else "0.00%"

        improvement = ((init_c - final_c) / init_c) * 100
        return f"{improvement:.2f}%"
    except (ValueError, TypeError):
        return "ERR_CALC"


def get_stats_from_list(results_list):
    # Calculates statistics for a list of result dictionaries
    stats = {
        'costs': [r.get('cost', INFINITO_COSTO) for r in results_list],
        'times': [r.get('time', 0.0) for r in results_list],
        'valid_runs': 0, 'invalid_runs': 0, 'min_cost': INFINITO_COSTO,
        'max_cost': -INFINITO_COSTO, 'avg_cost': INFINITO_COSTO, 'stdev_cost': 0.0,
        'avg_time': 0.0, 'total_time': 0.0, 'best_result': None,
        # Aggregate repair success/failure if the info is present in the list items
        'successful_repairs': sum(1 for r in results_list if r.get('repair_success') is True),
        'failed_repairs': sum(1 for r in results_list if r.get('repair_success') is False)
    }
    valid_costs = [c for c in stats['costs'] if c is not None and c != INFINITO_COSTO]
    stats['valid_runs'] = len(valid_costs)
    stats['invalid_runs'] = len(results_list) - stats['valid_runs']

    if stats['valid_runs'] > 0:
        stats['min_cost'] = min(valid_costs)
        # Handle case where max() gets an empty sequence if all costs are INF
        stats['max_cost'] = max(valid_costs) if valid_costs else -INFINITO_COSTO
        stats['avg_cost'] = statistics.mean(valid_costs)
        if stats['valid_runs'] > 1:
            try:
                stats['stdev_cost'] = statistics.stdev(valid_costs)
            except statistics.StatisticsError: # Handle case with single valid run or all same values
                stats['stdev_cost'] = 0.0
        else:
            stats['stdev_cost'] = 0.0 # Standard deviation is 0 for a single data point

        # Find best result (minimum cost among valid runs)
        best_res_index = -1
        current_min = INFINITO_COSTO
        for i, res in enumerate(results_list):
            cost = res.get('cost', INFINITO_COSTO)
            if cost is not None and cost != INFINITO_COSTO:
                if cost < current_min:
                    current_min = cost
                    best_res_index = i
                # If cost is equal, keep the first one found (or add tie-breaking logic if needed)
                elif cost == current_min and best_res_index == -1:
                     best_res_index = i
        if best_res_index != -1:
             # Ensure we don't deepcopy if the list is empty or index is invalid
             if best_res_index < len(results_list):
                  stats['best_result'] = copy.deepcopy(results_list[best_res_index])
             else:
                  print(f"WARN get_stats: best_res_index {best_res_index} out of bounds for list len {len(results_list)}")


    valid_times = [t for t in stats['times'] if t is not None]
    if valid_times:
         stats['avg_time'] = statistics.mean(valid_times) if valid_times else 0.0
         stats['total_time'] = sum(valid_times)

    return stats


# --- Función de Resumen por Caso ---
def print_case_summary(case_name, results, planes_data_for_summary=None): # Added planes_data
    """Imprime un resumen detallado de los resultados para un caso."""
    print(f"\n######### Resumen Detallado: {case_name} #########")
    if planes_data_for_summary is None:
        planes_data_for_summary = [] # Avoid error if not passed

    # --- 1. Greedy Determinista ---
    print("\n--- 1. Greedy Determinista ---")
    res_d1 = results.get('deterministic_1_runway', {})
    res_d2 = results.get('deterministic_2_runways', {})
    cost_d1 = res_d1.get('cost'); time_d1 = res_d1.get('time')
    cost_d2 = res_d2.get('cost'); time_d2 = res_d2.get('time')
    print(f"  1 Pista : Costo={format_cost(cost_d1)}, Tiempo={format_time(time_d1)}")
    print(f"  2 Pistas: Costo={format_cost(cost_d2)}, Tiempo={format_time(time_d2)}")

    # --- 2. Greedy Estocástico (Resultados BRUTOS, antes de reparar) ---
    print(f"\n--- 2. Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs) ---")
    stats_s1_raw = results.get('stochastic_1_runway_stats', {}) # Using stats saved from raw runs
    stats_s2_raw = results.get('stochastic_2_runways_stats', {})
    runs_s1_raw = results.get('stochastic_1_runway_runs_raw', [])
    runs_s2_raw = results.get('stochastic_2_runway_runs_raw', [])
    total_runs_s1 = len(runs_s1_raw)
    total_runs_s2 = len(runs_s2_raw)

    # Count actual feasible runs from the raw data
    feasible_raw_s1 = sum(1 for run in runs_s1_raw if run.get('cost') != INFINITO_COSTO and all(run.get('landing_times', {}).get(pid, INFINITO_COSTO) <= planes_data_for_summary[pid][2] for pid in run.get('schedule', [])))
    feasible_raw_s2 = sum(1 for run in runs_s2_raw if run.get('cost') != INFINITO_COSTO and all(run.get('landing_times', {}).get(pid, INFINITO_COSTO) <= planes_data_for_summary[pid][2] for pid in run.get('schedule', [])))

    print(f"  1 Pista : Runs={total_runs_s1}, Factibles Pre-Rep={feasible_raw_s1} (Resultados pre-reparación)")
    if total_runs_s1 > 0:
        print(f"    Costo (Potencialmente Infactible Lk): Min={format_cost(stats_s1_raw.get('min_cost'))}, Max={format_cost(stats_s1_raw.get('max_cost'))}, Prom={format_cost(stats_s1_raw.get('avg_cost'))}, StdDev={stats_s1_raw.get('stdev_cost', 0.0):.2f}")
        print(f"    Tiempo Ejec: Prom={format_time(stats_s1_raw.get('avg_time'))}, Total={format_time(stats_s1_raw.get('total_time'))}")
        best_res_s1_raw = stats_s1_raw.get('best_result')
        if best_res_s1_raw: print(f"    Mejor Run (Seed {best_res_s1_raw.get('seed', 'N/A')}): Costo={format_cost(best_res_s1_raw.get('cost'))}")
    else: print("    Costo/Tiempo: N/A")

    print(f"  2 Pistas: Runs={total_runs_s2}, Factibles Pre-Rep={feasible_raw_s2} (Resultados pre-reparación)")
    if total_runs_s2 > 0:
        print(f"    Costo (Potencialmente Infactible Lk): Min={format_cost(stats_s2_raw.get('min_cost'))}, Max={format_cost(stats_s2_raw.get('max_cost'))}, Prom={format_cost(stats_s2_raw.get('avg_cost'))}, StdDev={stats_s2_raw.get('stdev_cost', 0.0):.2f}")
        print(f"    Tiempo Ejec: Prom={format_time(stats_s2_raw.get('avg_time'))}, Total={format_time(stats_s2_raw.get('total_time'))}")
        best_res_s2_raw = stats_s2_raw.get('best_result')
        if best_res_s2_raw: print(f"    Mejor Run (Seed {best_res_s2_raw.get('seed', 'N/A')}): Costo={format_cost(best_res_s2_raw.get('cost'))}")
    else: print("    Costo/Tiempo: N/A")


    # --- 3. Hill Climbing desde Greedy Determinista ---
    print("\n--- 3. Hill Climbing desde Greedy Determinista ---")
    res_hc_d1 = results.get('hc_from_deterministic_1_runway', {})
    res_hc_d2 = results.get('hc_from_deterministic_2_runways', {})
    cost_hc_d1 = res_hc_d1.get('cost'); time_hc_d1 = res_hc_d1.get('time')
    cost_hc_d2 = res_hc_d2.get('cost'); time_hc_d2 = res_hc_d2.get('time')
    impr_hc_d1 = calculate_improvement(cost_d1, cost_hc_d1)
    impr_hc_d2 = calculate_improvement(cost_d2, cost_hc_d2)
    print(f"  1 Pista : Costo={format_cost(cost_hc_d1)}, Tiempo={format_time(time_hc_d1)}, Mejora vs Det={impr_hc_d1}")
    print(f"  2 Pistas: Costo={format_cost(cost_hc_d2)}, Tiempo={format_time(time_hc_d2)}, Mejora vs Det={impr_hc_d2}")

    # --- 4. GRASP (Resultados finales post-reparación y post-HC) ---
    print(f"\n--- 4. GRASP ({NUM_GRASP_EXECUTIONS} ejecuciones por config. restarts) ---")
    for grasp_iters in GRASP_RESTARTS_LIST:
        grasp_runs_1r = results.get(f'grasp_{grasp_iters}iters_1r_runs', [])
        grasp_runs_2r = results.get(f'grasp_{grasp_iters}iters_2r_runs', [])
        stats_g1 = get_stats_from_list(grasp_runs_1r) # Stats sobre resultados finales (post-HC)
        stats_g2 = get_stats_from_list(grasp_runs_2r)
        # Get repair stats from the result_info returned by solve_grasp if needed
        # For now, just report valid runs
        print(f"  Config: {grasp_iters} Restarts Internos:")
        print(f"    1 Pista : Válidas Finales={stats_g1['valid_runs']}/{len(grasp_runs_1r)} (Resultados post-reparación+HC)")
        if stats_g1['valid_runs'] > 0:
            print(f"      Costo Final GRASP: Min={format_cost(stats_g1['min_cost'])}, Max={format_cost(stats_g1['max_cost'])}, Prom={format_cost(stats_g1['avg_cost'])}, StdDev={stats_g1['stdev_cost']:.2f}")
            print(f"      Tiempo GRASP Ejec: Prom={format_time(stats_g1['avg_time'])}, Total={format_time(stats_g1['total_time'])}")
            best_res_g1 = stats_g1.get('best_result')
            if best_res_g1: print(f"      Mejor Ejecución (Run idx {best_res_g1.get('run_index', 'N/A')}): Costo GRASP={format_cost(best_res_g1.get('cost'))}")
        else: print("      Costo/Tiempo GRASP: N/A (Ninguna ejecución produjo solución factible final)")

        print(f"    2 Pistas: Válidas Finales={stats_g2['valid_runs']}/{len(grasp_runs_2r)} (Resultados post-reparación+HC)")
        if stats_g2['valid_runs'] > 0:
            print(f"      Costo Final GRASP: Min={format_cost(stats_g2['min_cost'])}, Max={format_cost(stats_g2['max_cost'])}, Prom={format_cost(stats_g2['avg_cost'])}, StdDev={stats_g2['stdev_cost']:.2f}")
            print(f"      Tiempo GRASP Ejec: Prom={format_time(stats_g2['avg_time'])}, Total={format_time(stats_g2['total_time'])}")
            best_res_g2 = stats_g2.get('best_result')
            if best_res_g2: print(f"      Mejor Ejecución (Run idx {best_res_g2.get('run_index', 'N/A')}): Costo GRASP={format_cost(best_res_g2.get('cost'))}")
        else: print("      Costo/Tiempo GRASP: N/A (Ninguna ejecución produjo solución factible final)")

    # --- 5. Simulated Annealing desde Greedy Determinista ---
    print(f"\n--- 5. Simulated Annealing desde Greedy Determinista ---")
    for T_init in SA_INITIAL_TEMPS:
        res_sa_d1 = results.get(f'sa_T{T_init}_from_det_1r', {})
        res_sa_d2 = results.get(f'sa_T{T_init}_from_det_2r', {})
        cost_sa_d1 = res_sa_d1.get('cost'); time_sa_d1 = res_sa_d1.get('time')
        cost_sa_d2 = res_sa_d2.get('cost'); time_sa_d2 = res_sa_d2.get('time')
        impr_sa_d1 = calculate_improvement(cost_d1, cost_sa_d1)
        impr_sa_d2 = calculate_improvement(cost_d2, cost_sa_d2)
        print(f"  T_Inicial = {T_init}:")
        print(f"    1 Pista : Costo={format_cost(cost_sa_d1)}, Tiempo={format_time(time_sa_d1)}, Mejora vs Det={impr_sa_d1}")
        print(f"    2 Pistas: Costo={format_cost(cost_sa_d2)}, Tiempo={format_time(time_sa_d2)}, Mejora vs Det={impr_sa_d2}")

    # --- 6. Simulated Annealing desde CADA Estocástico (Resultados post-reparación y post-SA) ---
    print(f"\n--- 6. Simulated Annealing desde CADA Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs base por Temp) ---")
    for T_init in SA_INITIAL_TEMPS:
        sa_runs_s1 = results.get(f'sa_T{T_init}_from_stochastic_1r_runs', [])
        sa_runs_s2 = results.get(f'sa_T{T_init}_from_stochastic_2r_runs', [])
        stats_sa_s1 = get_stats_from_list(sa_runs_s1) # Stats sobre resultados finales (post-SA)
        stats_sa_s2 = get_stats_from_list(sa_runs_s2)
        print(f"  T_Inicial = {T_init}:")
        # Report repair success rate from stats
        rep_succ_s1 = stats_sa_s1.get('successful_repairs', 0)
        rep_fail_s1 = stats_sa_s1.get('failed_repairs', 0)
        total_rep_s1 = rep_succ_s1 + rep_fail_s1
        rep_rate_s1 = f"{rep_succ_s1}/{total_rep_s1}" if total_rep_s1 > 0 else "N/A"

        print(f"    1 Pista : Válidas Finales={stats_sa_s1['valid_runs']}/{len(sa_runs_s1)} (Reparaciones Exitosas: {rep_rate_s1})")
        if stats_sa_s1['valid_runs'] > 0:
            print(f"      Costo Final SA: Min={format_cost(stats_sa_s1['min_cost'])}, Max={format_cost(stats_sa_s1['max_cost'])}, Prom={format_cost(stats_sa_s1['avg_cost'])}, StdDev={stats_sa_s1['stdev_cost']:.2f}")
            print(f"      Tiempo SA Ejec: Prom={format_time(stats_sa_s1['avg_time'])}, Total={format_time(stats_sa_s1['total_time'])}")
            best_res_sa_s1 = stats_sa_s1.get('best_result')
            if best_res_sa_s1:
                initial_c = best_res_sa_s1.get('initial_cost', 'N/A') # Costo post-reparación
                impr = calculate_improvement(initial_c, best_res_sa_s1.get('cost'))
                print(f"      Mejor Run (Stoch Seed {best_res_sa_s1.get('initial_seed', 'N/A')}): Costo SA={format_cost(best_res_sa_s1.get('cost'))}, Mejora vs Stoch Reparado={impr}")
        else: print("      Costo/Tiempo SA: N/A (Ninguna ejecución produjo solución factible final)")

        rep_succ_s2 = stats_sa_s2.get('successful_repairs', 0)
        rep_fail_s2 = stats_sa_s2.get('failed_repairs', 0)
        total_rep_s2 = rep_succ_s2 + rep_fail_s2
        rep_rate_s2 = f"{rep_succ_s2}/{total_rep_s2}" if total_rep_s2 > 0 else "N/A"

        print(f"    2 Pistas: Válidas Finales={stats_sa_s2['valid_runs']}/{len(sa_runs_s2)} (Reparaciones Exitosas: {rep_rate_s2})")
        if stats_sa_s2['valid_runs'] > 0:
            print(f"      Costo Final SA: Min={format_cost(stats_sa_s2['min_cost'])}, Max={format_cost(stats_sa_s2['max_cost'])}, Prom={format_cost(stats_sa_s2['avg_cost'])}, StdDev={stats_sa_s2['stdev_cost']:.2f}")
            print(f"      Tiempo SA Ejec: Prom={format_time(stats_sa_s2['avg_time'])}, Total={format_time(stats_sa_s2['total_time'])}")
            best_res_sa_s2 = stats_sa_s2.get('best_result')
            if best_res_sa_s2:
                initial_c = best_res_sa_s2.get('initial_cost', 'N/A') # Costo post-reparación
                impr = calculate_improvement(initial_c, best_res_sa_s2.get('cost'))
                print(f"      Mejor Run (Stoch Seed {best_res_sa_s2.get('initial_seed', 'N/A')}): Costo SA={format_cost(best_res_sa_s2.get('cost'))}, Mejora vs Stoch Reparado={impr}")
        else: print("      Costo/Tiempo SA: N/A (Ninguna ejecución produjo solución factible final)")

    print(f"######### Fin Resumen: {case_name} #########")


if __name__ == "__main__":
    # Define planes_data globally or pass it to print_case_summary
    planes = [] # Initialize planes data list

    case_dir = './Casos'
    results_dir = './results'
    # Crear directorio si no existe
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Usar nombres de archivo específicos para esta versión con reparación
    csv_output_filename = os.path.join(results_dir, f'results_summary.csv')
    json_output_filename = os.path.join(results_dir, f'all_results_details.json')

    # --- Archivos de Caso ---
    case_files = ['case1.txt', 'case2.txt', 'case3.txt', 'case4.txt'] # Incluir case4 si existe
    # case_files = ['case3.txt'] # Probar solo case3

    all_results_detailed = {}

    # --- Preparación CSV ---
    csv_header = [
        "Caso", "Algoritmo", "Pistas", "Parametros", "Punto Partida", "Run Index/Seed",
        "Estado Final", "Factible Final", "Costo Final", "Tiempo_s", "Seed Inicial Stoch",
        "Costo Post-Reparacion", "Mejora_% (vs Post-Rep)", "Solucion Orden", "Solucion Tiempos",
        "Reparacion Exitosa" # Columna para indicar si la reparación funcionó
    ]
    try:
        with open(csv_output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)

            # --- Bucle Principal por Caso ---
            for filename in case_files:
                filepath = os.path.join(case_dir, filename)
                if not os.path.exists(filepath):
                    print(f"ADVERTENCIA: Archivo de caso no encontrado '{filepath}', saltando.")
                    continue

                print(f"\n=============== Procesando {filename} ================ ")
                case_results = {}
                try:
                    D, planes, separations_matrix = read_case_data(filepath) # Read data for the current case
                    print(f"  Número de aviones: {D}")
                    # Basic validation of read data
                    if not planes or not separations_matrix or len(planes) != D or len(separations_matrix) != D:
                         raise ValueError(f"Datos leídos inconsistentes para {filename}. D={D}, len(planes)={len(planes)}, len(sep)={len(separations_matrix)}")
                    if any(len(row) != D for row in separations_matrix):
                         raise ValueError(f"Matriz de separación no es cuadrada ({D}x{D}) para {filename}.")

                except Exception as e:
                    print(f"  ERROR leyendo o validando el archivo del caso {filename}: {e}")
                    traceback.print_exc()
                    continue # Saltar al siguiente caso

                # --- Función interna para escribir fila en CSV ---
                def write_result_to_csv(algo_name, num_runways, params, start_point, result_dict, run_index=None, repair_success_flag=None):
                    final_cost = result_dict.get('cost', INFINITO_COSTO)
                    exec_time = result_dict.get('time', 0.0)
                    # 'initial_cost' es el costo DESPUÉS de la reparación (si aplica y fue exitosa)
                    initial_cost_post_repair = result_dict.get('initial_cost')
                    initial_seed = result_dict.get('initial_seed')
                    schedule = result_dict.get('schedule', [])
                    landing_times = result_dict.get('landing_times', {})

                    # Determinar factibilidad final basada en el costo
                    is_feasible_final = (final_cost != INFINITO_COSTO and final_cost is not None)
                    status_final = "VALID" if is_feasible_final else "INVALID"
                    feasibility_final_str = "Factible" if is_feasible_final else "Infactible"

                    final_cost_str = format_cost(final_cost)
                    improvement_str = calculate_improvement(initial_cost_post_repair, final_cost) # Mejora vs costo post-reparación
                    seed_str = str(initial_seed) if initial_seed is not None else "N/A"
                    params_str = str(params) if params is not None else "N/A"
                    start_point_str = str(start_point) if start_point is not None else "N/A"
                    initial_cost_post_repair_str = format_cost(initial_cost_post_repair) if initial_cost_post_repair is not None else "N/A"
                    time_str = format_time(exec_time)
                    run_index_str = str(run_index) if run_index is not None else "N/A"
                    # Convertir booleano a string legible
                    repair_success_str = str(repair_success_flag) if repair_success_flag is not None else "N/A"

                    order_str = ",".join(map(str, schedule)) if schedule else ""
                    # Usar formato más compacto para tiempos
                    times_str = ";".join([f"{p}:{t:.1f}" for p, t in landing_times.items()]) if landing_times else ""
                    if len(times_str) > 1500: # Limitar longitud en CSV
                        times_str = times_str[:1500] + "..."


                    row_data = [
                        filename, algo_name, num_runways, params_str, start_point_str, run_index_str,
                        status_final, feasibility_final_str, final_cost_str, time_str, seed_str,
                        initial_cost_post_repair_str, improvement_str, order_str, times_str,
                        repair_success_str # Nueva columna
                    ]
                    try:
                        csv_writer.writerow(row_data)
                    except Exception as csv_e:
                        print(f"  ERROR escribiendo fila en CSV: {csv_e} - Datos: {row_data}")
                    return status_final == "VALID"

                # --- Bucle por Número de Pistas ---
                for num_runways in [1, 2]:
                    print(f"\n  --- {num_runways} Pista(s) ---")
                    runway_tag = f"{num_runways}_runway" if num_runways == 1 else f"{num_runways}_runways"
                    results_key_suffix = f"{num_runways}r"

                    # --- 1. Greedy Determinista ---
                    print(f"    1. Ejecutando Greedy Determinista {results_key_suffix}...")
                    start_time = time.time()
                    schedule_d, times_d, cost_d = solve_greedy_deterministic(D, planes, separations_matrix, num_runways)
                    time_d = time.time() - start_time
                    is_feasible_d = (cost_d != INFINITO_COSTO and cost_d is not None)
                    res_d = {'cost': cost_d, 'time': time_d, 'schedule': schedule_d, 'landing_times': times_d, 'initial_cost': cost_d, 'initial_seed': None}
                    case_results[f'deterministic_{runway_tag}'] = copy.deepcopy(res_d)
                    is_valid_d = write_result_to_csv("Greedy Deterministic", num_runways, None, "N/A", res_d, repair_success_flag=None) # No aplica reparación
                    det_sol = res_d if is_valid_d else None # Guardar solo si fue válido

                    # --- 2. Greedy Estocástico (Genera solución completa, puede ser infactible Lk) ---
                    print(f"    2. Ejecutando Greedy Estocástico {results_key_suffix} ({NUM_STOCHASTIC_RUNS} runs)...")
                    stochastic_runs_raw = [] # Guardar resultados brutos (antes de reparar)
                    for seed in range(NUM_STOCHASTIC_RUNS):
                        start_time = time.time()
                        schedule_s, times_s, cost_s_orig = solve_greedy_stochastic(D, planes, separations_matrix, num_runways, seed, RCL_SIZE)
                        time_s = time.time() - start_time
                        # Verificar factibilidad real (T <= L para todos)
                        # Nota: cost_s_orig ya es el costo real (sin penalización Lk), pero la solución puede ser infactible
                        is_feasible_s_raw = False
                        if cost_s_orig != INFINITO_COSTO and schedule_s:
                            # Recalcular para confirmar factibilidad Lk (más seguro)
                            _, _, is_feasible_s_raw = recalculate_landing_times(schedule_s, D, planes, separations_matrix, num_runways)

                        res_s_raw = {'seed': seed, 'cost': cost_s_orig, 'time': time_s, 'schedule': schedule_s, 'landing_times': times_s}
                        stochastic_runs_raw.append(res_s_raw)
                        # Escribir al CSV el resultado BRUTO del estocástico
                        write_result_to_csv("Greedy Stochastic (Raw)", num_runways, f"RCL={RCL_SIZE}", "N/A",
                                            {'cost': cost_s_orig, 'time': time_s, 'schedule':schedule_s, 'landing_times':times_s, 'initial_cost': cost_s_orig, 'initial_seed': seed},
                                            run_index=seed, repair_success_flag=is_feasible_s_raw) # Usar factibilidad real aquí

                    stats_s_raw = get_stats_from_list(stochastic_runs_raw) # Stats sobre costos originales
                    case_results[f'stochastic_{runway_tag}_runs_raw'] = copy.deepcopy(stochastic_runs_raw)
                    case_results[f'stochastic_{runway_tag}_stats'] = copy.deepcopy(stats_s_raw)

                    # --- 3. HC desde Determinista ---
                    print(f"    3. Ejecutando HC desde Determinista {results_key_suffix}...")
                    res_hc_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': cost_d, 'initial_seed': None}
                    if det_sol: # Solo si el determinista fue válido
                        start_time = time.time()
                        hc_schedule_d, hc_times_d, hc_cost_d = solve_hc_from_deterministic(
                            D, planes, separations_matrix, num_runways, det_sol, HC_MAX_ITER
                        )
                        hc_time_d = time.time() - start_time
                        # Verificar factibilidad final de HC
                        is_feasible_hc_d = (hc_cost_d != INFINITO_COSTO and hc_cost_d is not None)
                        res_hc_d = {'cost': hc_cost_d, 'time': hc_time_d, 'schedule': hc_schedule_d, 'landing_times': hc_times_d, 'initial_cost': cost_d, 'initial_seed': None}
                    case_results[f'hc_from_deterministic_{runway_tag}'] = copy.deepcopy(res_hc_d)
                    write_result_to_csv("HC", num_runways, f"HC_iter={HC_MAX_ITER}", "Deterministic", res_hc_d, repair_success_flag=None)

                    # --- 4. GRASP (Ahora usa reparación interna) ---
                    print(f"    4. Ejecutando GRASP {results_key_suffix} ({NUM_GRASP_EXECUTIONS} runs por config)...")
                    for grasp_iters in GRASP_RESTARTS_LIST:
                         grasp_runs_final = []
                         grasp_repair_stats = {'successful': 0, 'failed': 0}
                         print(f"      GRASP con {grasp_iters} restarts internos ({NUM_GRASP_EXECUTIONS} veces)...")
                         params_grasp = f"Restarts={grasp_iters}, RCL={RCL_SIZE}, HC_iter={HC_MAX_ITER}, RepairAtt={MAX_REPAIR_ATTEMPTS or '5*D'}"
                         for run_idx in range(NUM_GRASP_EXECUTIONS):
                             start_time_grasp = time.time()
                             # solve_grasp ahora hace construcción -> reparación -> HC
                             # Y devuelve información de reparación en grasp_info
                             grasp_schedule, grasp_times, grasp_cost, grasp_info = solve_grasp(
                                 D, planes, separations_matrix, num_runways,
                                 grasp_iters, RCL_SIZE, HC_MAX_ITER, MAX_REPAIR_ATTEMPTS
                             )
                             grasp_time = time.time() - start_time_grasp
                             # Acumular estadísticas de reparación de GRASP
                             grasp_repair_stats['successful'] += grasp_info.get('successful_repairs', 0)
                             grasp_repair_stats['failed'] += grasp_info.get('failed_repairs', 0)

                             # El costo de GRASP ya debería ser factible si tuvo éxito
                             is_feasible_grasp = (grasp_cost != INFINITO_COSTO and grasp_cost is not None)
                             res_grasp = {'cost': grasp_cost, 'time': grasp_time, 'schedule': grasp_schedule, 'landing_times': grasp_times,
                                          'initial_cost': None, 'initial_seed': None, 'run_index': run_idx} # initial_cost no aplica directamente aquí
                             grasp_runs_final.append(res_grasp)
                             # Escribimos el resultado final de GRASP
                             # Usamos is_feasible_grasp para indicar si el resultado final fue válido
                             write_result_to_csv("GRASP", num_runways, params_grasp, "Stoch->Repair->HC", res_grasp, run_index=run_idx, repair_success_flag=is_feasible_grasp) # Indica si el resultado final fue factible
                         case_results[f'grasp_{grasp_iters}iters_{results_key_suffix}_runs'] = copy.deepcopy(grasp_runs_final)
                         # Imprimir resumen de reparación para esta configuración de GRASP
                         total_grasp_starts = grasp_repair_stats['successful'] + grasp_repair_stats['failed']
                         # Evitar división por cero si no hubo inicios válidos
                         success_rate_str = f"{grasp_repair_stats['successful']}/{total_grasp_starts}" if total_grasp_starts > 0 else "0/0"
                         print(f"        GRASP {grasp_iters} restarts: Reparaciones Exitosas={success_rate_str}")


                    # --- 5. SA desde Determinista ---
                    print(f"    5. Ejecutando SA desde Determinista {results_key_suffix}...")
                    for T_init in SA_INITIAL_TEMPS:
                        res_sa_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': cost_d, 'initial_seed': None}
                        params_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                        if det_sol: # Solo si el determinista fue válido
                            start_time = time.time()
                            sa_schedule_d, sa_times_d, sa_cost_d = solve_simulated_annealing(
                                D, planes, separations_matrix, num_runways, det_sol, # Pasar solución determinista
                                T_init, SA_T_MIN, SA_ALPHA, SA_ITER_PER_TEMP, SA_MAX_NEIGHBOR_ATTEMPTS
                            )
                            sa_time_d = time.time() - start_time
                            # Verificar factibilidad
                            is_feasible_sa_d = (sa_cost_d != INFINITO_COSTO and sa_cost_d is not None)
                            res_sa_d = {'cost': sa_cost_d, 'time': sa_time_d, 'schedule': sa_schedule_d, 'landing_times': sa_times_d, 'initial_cost': cost_d, 'initial_seed': None}
                        case_results[f'sa_T{T_init}_from_det_{results_key_suffix}'] = copy.deepcopy(res_sa_d)
                        write_result_to_csv("SA", num_runways, params_sa, "Deterministic", res_sa_d, repair_success_flag=None) # No aplica reparación

                    # --- 6. SA desde CADA Estocástico (con Reparación PREVIA) ---
                    print(f"    6. Ejecutando SA desde CADA Estocástico {results_key_suffix} (con Reparación)...")
                    for T_init in SA_INITIAL_TEMPS:
                        sa_from_stoch_runs = []
                        params_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                        print(f"      T_Inicial = {T_init} para {len(stochastic_runs_raw)} soluciones Stoch...")

                        successful_repairs_sa = 0
                        failed_repairs_sa = 0

                        for stoch_run_data in stochastic_runs_raw:
                            initial_seed = stoch_run_data.get('seed')
                            initial_schedule_raw = stoch_run_data.get('schedule')
                            initial_times_raw = stoch_run_data.get('landing_times')
                            # initial_cost_raw = stoch_run_data.get('cost') # Costo original sin penalización Lk

                            # --- Intentar Reparar ANTES de SA ---
                            repaired_schedule, repaired_times, repaired_cost, repair_success = repair_schedule_simple_swap(
                                initial_schedule_raw, initial_times_raw,
                                D, planes, separations_matrix, num_runways, MAX_REPAIR_ATTEMPTS
                            )

                            res_sa_s = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': None, 'initial_seed': initial_seed, 'repair_success': repair_success}
                            is_feasible_sa_s = False

                            if repair_success:
                                successful_repairs_sa += 1
                                # Si la reparación funcionó, usar la solución reparada para SA
                                start_point_str = f"Stochastic (Seed {initial_seed}, Reparado)"
                                start_sol_dict = {'schedule': repaired_schedule, 'landing_times': repaired_times, 'cost': repaired_cost} # Costo factible
                                initial_cost_for_improvement_calc = repaired_cost # Usamos el costo después de reparar

                                start_time = time.time()
                                sa_schedule_s, sa_times_s, sa_cost_s = solve_simulated_annealing(
                                    D, planes, separations_matrix, num_runways, start_sol_dict, # Pasar solución reparada
                                    T_init, SA_T_MIN, SA_ALPHA, SA_ITER_PER_TEMP, SA_MAX_NEIGHBOR_ATTEMPTS
                                )
                                sa_time_s = time.time() - start_time
                                # Verificar factibilidad final de SA
                                is_feasible_sa_s = (sa_cost_s != INFINITO_COSTO and sa_cost_s is not None)

                                res_sa_s = {'cost': sa_cost_s, 'time': sa_time_s, 'schedule': sa_schedule_s, 'landing_times': sa_times_s, 'initial_cost': initial_cost_for_improvement_calc, 'initial_seed': initial_seed, 'repair_success': True}

                            else:
                                failed_repairs_sa += 1
                                # Si la reparación falló, no ejecutamos SA para esta semilla
                                start_point_str = f"Stochastic (Seed {initial_seed}, Reparación Fallida)"
                                # Dejamos res_sa_s con costo infinito y repair_success=False

                            sa_from_stoch_runs.append(res_sa_s)
                            # Escribir al CSV indicando si la reparación funcionó y el resultado de SA
                            write_result_to_csv("SA (from Stoch+Repair)", num_runways, params_sa, start_point_str, res_sa_s, run_index=initial_seed, repair_success_flag=repair_success)

                        print(f"        Reparaciones Exitosas: {successful_repairs_sa}/{len(stochastic_runs_raw)}")
                        case_results[f'sa_T{T_init}_from_stochastic_{results_key_suffix}_runs'] = copy.deepcopy(sa_from_stoch_runs)


                # --- Fin del Bucle por Pistas ---

                all_results_detailed[filename] = copy.deepcopy(case_results)
                print(f"--------------- Fin Procesamiento {filename} ---------------")
                # Llamar a la función de resumen, pasando planes_data
                print_case_summary(filename, case_results, planes_data_for_summary=planes)


            # --- Fin del Bucle por Casos ---

    except IOError as e:
        print(f"\nERROR FATAL: No se pudo abrir o escribir el archivo CSV '{csv_output_filename}'. Verifica permisos o ruta.")
        print(f"Detalle: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR FATAL durante la ejecución principal o escritura de CSV: {e}")
        traceback.print_exc()
        sys.exit(1)


    # --- Guardar Resultados Detallados en JSON ---
    print("\nGuardando resultados detallados en JSON...")
    try:
        def default_serializer(obj):
            if isinstance(obj, set): return list(obj)
            if obj == float('inf'): return "Infinity"
            if obj == float('-inf'): return "-Infinity"
            if isinstance(obj, float) and math.isnan(obj): return "NaN"
            # Attempt to serialize, fallback to string representation
            try:
                # Use a simple check for basic types
                if isinstance(obj, (dict, list, str, int, float, bool, type(None))):
                    return obj
                # Fallback for other types
                return str(obj)
            except TypeError:
                return str(obj) # Fallback if check fails for some reason

        with open(json_output_filename, 'w', encoding='utf-8') as jsonfile:
            # Use default serializer
            json.dump(all_results_detailed, jsonfile, indent=2, default=default_serializer)
        print(f"--- Resultados detallados guardados en '{json_output_filename}' ---")
    except Exception as e:
        print(f"--- ERROR al guardar resultados detallados en JSON: {e}")
        traceback.print_exc()

    # --- Resumen Final ---
    print("\n\n#############################################################")
    # total_cases = len(all_results_detailed) # This counts successfully processed cases
    print(f"Casos intentados: {len(case_files)}") # Print number of attempted cases
    print(f"Resultados detallados guardados en '{csv_output_filename}' y '{json_output_filename}'")
    print("#############################################################")

    print("\n--- Ejecución Finalizada ---")

