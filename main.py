# main.py
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
from scripts.read_case_data import read_case_data
from scripts.greedy_deterministic import solve_greedy_deterministic
from scripts.greedy_stochastic import solve_greedy_stochastic
from scripts.hill_climbing import hill_climbing_first_improvement, recalculate_landing_times
from scripts.simulated_annealing import solve_simulated_annealing
from scripts.grasp import solve_grasp

# --- Constantes y Parámetros ---
INFINITO_COSTO = float('inf')
NUM_STOCHASTIC_RUNS = 10
RCL_SIZE = 3
HC_MAX_ITER = 500
GRASP_RESTARTS_LIST = [10, 25, 50]
NUM_GRASP_EXECUTIONS = 10
SA_INITIAL_TEMPS = [10000, 5000, 1000, 500, 100]
SA_T_MIN = 0.1
SA_ALPHA = 0.95
SA_ITER_PER_TEMP = 100
SA_MAX_NEIGHBOR_ATTEMPTS = 50

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
    if initial_cost == final_cost: return "0.00%"
    try:
        init_c = float(initial_cost)
        final_c = float(final_cost)
        if init_c == 0: return "-INF%" if final_c > 0 else "0.00%"
        improvement = ((init_c - final_c) / init_c) * 100
        return f"{improvement:.2f}%"
    except (ValueError, TypeError): return "ERR_CALC"

def get_stats_from_list(results_list):
    stats = {
        'costs': [r.get('cost', INFINITO_COSTO) for r in results_list],
        'times': [r.get('time', 0.0) for r in results_list],
        'valid_runs': 0, 'invalid_runs': 0, 'min_cost': INFINITO_COSTO,
        'max_cost': -INFINITO_COSTO, 'avg_cost': INFINITO_COSTO, 'stdev_cost': 0.0,
        'avg_time': 0.0, 'total_time': 0.0, 'best_result': None
    }
    valid_costs = [c for c in stats['costs'] if c is not None and c != INFINITO_COSTO]
    stats['valid_runs'] = len(valid_costs)
    stats['invalid_runs'] = len(results_list) - stats['valid_runs']
    if stats['valid_runs'] > 0:
        stats['min_cost'] = min(valid_costs)
        stats['max_cost'] = max(valid_costs)
        stats['avg_cost'] = statistics.mean(valid_costs)
        if stats['valid_runs'] > 1:
            try: stats['stdev_cost'] = statistics.stdev(valid_costs)
            except statistics.StatisticsError: stats['stdev_cost'] = 0.0
        else: stats['stdev_cost'] = 0.0
        best_res_index = -1
        current_min = INFINITO_COSTO
        for i, res in enumerate(results_list):
            cost = res.get('cost', INFINITO_COSTO)
            if cost is not None and cost != INFINITO_COSTO:
                if cost < current_min: current_min = cost; best_res_index = i
                elif cost == current_min and best_res_index == -1: best_res_index = i
        if best_res_index != -1: stats['best_result'] = copy.deepcopy(results_list[best_res_index])
    valid_times = [t for t in stats['times'] if t is not None]
    if valid_times: stats['avg_time'] = statistics.mean(valid_times); stats['total_time'] = sum(valid_times)
    return stats


# --- Función de Resumen por Caso ---
def print_case_summary(case_name, results):
    """Imprime un resumen detallado de los resultados para un caso."""
    print(f"\n######### Resumen Detallado: {case_name} #########")
    print("\n--- 1. Greedy Determinista ---")
    res_d1 = results.get('deterministic_1_runway', {})
    res_d2 = results.get('deterministic_2_runways', {})
    cost_d1 = res_d1.get('cost'); time_d1 = res_d1.get('time')
    cost_d2 = res_d2.get('cost'); time_d2 = res_d2.get('time')
    print(f"  1 Pista : Costo={format_cost(cost_d1)}, Tiempo={format_time(time_d1)}")
    print(f"  2 Pistas: Costo={format_cost(cost_d2)}, Tiempo={format_time(time_d2)}")

    print(f"\n--- 2. Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs) ---")
    stats_s1 = results.get('stochastic_1_runway_stats', {})
    stats_s2 = results.get('stochastic_2_runways_stats', {})
    total_runs_s1 = stats_s1.get('valid_runs', 0) + stats_s1.get('invalid_runs', 0)
    total_runs_s2 = stats_s2.get('valid_runs', 0) + stats_s2.get('invalid_runs', 0)
    print(f"  1 Pista : Válidas={stats_s1.get('valid_runs', 0)}/{total_runs_s1}")
    if stats_s1.get('valid_runs', 0) > 0:
        print(f"    Costo: Min={format_cost(stats_s1.get('min_cost'))}, Max={format_cost(stats_s1.get('max_cost'))}, Prom={format_cost(stats_s1.get('avg_cost'))}, StdDev={stats_s1.get('stdev_cost', 0.0):.2f}")
        print(f"    Tiempo Ejec: Prom={format_time(stats_s1.get('avg_time'))}, Total={format_time(stats_s1.get('total_time'))}")
        best_res_s1 = stats_s1.get('best_result')
        if best_res_s1: print(f"    Mejor Run (Seed {best_res_s1.get('seed', 'N/A')}): Costo={format_cost(best_res_s1.get('cost'))}")
    else: print("    Costo/Tiempo: N/A")
    print(f"  2 Pistas: Válidas={stats_s2.get('valid_runs', 0)}/{total_runs_s2}")
    if stats_s2.get('valid_runs', 0) > 0:
        print(f"    Costo: Min={format_cost(stats_s2.get('min_cost'))}, Max={format_cost(stats_s2.get('max_cost'))}, Prom={format_cost(stats_s2.get('avg_cost'))}, StdDev={stats_s2.get('stdev_cost', 0.0):.2f}")
        print(f"    Tiempo Ejec: Prom={format_time(stats_s2.get('avg_time'))}, Total={format_time(stats_s2.get('total_time'))}")
        best_res_s2 = stats_s2.get('best_result')
        if best_res_s2: print(f"    Mejor Run (Seed {best_res_s2.get('seed', 'N/A')}): Costo={format_cost(best_res_s2.get('cost'))}")
    else: print("    Costo/Tiempo: N/A")

    print("\n--- 3. Hill Climbing desde Greedy Determinista ---")
    res_hc_d1 = results.get('hc_from_deterministic_1_runway', {})
    res_hc_d2 = results.get('hc_from_deterministic_2_runways', {})
    cost_hc_d1 = res_hc_d1.get('cost'); time_hc_d1 = res_hc_d1.get('time')
    cost_hc_d2 = res_hc_d2.get('cost'); time_hc_d2 = res_hc_d2.get('time')
    impr_hc_d1 = calculate_improvement(cost_d1, cost_hc_d1)
    impr_hc_d2 = calculate_improvement(cost_d2, cost_hc_d2)
    print(f"  1 Pista : Costo={format_cost(cost_hc_d1)}, Tiempo={format_time(time_hc_d1)}, Mejora vs Det={impr_hc_d1}")
    print(f"  2 Pistas: Costo={format_cost(cost_hc_d2)}, Tiempo={format_time(time_hc_d2)}, Mejora vs Det={impr_hc_d2}")

    print(f"\n--- 4. GRASP ({NUM_GRASP_EXECUTIONS} ejecuciones por config. restarts) ---")
    for grasp_iters in GRASP_RESTARTS_LIST:
        grasp_runs_1r = results.get(f'grasp_{grasp_iters}iters_1r_runs', [])
        grasp_runs_2r = results.get(f'grasp_{grasp_iters}iters_2r_runs', [])
        stats_g1 = get_stats_from_list(grasp_runs_1r)
        stats_g2 = get_stats_from_list(grasp_runs_2r)
        print(f"  Config: {grasp_iters} Restarts Internos:")
        print(f"    1 Pista : Válidas={stats_g1['valid_runs']}/{len(grasp_runs_1r)}")
        if stats_g1['valid_runs'] > 0:
            print(f"      Costo Final GRASP: Min={format_cost(stats_g1['min_cost'])}, Max={format_cost(stats_g1['max_cost'])}, Prom={format_cost(stats_g1['avg_cost'])}, StdDev={stats_g1['stdev_cost']:.2f}")
            print(f"      Tiempo GRASP Ejec: Prom={format_time(stats_g1['avg_time'])}, Total={format_time(stats_g1['total_time'])}")
            best_res_g1 = stats_g1.get('best_result')
            if best_res_g1: print(f"      Mejor Ejecución (Run idx {best_res_g1.get('run_index', 'N/A')}): Costo GRASP={format_cost(best_res_g1.get('cost'))}")
        else: print("      Costo/Tiempo GRASP: N/A")
        print(f"    2 Pistas: Válidas={stats_g2['valid_runs']}/{len(grasp_runs_2r)}")
        if stats_g2['valid_runs'] > 0:
            print(f"      Costo Final GRASP: Min={format_cost(stats_g2['min_cost'])}, Max={format_cost(stats_g2['max_cost'])}, Prom={format_cost(stats_g2['avg_cost'])}, StdDev={stats_g2['stdev_cost']:.2f}")
            print(f"      Tiempo GRASP Ejec: Prom={format_time(stats_g2['avg_time'])}, Total={format_time(stats_g2['total_time'])}")
            best_res_g2 = stats_g2.get('best_result')
            if best_res_g2: print(f"      Mejor Ejecución (Run idx {best_res_g2.get('run_index', 'N/A')}): Costo GRASP={format_cost(best_res_g2.get('cost'))}")
        else: print("      Costo/Tiempo GRASP: N/A")

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

    print(f"\n--- 6. Simulated Annealing desde CADA Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs base por Temp) ---")
    for T_init in SA_INITIAL_TEMPS:
        sa_runs_s1 = results.get(f'sa_T{T_init}_from_stochastic_1r_runs', [])
        sa_runs_s2 = results.get(f'sa_T{T_init}_from_stochastic_2r_runs', [])
        stats_sa_s1 = get_stats_from_list(sa_runs_s1)
        stats_sa_s2 = get_stats_from_list(sa_runs_s2)
        print(f"  T_Inicial = {T_init}:")
        print(f"    1 Pista : Válidas={stats_sa_s1['valid_runs']}/{len(sa_runs_s1)}")
        if stats_sa_s1['valid_runs'] > 0:
            print(f"      Costo Final SA: Min={format_cost(stats_sa_s1['min_cost'])}, Max={format_cost(stats_sa_s1['max_cost'])}, Prom={format_cost(stats_sa_s1['avg_cost'])}, StdDev={stats_sa_s1['stdev_cost']:.2f}")
            print(f"      Tiempo SA Ejec: Prom={format_time(stats_sa_s1['avg_time'])}, Total={format_time(stats_sa_s1['total_time'])}")
            best_res_sa_s1 = stats_sa_s1.get('best_result')
            if best_res_sa_s1:
                initial_c = best_res_sa_s1.get('initial_cost', 'N/A')
                impr = calculate_improvement(initial_c, best_res_sa_s1.get('cost'))
                print(f"      Mejor Run (Stoch Seed {best_res_sa_s1.get('initial_seed', 'N/A')}): Costo SA={format_cost(best_res_sa_s1.get('cost'))}, Mejora vs Stoch Ini={impr}")
        else: print("      Costo/Tiempo SA: N/A")
        print(f"    2 Pistas: Válidas={stats_sa_s2['valid_runs']}/{len(sa_runs_s2)}")
        if stats_sa_s2['valid_runs'] > 0:
            print(f"      Costo Final SA: Min={format_cost(stats_sa_s2['min_cost'])}, Max={format_cost(stats_sa_s2['max_cost'])}, Prom={format_cost(stats_sa_s2['avg_cost'])}, StdDev={stats_sa_s2['stdev_cost']:.2f}")
            print(f"      Tiempo SA Ejec: Prom={format_time(stats_sa_s2['avg_time'])}, Total={format_time(stats_sa_s2['total_time'])}")
            best_res_sa_s2 = stats_sa_s2.get('best_result')
            if best_res_sa_s2:
                initial_c = best_res_sa_s2.get('initial_cost', 'N/A')
                impr = calculate_improvement(initial_c, best_res_sa_s2.get('cost'))
                print(f"      Mejor Run (Stoch Seed {best_res_sa_s2.get('initial_seed', 'N/A')}): Costo SA={format_cost(best_res_sa_s2.get('cost'))}, Mejora vs Stoch Ini={impr}")
        else: print("      Costo/Tiempo SA: N/A")

    print(f"######### Fin Resumen: {case_name} #########")


if __name__ == "__main__":
    case_dir = './Casos'
    results_dir = './results'
    csv_output_filename = os.path.join(results_dir, f'results_summary.csv')
    json_output_filename = os.path.join(results_dir, f'all_results_details.json')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # --- Encontrar Archivos de Caso ---
    try:
        case_files = sorted([f for f in os.listdir(case_dir) if f.startswith('case') and f.endswith('.txt')])
        if not case_files:
            print(f"ADVERTENCIA: No se encontraron archivos 'case*.txt' en el directorio '{case_dir}'")
            sys.exit(0)
    except FileNotFoundError:
        print(f"ERROR: El directorio de casos '{case_dir}' no existe.")
        sys.exit(1)

    print(f"Archivos de caso encontrados para procesar: {case_files}\n")

    all_results_detailed = {}

    # --- Preparación CSV ---
    # *** AÑADIR NUEVAS COLUMNAS AL HEADER ***
    csv_header = [
        "Caso", "Algoritmo", "Pistas", "Parametros", "Punto Partida", "Run Index",
        "Estado", "Factible", "Costo Final", "Tiempo_s", "Seed Inicial", # Añadido "Factible"
        "Costo Inicial", "Mejora_%", "Solucion Orden", "Solucion Tiempos" # Añadido Orden y Tiempos
    ]
    try:
        with open(csv_output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)

            # --- Bucle Principal por Caso ---
            for filename in case_files:
                filepath = os.path.join(case_dir, filename)
                print(f"\n=============== Procesando {filename} ================ ")
                case_results = {}
                try:
                    D, planes, separations_matrix = read_case_data(filepath)
                    print(f"  Número de aviones: {D}")
                except Exception as e:
                    print(f"  ERROR leyendo el archivo del caso {filename}: {e}")
                    traceback.print_exc()
                    continue

                # --- Función interna para escribir fila en CSV (MODIFICADA) ---
                def write_result_to_csv(algo_name, num_runways, params, start_point, result_dict, run_index=None, is_feasible=None):
                    final_cost = result_dict.get('cost', INFINITO_COSTO)
                    exec_time = result_dict.get('time', 0.0)
                    initial_cost = result_dict.get('initial_cost')
                    initial_seed = result_dict.get('initial_seed')
                    schedule = result_dict.get('schedule', [])
                    landing_times = result_dict.get('landing_times', {})

                    status = "INVALID" if final_cost == INFINITO_COSTO or final_cost is None else "VALID"
                    # *** AÑADIR COLUMNA FACTIBLE ***
                    feasibility_str = "Factible" if is_feasible else "Infactible" if is_feasible is not None else "N/A"

                    final_cost_str = format_cost(final_cost)
                    improvement_str = calculate_improvement(initial_cost, final_cost)
                    seed_str = str(initial_seed) if initial_seed is not None else "N/A"
                    params_str = str(params) if params is not None else "N/A"
                    start_point_str = str(start_point) if start_point is not None else "N/A"
                    initial_cost_str = format_cost(initial_cost) if initial_cost is not None else "N/A"
                    time_str = format_time(exec_time)
                    run_index_str = str(run_index) if run_index is not None else "N/A"

                    # *** FORMATEAR SOLUCIÓN PARA CSV ***
                    # Orden: Lista de IDs como string separado por comas
                    order_str = ",".join(map(str, schedule)) if schedule else ""
                    # Tiempos: Diccionario como string JSON o similar
                    # Usar repr() puede ser simple, o un formato más específico
                    # Aquí usamos un formato más legible: ID1:Tiempo1,ID2:Tiempo2...
                    times_str = ",".join([f"{p}:{format_cost(t)}" for p, t in landing_times.items()]) if landing_times else ""


                    # *** AÑADIR NUEVOS DATOS A LA FILA ***
                    row_data = [
                        filename, algo_name, num_runways, params_str, start_point_str, run_index_str,
                        status, feasibility_str, final_cost_str, time_str, seed_str, # feasibility_str añadida
                        initial_cost_str, improvement_str, order_str, times_str # order_str y times_str añadidos
                    ]
                    try:
                        csv_writer.writerow(row_data)
                    except Exception as csv_e:
                        print(f"  ERROR escribiendo fila en CSV: {csv_e} - Datos: {row_data}")
                    return status == "VALID"

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
                    res_d = {'cost': cost_d, 'time': time_d, 'schedule': schedule_d, 'landing_times': times_d, 'initial_cost': cost_d, 'initial_seed': None}
                    # Verificar factibilidad de la solución final
                    _, _, is_feasible_d = recalculate_landing_times(schedule_d, D, planes, separations_matrix, num_runways) if schedule_d else (None, INFINITO_COSTO, False)
                    case_results[f'deterministic_{runway_tag}'] = copy.deepcopy(res_d)
                    is_valid_d = write_result_to_csv("Greedy Deterministic", num_runways, None, "N/A", res_d, is_feasible=is_feasible_d)
                    det_sol = res_d if is_valid_d else None

                    # --- 2. Greedy Estocástico ---
                    print(f"    2. Ejecutando Greedy Estocástico {results_key_suffix} ({NUM_STOCHASTIC_RUNS} runs)...")
                    stochastic_runs = []
                    for seed in range(NUM_STOCHASTIC_RUNS):
                        start_time = time.time()
                        # Usar la versión importada (asumiendo que es la correcta)
                        schedule_s, times_s, cost_s = solve_greedy_stochastic(D, planes, separations_matrix, num_runways, seed, RCL_SIZE)
                        time_s = time.time() - start_time
                        res_s = {'seed': seed, 'cost': cost_s, 'time': time_s, 'schedule': schedule_s, 'landing_times': times_s, 'initial_cost': cost_s, 'initial_seed': seed}
                        # Verificar factibilidad
                        _, _, is_feasible_s = recalculate_landing_times(schedule_s, D, planes, separations_matrix, num_runways) if schedule_s else (None, INFINITO_COSTO, False)
                        stochastic_runs.append(res_s)
                        write_result_to_csv("Greedy Stochastic", num_runways, f"RCL={RCL_SIZE}", "N/A", res_s, is_feasible=is_feasible_s)
                    stats_s = get_stats_from_list(stochastic_runs)
                    case_results[f'stochastic_{runway_tag}_runs'] = copy.deepcopy(stochastic_runs)
                    case_results[f'stochastic_{runway_tag}_stats'] = copy.deepcopy(stats_s)
                    valid_stochastic_runs = [run for run in stochastic_runs if run.get('cost') is not None and run['cost'] != INFINITO_COSTO]

                    # --- 3. HC desde Determinista ---
                    print(f"    3. Ejecutando HC desde Determinista {results_key_suffix}...")
                    initial_cost_hc_d = cost_d
                    res_hc_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': initial_cost_hc_d, 'initial_seed': None}
                    is_feasible_hc_d = False
                    if det_sol:
                        start_time = time.time()
                        hc_schedule_d, hc_times_d, hc_cost_d = hill_climbing_first_improvement(
                            det_sol['schedule'], det_sol['landing_times'], det_sol['cost'],
                            D, planes, separations_matrix, num_runways, HC_MAX_ITER
                        )
                        hc_time_d = time.time() - start_time
                        # Verificar factibilidad de la *solución final* de HC
                        _, _, is_feasible_hc_d = recalculate_landing_times(hc_schedule_d, D, planes, separations_matrix, num_runways) if hc_schedule_d else (None, INFINITO_COSTO, False)
                        res_hc_d = {'cost': hc_cost_d, 'time': hc_time_d, 'schedule': hc_schedule_d, 'landing_times': hc_times_d, 'initial_cost': initial_cost_hc_d, 'initial_seed': None}
                    case_results[f'hc_from_deterministic_{runway_tag}'] = copy.deepcopy(res_hc_d)
                    write_result_to_csv("HC", num_runways, f"HC_iter={HC_MAX_ITER}", "Deterministic", res_hc_d, is_feasible=is_feasible_hc_d)

                    # --- 4. GRASP (Literal: 10 ejecuciones por config) ---
                    print(f"    4. Ejecutando GRASP {results_key_suffix} ({NUM_GRASP_EXECUTIONS} runs por config)...")
                    for grasp_iters in GRASP_RESTARTS_LIST:
                         grasp_runs = []
                         print(f"      GRASP con {grasp_iters} restarts internos ({NUM_GRASP_EXECUTIONS} veces)...")
                         params_grasp = f"Restarts={grasp_iters}, RCL={RCL_SIZE}, HC_iter={HC_MAX_ITER}"
                         for run_idx in range(NUM_GRASP_EXECUTIONS):
                             start_time_grasp = time.time()
                             grasp_schedule, grasp_times, grasp_cost = solve_grasp(
                                 D, planes, separations_matrix, num_runways,
                                 grasp_iters, RCL_SIZE, HC_MAX_ITER
                             )
                             grasp_time = time.time() - start_time_grasp
                             # Verificar factibilidad de la solución final de GRASP
                             _, _, is_feasible_grasp = recalculate_landing_times(grasp_schedule, D, planes, separations_matrix, num_runways) if grasp_schedule else (None, INFINITO_COSTO, False)
                             res_grasp = {'cost': grasp_cost, 'time': grasp_time, 'schedule': grasp_schedule, 'landing_times': grasp_times,
                                          'initial_cost': None, 'initial_seed': None, 'run_index': run_idx}
                             grasp_runs.append(res_grasp)
                             write_result_to_csv("GRASP", num_runways, params_grasp, "Stoch (Internal)", res_grasp, run_index=run_idx, is_feasible=is_feasible_grasp)
                         case_results[f'grasp_{grasp_iters}iters_{results_key_suffix}_runs'] = copy.deepcopy(grasp_runs)

                    # --- 5. SA desde Determinista ---
                    print(f"    5. Ejecutando SA desde Determinista {results_key_suffix}...")
                    initial_cost_sa_d = cost_d
                    for T_init in SA_INITIAL_TEMPS:
                        res_sa_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': initial_cost_sa_d, 'initial_seed': None}
                        is_feasible_sa_d = False
                        params_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                        if det_sol:
                            start_time = time.time()
                            sa_schedule_d, sa_times_d, sa_cost_d = solve_simulated_annealing(
                                D, planes, separations_matrix, num_runways, det_sol,
                                T_init, SA_T_MIN, SA_ALPHA, SA_ITER_PER_TEMP, SA_MAX_NEIGHBOR_ATTEMPTS
                            )
                            sa_time_d = time.time() - start_time
                            # Verificar factibilidad
                            _, _, is_feasible_sa_d = recalculate_landing_times(sa_schedule_d, D, planes, separations_matrix, num_runways) if sa_schedule_d else (None, INFINITO_COSTO, False)
                            res_sa_d = {'cost': sa_cost_d, 'time': sa_time_d, 'schedule': sa_schedule_d, 'landing_times': sa_times_d, 'initial_cost': initial_cost_sa_d, 'initial_seed': None}
                        case_results[f'sa_T{T_init}_from_det_{results_key_suffix}'] = copy.deepcopy(res_sa_d)
                        write_result_to_csv("SA", num_runways, params_sa, "Deterministic", res_sa_d, is_feasible=is_feasible_sa_d)

                    # --- 6. SA desde CADA Estocástica Válida ---
                    print(f"    6. Ejecutando SA desde CADA Estocástica {results_key_suffix}...")
                    for T_init in SA_INITIAL_TEMPS:
                        sa_from_stoch_runs = []
                        params_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                        print(f"      T_Inicial = {T_init} para {len(valid_stochastic_runs)} soluciones Stoch válidas...")
                        for stoch_run in valid_stochastic_runs:
                            initial_seed = stoch_run.get('seed')
                            initial_cost_s = stoch_run.get('cost')
                            start_point_str = f"Stochastic (Seed {initial_seed})"
                            start_time = time.time()
                            sa_schedule_s, sa_times_s, sa_cost_s = solve_simulated_annealing(
                                D, planes, separations_matrix, num_runways, stoch_run,
                                T_init, SA_T_MIN, SA_ALPHA, SA_ITER_PER_TEMP, SA_MAX_NEIGHBOR_ATTEMPTS
                            )
                            sa_time_s = time.time() - start_time
                            # Verificar factibilidad
                            _, _, is_feasible_sa_s = recalculate_landing_times(sa_schedule_s, D, planes, separations_matrix, num_runways) if sa_schedule_s else (None, INFINITO_COSTO, False)
                            res_sa_s = {'cost': sa_cost_s, 'time': sa_time_s, 'schedule': sa_schedule_s, 'landing_times': sa_times_s, 'initial_cost': initial_cost_s, 'initial_seed': initial_seed}
                            sa_from_stoch_runs.append(res_sa_s)
                            write_result_to_csv("SA", num_runways, params_sa, start_point_str, res_sa_s, is_feasible=is_feasible_sa_s)
                        case_results[f'sa_T{T_init}_from_stochastic_{results_key_suffix}_runs'] = copy.deepcopy(sa_from_stoch_runs)

                # --- Fin del Bucle por Pistas ---

                all_results_detailed[filename] = copy.deepcopy(case_results)
                print(f"--------------- Fin Procesamiento {filename} ---------------")
                # Llamar a la función de resumen (opcionalmente añadir factibilidad aquí también)
                print_case_summary(filename, case_results)

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
            # Manejar np.inf y np.nan si existieran (aunque usamos float('inf'))
            if obj == float('inf'): return "Infinity"
            if obj == float('-inf'): return "-Infinity"
            if isinstance(obj, float) and math.isnan(obj): return "NaN"
            try: json.dumps(obj); return obj
            except TypeError: return str(obj)

        with open(json_output_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(copy.deepcopy(all_results_detailed), jsonfile, indent=2, default=default_serializer)
        print(f"--- Resultados detallados guardados en '{json_output_filename}' ---")
    except Exception as e:
        print(f"--- ERROR al guardar resultados detallados en JSON: {e}")
        traceback.print_exc()

    # --- Resumen Final ---
    print("\n\n#############################################################")
    total_cases = len(all_results_detailed)
    print(f"Casos procesados: {total_cases}")
    print("Resultados detallados (incluyendo factibilidad y soluciones) guardados en CSV y JSON.")
    print("#############################################################")

    print("\n--- Ejecución Finalizada ---")