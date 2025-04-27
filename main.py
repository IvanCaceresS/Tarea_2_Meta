import os
import math
import time
import statistics
import csv
import json
import random
import copy

# --- Módulos de Algoritmos (Asegúrate que estén en la carpeta 'scripts') ---
try:
    from scripts.read_case_data import read_case_data
    from scripts.greedy_deterministic import solve_greedy_deterministic
    from scripts.greedy_stochastic import solve_greedy_stochastic
    # calculate_cost no se usa directamente aquí, pero sí en los otros módulos
    from scripts.hill_climbing import hill_climbing_first_improvement
    # solve_grasp no se usará directamente según la nueva lógica, solo HC y SA
    # from scripts.grasp import solve_grasp
    from scripts.simulated_annealing import solve_simulated_annealing
except ImportError as e:
    print(f"Error importando módulos: {e}")
    print("Asegúrate que los archivos .py estén en una carpeta llamada 'scripts' al mismo nivel que este script,")
    print("o ajusta las rutas de importación ('from scripts...') según tu estructura.")
    import sys
    sys.exit(1)

# --- Constantes y Parámetros ---
INFINITO_COSTO = float('inf')
NUM_STOCHASTIC_RUNS = 10 # Ejecuciones para Greedy Estocástico
# Los siguientes son solo para etiquetar las 3 series de ejecuciones de HC desde Estocástico según tu plan.
# HC_MAX_ITER es la única configuración real de HC que se usa.
HC_FROM_STOCH_LABELS = [10, 25, 50]
RCL_SIZE = 3 # Para Greedy Estocástico
HC_MAX_ITER = 500 # Iteraciones MÁXIMAS para CADA llamada a HC
SA_INITIAL_TEMPS = [10000, 5000, 1000, 500, 100] # Temperaturas iniciales para SA
SA_T_MIN = 0.1
SA_ALPHA = 0.95
SA_ITER_PER_TEMP = 100 # Iteraciones por nivel de temperatura en SA
SA_MAX_NEIGHBOR_ATTEMPTS = 50 # Intentos para generar vecino factible en SA

# --- Funciones Auxiliares de Formato y Cálculo ---

def format_cost(cost):
    """Formatea el costo para mostrar 'INF' o un número con 2 decimales."""
    if cost == INFINITO_COSTO or cost is None:
        return "INF"
    try:
        return f"{float(cost):.2f}"
    except (ValueError, TypeError):
        return "ERR_FMT"

def format_time(exec_time):
    """Formatea el tiempo de ejecución a 4 decimales."""
    if exec_time is None:
        return "N/A"
    try:
        return f"{float(exec_time):.4f}s"
    except (ValueError, TypeError):
        return "ERR_FMT"

def calculate_improvement(initial_cost, final_cost):
    """Calcula el porcentaje de mejora. Retorna 'N/A' si no aplica o hay error."""
    if initial_cost in [INFINITO_COSTO, None] or final_cost in [INFINITO_COSTO, None]:
        return "N/A"
    # Asegurarse que initial_cost no sea 0 antes de dividir
    if initial_cost == 0:
         # Si el costo inicial es 0, cualquier costo final > 0 es empeoramiento infinito
         # Si el costo final también es 0, la mejora es 0%
         return "-INF%" if final_cost > 0 else "0.00%"
    try:
        init_c = float(initial_cost)
        final_c = float(final_cost)
        improvement = ((init_c - final_c) / init_c) * 100
        return f"{improvement:.2f}%"
    except (ValueError, TypeError):
        return "ERR_CALC"

def get_stats_from_list(results_list):
    """Calcula estadísticas (costo y tiempo) para una lista de resultados."""
    stats = {
        'costs': [r.get('cost', INFINITO_COSTO) for r in results_list],
        'times': [r.get('time', 0) for r in results_list], # Asume que time=0 si falta
        'valid_runs': 0,
        'invalid_runs': 0,
        'min_cost': INFINITO_COSTO,
        'max_cost': INFINITO_COSTO,
        'avg_cost': INFINITO_COSTO,
        'stdev_cost': 0.0,
        'avg_time': 0.0,
        'total_time': 0.0,
        'best_result': None # El dict completo del mejor resultado
    }
    # Filtrar costos None o INF
    valid_costs = [c for c in stats['costs'] if c is not None and c != INFINITO_COSTO]
    stats['valid_runs'] = len(valid_costs)
    stats['invalid_runs'] = len(results_list) - stats['valid_runs']

    if stats['valid_runs'] > 0:
        stats['min_cost'] = min(valid_costs)
        stats['max_cost'] = max(valid_costs)
        stats['avg_cost'] = statistics.mean(valid_costs)
        if stats['valid_runs'] > 1:
            try:
                stats['stdev_cost'] = statistics.stdev(valid_costs)
            except statistics.StatisticsError:
                 stats['stdev_cost'] = 0.0 # Stdev de un solo elemento es 0

        # Encontrar el mejor resultado (menor costo válido)
        best_res_index = -1
        current_min = INFINITO_COSTO
        for i, res in enumerate(results_list):
            cost = res.get('cost', INFINITO_COSTO)
            if cost is not None and cost != INFINITO_COSTO:
                if cost < current_min:
                    current_min = cost
                    best_res_index = i
                # Desempate (opcional): si el costo es igual, preferir el primero encontrado
                elif cost == current_min and best_res_index == -1:
                     best_res_index = i

        if best_res_index != -1:
             # Crear una copia para evitar modificar el original al añadir stats
             stats['best_result'] = copy.deepcopy(results_list[best_res_index])


    # Filtrar tiempos None y calcular promedio y total
    valid_times = [t for t in stats['times'] if t is not None]
    if valid_times:
         stats['avg_time'] = statistics.mean(valid_times) if valid_times else 0.0
         stats['total_time'] = sum(valid_times)

    return stats


# --- Función de Resumen por Caso (Adaptada al plan exacto) ---
def print_case_summary(case_name, results):
    """Imprime un resumen detallado y estadístico para un caso específico según el plan exacto."""
    print(f"\n######### Resumen Detallado: {case_name} (Plan Exacto v2) #########")

    # --- 1. Greedy Determinista ---
    print("\n--- 1. Greedy Determinista ---")
    res_d1 = results.get('deterministic_1_runway', {})
    res_d2 = results.get('deterministic_2_runways', {})
    cost_d1 = res_d1.get('cost')
    time_d1 = res_d1.get('time')
    cost_d2 = res_d2.get('cost')
    time_d2 = res_d2.get('time')
    print(f"  1 Pista : Costo={format_cost(cost_d1)}, Tiempo={format_time(time_d1)}")
    print(f"  2 Pistas: Costo={format_cost(cost_d2)}, Tiempo={format_time(time_d2)}")

    # --- 2. Greedy Estocástico (10 runs) ---
    print(f"\n--- 2. Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs) ---")
    runs_s1 = results.get('stochastic_1_runway_runs', [])
    runs_s2 = results.get('stochastic_2_runway_runs', [])
    stats_s1 = get_stats_from_list(runs_s1)
    stats_s2 = get_stats_from_list(runs_s2)

    print(f"  1 Pista : Válidas={stats_s1['valid_runs']}/{len(runs_s1)}")
    if stats_s1['valid_runs'] > 0:
        print(f"    Costo: Min={format_cost(stats_s1['min_cost'])}, Max={format_cost(stats_s1['max_cost'])}, Prom={format_cost(stats_s1['avg_cost'])}, StdDev={stats_s1['stdev_cost']:.2f}")
        print(f"    Tiempo Ejec: Prom={format_time(stats_s1['avg_time'])}, Total={format_time(stats_s1['total_time'])}")
        if stats_s1['best_result']: print(f"    Mejor Run (Seed {stats_s1['best_result'].get('seed', 'N/A')}): Costo={format_cost(stats_s1['min_cost'])}")
    else: print("    Costo/Tiempo: N/A (Ninguna ejecución válida)")

    print(f"  2 Pistas: Válidas={stats_s2['valid_runs']}/{len(runs_s2)}")
    if stats_s2['valid_runs'] > 0:
        print(f"    Costo: Min={format_cost(stats_s2['min_cost'])}, Max={format_cost(stats_s2['max_cost'])}, Prom={format_cost(stats_s2['avg_cost'])}, StdDev={stats_s2['stdev_cost']:.2f}")
        print(f"    Tiempo Ejec: Prom={format_time(stats_s2['avg_time'])}, Total={format_time(stats_s2['total_time'])}")
        if stats_s2['best_result']: print(f"    Mejor Run (Seed {stats_s2['best_result'].get('seed', 'N/A')}): Costo={format_cost(stats_s2['min_cost'])}")
    else: print("    Costo/Tiempo: N/A (Ninguna ejecución válida)")


    # --- 3. HC desde Determinista (Equivalente a "GRASP 0 restarts") ---
    print("\n--- 3. Hill Climbing desde Greedy Determinista ---")
    res_hc_d1 = results.get('hc_from_deterministic_1_runway', {})
    res_hc_d2 = results.get('hc_from_deterministic_2_runways', {})
    cost_hc_d1 = res_hc_d1.get('cost')
    time_hc_d1 = res_hc_d1.get('time')
    cost_hc_d2 = res_hc_d2.get('cost')
    time_hc_d2 = res_hc_d2.get('time')
    impr_hc_d1 = calculate_improvement(cost_d1, cost_hc_d1)
    impr_hc_d2 = calculate_improvement(cost_d2, cost_hc_d2)
    print(f"  1 Pista : Costo={format_cost(cost_hc_d1)}, Tiempo={format_time(time_hc_d1)}, Mejora vs Det={impr_hc_d1}")
    print(f"  2 Pistas: Costo={format_cost(cost_hc_d2)}, Tiempo={format_time(time_hc_d2)}, Mejora vs Det={impr_hc_d2}")


    # --- 4. HC desde CADA Estocástica (Interpretación de "GRASP desde Estocástico") ---
    print(f"\n--- 4. Hill Climbing desde CADA Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs base) ---")
    # Las etiquetas 10, 25, 50 se usan para agrupar las 3 series de ejecuciones de HC
    for hc_iter_label in HC_FROM_STOCH_LABELS:
        hc_runs_s1 = results.get(f'hc_from_stochastic_{hc_iter_label}_1r_runs', [])
        hc_runs_s2 = results.get(f'hc_from_stochastic_{hc_iter_label}_2r_runs', [])
        stats_hc_s1 = get_stats_from_list(hc_runs_s1)
        stats_hc_s2 = get_stats_from_list(hc_runs_s2)
        print(f"  Configuración HC (Label '{hc_iter_label}', MaxIter={HC_MAX_ITER}):")

        print(f"    1 Pista : Válidas={stats_hc_s1['valid_runs']}/{len(hc_runs_s1)}")
        if stats_hc_s1['valid_runs'] > 0:
            print(f"      Costo Final HC: Min={format_cost(stats_hc_s1['min_cost'])}, Max={format_cost(stats_hc_s1['max_cost'])}, Prom={format_cost(stats_hc_s1['avg_cost'])}, StdDev={stats_hc_s1['stdev_cost']:.2f}")
            print(f"      Tiempo HC Ejec: Prom={format_time(stats_hc_s1['avg_time'])}, Total={format_time(stats_hc_s1['total_time'])}")
            if stats_hc_s1['best_result']:
                 initial_c = stats_hc_s1['best_result'].get('initial_cost', 'N/A')
                 impr = calculate_improvement(initial_c, stats_hc_s1['min_cost'])
                 print(f"      Mejor Run (Stoch Seed {stats_hc_s1['best_result'].get('initial_seed', 'N/A')}): Costo HC={format_cost(stats_hc_s1['min_cost'])}, Mejora vs Stoch Ini={impr}")
        else: print("      Costo/Tiempo HC: N/A (Ninguna ejecución válida)")

        print(f"    2 Pistas: Válidas={stats_hc_s2['valid_runs']}/{len(hc_runs_s2)}")
        if stats_hc_s2['valid_runs'] > 0:
            print(f"      Costo Final HC: Min={format_cost(stats_hc_s2['min_cost'])}, Max={format_cost(stats_hc_s2['max_cost'])}, Prom={format_cost(stats_hc_s2['avg_cost'])}, StdDev={stats_hc_s2['stdev_cost']:.2f}")
            print(f"      Tiempo HC Ejec: Prom={format_time(stats_hc_s2['avg_time'])}, Total={format_time(stats_hc_s2['total_time'])}")
            if stats_hc_s2['best_result']:
                 initial_c = stats_hc_s2['best_result'].get('initial_cost', 'N/A')
                 impr = calculate_improvement(initial_c, stats_hc_s2['min_cost'])
                 print(f"      Mejor Run (Stoch Seed {stats_hc_s2['best_result'].get('initial_seed', 'N/A')}): Costo HC={format_cost(stats_hc_s2['min_cost'])}, Mejora vs Stoch Ini={impr}")
        else: print("      Costo/Tiempo HC: N/A (Ninguna ejecución válida)")

    # --- 5. SA desde Determinista (5 Temps) ---
    print(f"\n--- 5. Simulated Annealing desde Greedy Determinista ---")
    for T_init in SA_INITIAL_TEMPS:
        res_sa_d1 = results.get(f'sa_T{T_init}_from_det_1r', {})
        res_sa_d2 = results.get(f'sa_T{T_init}_from_det_2r', {})
        cost_sa_d1 = res_sa_d1.get('cost')
        time_sa_d1 = res_sa_d1.get('time')
        cost_sa_d2 = res_sa_d2.get('cost')
        time_sa_d2 = res_sa_d2.get('time')
        impr_sa_d1 = calculate_improvement(cost_d1, cost_sa_d1)
        impr_sa_d2 = calculate_improvement(cost_d2, cost_sa_d2)
        print(f"  T_Inicial = {T_init}:")
        print(f"    1 Pista : Costo={format_cost(cost_sa_d1)}, Tiempo={format_time(time_sa_d1)}, Mejora vs Det={impr_sa_d1}")
        print(f"    2 Pistas: Costo={format_cost(cost_sa_d2)}, Tiempo={format_time(time_sa_d2)}, Mejora vs Det={impr_sa_d2}")


    # --- 6. SA desde CADA Estocástica (5 Temps x 10 runs) ---
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
            if stats_sa_s1['best_result']:
                 initial_c = stats_sa_s1['best_result'].get('initial_cost', 'N/A')
                 impr = calculate_improvement(initial_c, stats_sa_s1['min_cost'])
                 print(f"      Mejor Run (Stoch Seed {stats_sa_s1['best_result'].get('initial_seed', 'N/A')}): Costo SA={format_cost(stats_sa_s1['min_cost'])}, Mejora vs Stoch Ini={impr}")
        else: print("      Costo/Tiempo SA: N/A (Ninguna ejecución válida)")

        print(f"    2 Pistas: Válidas={stats_sa_s2['valid_runs']}/{len(sa_runs_s2)}")
        if stats_sa_s2['valid_runs'] > 0:
            print(f"      Costo Final SA: Min={format_cost(stats_sa_s2['min_cost'])}, Max={format_cost(stats_sa_s2['max_cost'])}, Prom={format_cost(stats_sa_s2['avg_cost'])}, StdDev={stats_sa_s2['stdev_cost']:.2f}")
            print(f"      Tiempo SA Ejec: Prom={format_time(stats_sa_s2['avg_time'])}, Total={format_time(stats_sa_s2['total_time'])}")
            if stats_sa_s2['best_result']:
                 initial_c = stats_sa_s2['best_result'].get('initial_cost', 'N/A')
                 impr = calculate_improvement(initial_c, stats_sa_s2['min_cost'])
                 print(f"      Mejor Run (Stoch Seed {stats_sa_s2['best_result'].get('initial_seed', 'N/A')}): Costo SA={format_cost(stats_sa_s2['min_cost'])}, Mejora vs Stoch Ini={impr}")
        else: print("      Costo/Tiempo SA: N/A (Ninguna ejecución válida)")


    print(f"######### Fin Resumen: {case_name} #########")


if __name__ == "__main__":
    # --- Rutas y Configuración ---
    case_dir = './Casos'
    results_dir = './results'
    csv_output_filename = os.path.join(results_dir, 'results_summary.csv')
    json_output_filename = os.path.join(results_dir, 'all_results_details.json')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    try:
        case_files = sorted([f for f in os.listdir(case_dir) if f.startswith('case') and f.endswith('.txt')])
        if not case_files:
             print(f"ADVERTENCIA: No se encontraron archivos 'case*.txt' en el directorio '{case_dir}'")
             case_files = []
    except FileNotFoundError:
        print(f"ERROR: El directorio de casos '{case_dir}' no existe.")
        case_files = []
        import sys
        sys.exit(1) # Salir si no hay casos

    print(f"Archivos de caso encontrados para procesar: {case_files}\n")

    all_results_detailed = {} # Para guardar todos los detalles para el JSON final

    # --- Preparación CSV ---
    csv_header = [
        "Caso", "Algoritmo", "Pistas", "Parametros", "Punto Partida",
        "Estado", "Costo Final", "Tiempo_s", "Seed Inicial",
        "Costo Inicial", "Mejora_%"
    ]
    try:
        with open(csv_output_filename, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)

            # --- Bucle Principal por Caso ---
            for filename in case_files:
                filepath = os.path.join(case_dir, filename)
                print(f"\n=============== Procesando {filename} (Plan Exacto v2) ================")
                case_results = {} # Diccionario para los resultados de ESTE caso

                try:
                    D, planes, separations_matrix = read_case_data(filepath)
                    print(f"  Número de aviones: {D}")
                except Exception as e:
                    print(f"  ERROR leyendo el archivo del caso {filename}: {e}")
                    continue # Saltar al siguiente caso si no se puede leer

                # --- Función auxiliar para escribir en CSV ---
                def write_result_to_csv(algo_name, num_runways, params, start_point, result_dict):
                    # Extraer datos del diccionario de resultados
                    final_cost = result_dict.get('cost', INFINITO_COSTO)
                    exec_time = result_dict.get('time', 0)
                    # Datos de la solución inicial usada
                    initial_cost = result_dict.get('initial_cost')
                    initial_seed = result_dict.get('initial_seed')

                    # Determinar estado y formatear
                    status = "INVALID" if final_cost == INFINITO_COSTO or final_cost is None else "VALID"
                    final_cost_str = format_cost(final_cost)
                    improvement_str = calculate_improvement(initial_cost, final_cost)
                    seed_str = str(initial_seed) if initial_seed is not None else "N/A"
                    params_str = str(params) if params else "N/A"
                    start_point_str = str(start_point) if start_point else "N/A"
                    initial_cost_str = format_cost(initial_cost) if initial_cost is not None else "N/A"
                    time_str = format_time(exec_time)

                    row_data = [
                        filename, algo_name, num_runways, params_str, start_point_str,
                        status, final_cost_str, time_str, seed_str,
                        initial_cost_str,
                        improvement_str
                    ]
                    try:
                         csv_writer.writerow(row_data)
                    except Exception as csv_e:
                         print(f"  ERROR escribiendo fila en CSV: {csv_e} - Datos: {row_data}")
                    return status == "VALID"

                # --- Bucle por Número de Pistas (1 o 2) ---
                for num_runways in [1, 2]:
                    print(f"\n  --- {num_runways} Pista(s) ---")
                    runway_tag = f"{num_runways}_runway" if num_runways == 1 else f"{num_runways}_runways"
                    results_key_suffix = f"{num_runways}r" # Sufijo para claves de diccionario

                    # --- 1. Greedy Determinista (1 run) ---
                    print(f"    Ejecutando Greedy Determinista {results_key_suffix}...")
                    start_time = time.time()
                    schedule_d, times_d, cost_d = solve_greedy_deterministic(D, planes, separations_matrix, num_runways)
                    time_d = time.time() - start_time
                    # Para Greedy, el costo inicial es él mismo (o INF si falló), y no hay mejora
                    res_d = {'cost': cost_d, 'time': time_d, 'schedule': schedule_d, 'landing_times': times_d, 'initial_cost': cost_d, 'initial_seed': None}
                    case_results[f'deterministic_{runway_tag}'] = res_d
                    is_valid_d = write_result_to_csv("Greedy Deterministic", num_runways, None, "N/A", res_d)
                    det_sol = res_d if is_valid_d else None # Guardar si es válido para usar después

                    # --- 2. Greedy Estocástico (10 runs) ---
                    print(f"    Ejecutando Greedy Estocástico {results_key_suffix} ({NUM_STOCHASTIC_RUNS} runs)...")
                    stochastic_runs = []
                    for seed in range(NUM_STOCHASTIC_RUNS):
                        start_time = time.time()
                        schedule_s, times_s, cost_s = solve_greedy_stochastic(D, planes, separations_matrix, num_runways, seed, RCL_SIZE)
                        time_s = time.time() - start_time
                        # Para Greedy, el costo inicial es él mismo (o INF si falló)
                        res_s = {'seed': seed, 'cost': cost_s, 'time': time_s, 'schedule': schedule_s, 'landing_times': times_s, 'initial_cost': cost_s, 'initial_seed': seed}
                        stochastic_runs.append(res_s)
                        write_result_to_csv("Greedy Stochastic", num_runways, f"RCL={RCL_SIZE}", "N/A", res_s)
                    case_results[f'stochastic_{runway_tag}_runs'] = stochastic_runs
                    # Filtrar solo las corridas estocásticas válidas para usar como punto de partida
                    valid_stochastic_runs = [run for run in stochastic_runs if run.get('cost') is not None and run['cost'] != INFINITO_COSTO]


                    # --- 3. HC desde Determinista (1 run) ---
                    print(f"    Ejecutando HC desde Determinista {results_key_suffix}...")
                    hc_cost_d, hc_time_d = INFINITO_COSTO, 0
                    hc_schedule_d, hc_times_d = [], {}
                    initial_cost_hc_d = cost_d # Costo inicial es el del determinista
                    res_hc_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': initial_cost_hc_d, 'initial_seed': None} # Default
                    if det_sol: # Solo ejecutar si la solución determinista fue válida
                        start_time = time.time()
                        hc_schedule_d, hc_times_d, hc_cost_d = hill_climbing_first_improvement(
                            det_sol['schedule'], det_sol['landing_times'], det_sol['cost'],
                            D, planes, separations_matrix, num_runways, HC_MAX_ITER
                        )
                        hc_time_d = time.time() - start_time
                        res_hc_d = {'cost': hc_cost_d, 'time': hc_time_d, 'schedule': hc_schedule_d, 'landing_times': hc_times_d, 'initial_cost': initial_cost_hc_d, 'initial_seed': None}

                    case_results[f'hc_from_deterministic_{runway_tag}'] = res_hc_d
                    write_result_to_csv("HC", num_runways, f"HC_iter={HC_MAX_ITER}", "Deterministic", res_hc_d)


                    # --- 4. HC desde CADA Estocástica (10 runs * 3 "configuraciones" = 30 runs) ---
                    print(f"    Ejecutando HC desde CADA Estocástica {results_key_suffix}...")
                    # Las 3 "configuraciones" (10, 25, 50) son solo etiquetas según tu plan, HC_MAX_ITER es el parámetro real usado.
                    # Ejecutaremos HC_MAX_ITER para cada solución estocástica válida una vez por cada etiqueta.
                    for hc_iter_label in HC_FROM_STOCH_LABELS:
                         hc_from_stoch_runs = []
                         print(f"      Config HC (Label '{hc_iter_label}', Iter={HC_MAX_ITER}) para {len(valid_stochastic_runs)} soluciones Stoch válidas...")
                         params_hc = f"HC_iter={HC_MAX_ITER} (Label: {hc_iter_label})" # Parámetro para CSV
                         for stoch_run in valid_stochastic_runs:
                             initial_seed = stoch_run.get('seed')
                             initial_cost_s = stoch_run.get('cost')
                             start_point_str = f"Stochastic (Seed {initial_seed})"
                             start_time = time.time()
                             hc_schedule_s, hc_times_s, hc_cost_s = hill_climbing_first_improvement(
                                 stoch_run['schedule'], stoch_run['landing_times'], initial_cost_s,
                                 D, planes, separations_matrix, num_runways, HC_MAX_ITER
                             )
                             hc_time_s = time.time() - start_time
                             res_hc_s = {'cost': hc_cost_s, 'time': hc_time_s, 'schedule': hc_schedule_s, 'landing_times': hc_times_s, 'initial_cost': initial_cost_s, 'initial_seed': initial_seed}
                             hc_from_stoch_runs.append(res_hc_s)
                             write_result_to_csv("HC", num_runways, params_hc, start_point_str, res_hc_s)
                         # Guardar la lista de resultados de HC para esta "configuración" y pista
                         case_results[f'hc_from_stochastic_{hc_iter_label}_{results_key_suffix}_runs'] = hc_from_stoch_runs


                    # --- 5. SA desde Determinista (5 runs) ---
                    print(f"    Ejecutando SA desde Determinista {results_key_suffix}...")
                    initial_cost_sa_d = cost_d # Costo inicial es el del determinista
                    for T_init in SA_INITIAL_TEMPS:
                        sa_cost_d, sa_time_d = INFINITO_COSTO, 0
                        sa_schedule_d, sa_times_d = [], {}
                        res_sa_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': initial_cost_sa_d, 'initial_seed': None} # Default
                        params_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                        if det_sol: # Solo ejecutar si la solución determinista fue válida
                            start_time = time.time()
                            sa_schedule_d, sa_times_d, sa_cost_d = solve_simulated_annealing(
                                D, planes, separations_matrix, num_runways, det_sol,
                                T_init, SA_T_MIN, SA_ALPHA, SA_ITER_PER_TEMP, SA_MAX_NEIGHBOR_ATTEMPTS
                            )
                            sa_time_d = time.time() - start_time
                            res_sa_d = {'cost': sa_cost_d, 'time': sa_time_d, 'schedule': sa_schedule_d, 'landing_times': sa_times_d, 'initial_cost': initial_cost_sa_d, 'initial_seed': None}

                        case_results[f'sa_T{T_init}_from_det_{results_key_suffix}'] = res_sa_d # Clave ajustada
                        write_result_to_csv("SA", num_runways, params_sa, "Deterministic", res_sa_d)

                    # --- 6. SA desde CADA Estocástica (5 Temps * 10 runs = 50 runs) ---
                    print(f"    Ejecutando SA desde CADA Estocástica {results_key_suffix}...")
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
                            res_sa_s = {'cost': sa_cost_s, 'time': sa_time_s, 'schedule': sa_schedule_s, 'landing_times': sa_times_s, 'initial_cost': initial_cost_s, 'initial_seed': initial_seed}
                            sa_from_stoch_runs.append(res_sa_s)
                            write_result_to_csv("SA", num_runways, params_sa, start_point_str, res_sa_s)
                        # Guardar la lista de resultados de SA para esta Temp y pista
                        case_results[f'sa_T{T_init}_from_stochastic_{results_key_suffix}_runs'] = sa_from_stoch_runs


                # --- Fin del Bucle por Pistas ---

                # --- Fin de Procesamiento del Caso ---
                all_results_detailed[filename] = copy.deepcopy(case_results) # Guardar copia profunda para JSON
                print(f"--------------- Fin Procesamiento {filename} ---------------")

                # --- Imprimir Resumen Detallado para este Caso ---
                print_case_summary(filename, case_results)

            # --- Fin del Bucle por Casos ---

    except Exception as e:
        print(f"\nERROR FATAL durante la ejecución principal o escritura de CSV: {e}")
        import traceback
        traceback.print_exc()

    # --- Guardar Resultados Detallados en JSON ---
    print("\nGuardando resultados detallados en JSON...")
    try:
        # Serializador para manejar tipos no estándar (como sets, si los hubiera)
        def default_serializer(obj):
            if isinstance(obj, set): return list(obj)
            # Convertir cualquier otro objeto no serializable a string
            try:
                # Intenta convertir a float si parece número, sino string
                return float(obj)
            except (ValueError, TypeError):
                 try:
                      json.dumps(obj) # Intenta serializar directamente
                      return obj
                 except TypeError:
                      return str(obj) # Fallback a string


        with open(json_output_filename, 'w') as jsonfile:
            # Usar copy.deepcopy para asegurar que no haya referencias circulares o tipos complejos inesperados
            json.dump(copy.deepcopy(all_results_detailed), jsonfile, indent=2, default=default_serializer)
        print(f"--- Resultados detallados guardados en '{json_output_filename}' ---")
    except Exception as e:
        print(f"--- ERROR al guardar resultados detallados en JSON: {e}")
        import traceback
        traceback.print_exc()


    # --- Resumen Global Rápido (Opcional) ---
    # Este resumen ya no refleja bien las 97*2 corridas, es más útil el resumen por caso.
    print("\n\n################ Resumen Global Rápido (Aproximado) ################")
    total_cases = len(all_results_detailed)
    print(f"Casos procesados: {total_cases}")
    # Contar corridas válidas/inválidas podría hacerse recorriendo all_results_detailed si es necesario
    print("#############################################################")

    print("\n--- Ejecución Finalizada ---")