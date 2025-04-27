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
import traceback # Importar traceback para errores

# Asegúrate de que estas importaciones funcionen
try:
    from scripts.read_case_data import read_case_data
    from scripts.greedy_deterministic import solve_greedy_deterministic
    # Usaremos la versión con Selección Ponderada (última proporcionada)
    from scripts.greedy_stochastic import solve_greedy_stochastic
    from scripts.hill_climbing import hill_climbing_first_improvement
    from scripts.simulated_annealing import solve_simulated_annealing
except ImportError as e:
    print(f"ERROR: No se pudo importar un módulo necesario desde 'scripts'.")
    print(f"Detalle: {e}")
    print("Asegúrate que los archivos .py estén en la carpeta 'scripts' o ajusta las rutas.")
    # Intentar importar localmente como fallback (puede no funcionar si no están ahí)
    try:
        from read_case_data import read_case_data
        from greedy_deterministic import solve_greedy_deterministic
        from greedy_stochastic import solve_greedy_stochastic
        from hill_climbing import hill_climbing_first_improvement
        from simulated_annealing import solve_simulated_annealing
        print("ADVERTENCIA: Usando importaciones locales como fallback.")
    except ImportError:
         print("ERROR FATAL: Falló también la importación local. Saliendo.")
         sys.exit(1)


# --- Constantes y Parámetros ---
INFINITO_COSTO = float('inf')
NUM_STOCHASTIC_RUNS = 10
# Ya no necesitamos HC_FROM_STOCH_LABELS si no ejecutamos GRASP explícitamente aquí
# Si necesitaras ejecutar GRASP, deberías importar `solve_grasp` y llamarlo
# HC_FROM_STOCH_LABELS = [10, 25, 50] # Comentado - No se usa en esta versión de main
RCL_SIZE = 3 # Para Greedy Estocástico
HC_MAX_ITER = 500 # Iteraciones MÁXIMAS para CADA llamada a HC
SA_INITIAL_TEMPS = [10000, 5000, 1000, 500, 100] # Temperaturas iniciales para SA
SA_T_MIN = 0.1
SA_ALPHA = 0.95
SA_ITER_PER_TEMP = 100 # Iteraciones por nivel de temperatura en SA
SA_MAX_NEIGHBOR_ATTEMPTS = 50 # Intentos para generar vecino factible en SA


# --- Funciones Auxiliares de Formato y Cálculo ---
def format_cost(cost):
    """Formatea el costo para impresión, manejando INF y errores."""
    if cost == INFINITO_COSTO or cost is None:
        return "INF"
    try:
        # Intenta convertir a float y formatear
        return f"{float(cost):.2f}"
    except (ValueError, TypeError):
        # Si falla, retorna un indicador de error
        return "ERR_FMT"

def format_time(exec_time):
    """Formatea el tiempo para impresión, manejando None y errores."""
    if exec_time is None:
        return "N/A"
    try:
        # Intenta convertir a float y formatear
        return f"{float(exec_time):.4f}s"
    except (ValueError, TypeError):
        return "ERR_FMT"

def calculate_improvement(initial_cost, final_cost):
    """Calcula la mejora porcentual, manejando INF, None y división por cero."""
    # Casos donde no se puede calcular mejora
    if initial_cost in [INFINITO_COSTO, None] or final_cost in [INFINITO_COSTO, None]:
        return "N/A"
    if initial_cost == final_cost: # Caso sin cambio
        return "0.00%"
    try:
        init_c = float(initial_cost)
        final_c = float(final_cost)
        # Evitar división por cero si el costo inicial era 0
        if init_c == 0:
             return "-INF%" if final_c > 0 else "0.00%" # Si el final es mayor, mejora infinita negativa

        improvement = ((init_c - final_c) / init_c) * 100
        return f"{improvement:.2f}%"
    except (ValueError, TypeError):
        return "ERR_CALC" # Error en conversión a float

def get_stats_from_list(results_list):
    """Calcula estadísticas descriptivas para una lista de resultados (diccionarios)."""
    # Inicializar diccionario de estadísticas
    stats = {
        'costs': [r.get('cost', INFINITO_COSTO) for r in results_list],
        'times': [r.get('time', 0.0) for r in results_list], # Usar 0.0 como default para tiempo
        'valid_runs': 0,
        'invalid_runs': 0,
        'min_cost': INFINITO_COSTO,
        'max_cost': INFINITO_COSTO, # Empezar con infinito para max también es incorrecto
        'avg_cost': INFINITO_COSTO,
        'stdev_cost': 0.0,
        'avg_time': 0.0,
        'total_time': 0.0,
        'best_result': None # Guardará la copia del diccionario del mejor resultado
    }
    # Filtrar costos válidos (ni None ni infinito)
    valid_costs = [c for c in stats['costs'] if c is not None and c != INFINITO_COSTO]
    stats['valid_runs'] = len(valid_costs)
    stats['invalid_runs'] = len(results_list) - stats['valid_runs']

    # Calcular estadísticas de costo solo si hay corridas válidas
    if stats['valid_runs'] > 0:
        stats['min_cost'] = min(valid_costs)
        stats['max_cost'] = max(valid_costs) # Ahora esto es correcto
        stats['avg_cost'] = statistics.mean(valid_costs)
        # Calcular desviación estándar solo si hay más de una corrida válida
        if stats['valid_runs'] > 1:
            try:
                stats['stdev_cost'] = statistics.stdev(valid_costs)
            except statistics.StatisticsError: # Puede ocurrir si todos los valores son iguales
                stats['stdev_cost'] = 0.0
        else: # Si solo hay una corrida válida, la desviación es 0
             stats['stdev_cost'] = 0.0

        # Encontrar el índice y guardar el MEJOR resultado completo (menor costo)
        best_res_index = -1
        current_min = INFINITO_COSTO
        for i, res in enumerate(results_list):
            cost = res.get('cost', INFINITO_COSTO)
            # Asegurarse que el costo sea un número válido y menor al mínimo actual
            if cost is not None and cost != INFINITO_COSTO:
                if cost < current_min:
                    current_min = cost
                    best_res_index = i
                # Manejar caso inicial donde current_min es infinito
                elif cost == current_min and best_res_index == -1:
                     best_res_index = i


        if best_res_index != -1:
             # Guardar una copia profunda para evitar modificaciones accidentales
            stats['best_result'] = copy.deepcopy(results_list[best_res_index])

    # Calcular estadísticas de tiempo (basadas en todas las corridas intentadas)
    valid_times = [t for t in stats['times'] if t is not None]
    if valid_times:
        stats['avg_time'] = statistics.mean(valid_times)
        stats['total_time'] = sum(valid_times)

    return stats


# --- Función de Resumen por Caso (MODIFICADA para usar stats precalculadas) ---
def print_case_summary(case_name, results):
    """Imprime un resumen detallado de los resultados para un caso."""
    print(f"\n######### Resumen Detallado: {case_name} #########")

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

    # --- 2. Greedy Estocástico (Usa stats precalculadas) ---
    print(f"\n--- 2. Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs) ---")
    # Recuperar las estadísticas calculadas y guardadas previamente
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
    else: print("    Costo/Tiempo: N/A (Ninguna ejecución válida)")

    print(f"  2 Pistas: Válidas={stats_s2.get('valid_runs', 0)}/{total_runs_s2}")
    if stats_s2.get('valid_runs', 0) > 0:
        print(f"    Costo: Min={format_cost(stats_s2.get('min_cost'))}, Max={format_cost(stats_s2.get('max_cost'))}, Prom={format_cost(stats_s2.get('avg_cost'))}, StdDev={stats_s2.get('stdev_cost', 0.0):.2f}")
        print(f"    Tiempo Ejec: Prom={format_time(stats_s2.get('avg_time'))}, Total={format_time(stats_s2.get('total_time'))}")
        best_res_s2 = stats_s2.get('best_result')
        if best_res_s2: print(f"    Mejor Run (Seed {best_res_s2.get('seed', 'N/A')}): Costo={format_cost(best_res_s2.get('cost'))}")
    else: print("    Costo/Tiempo: N/A (Ninguna ejecución válida)")

    # --- 3. HC desde Determinista ---
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

    # --- 4. HC desde CADA Estocástica ---
    print(f"\n--- 4. Hill Climbing desde CADA Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs base) ---")
    # Recuperar las listas de resultados de HC para cada "configuración" (aunque aquí solo hay una implícita)
    # Usaremos un label fijo 'Std' ya que no hay diferentes configuraciones de HC aquí
    hc_runs_s1 = results.get(f'hc_from_stochastic_Std_1r_runs', [])
    hc_runs_s2 = results.get(f'hc_from_stochastic_Std_2r_runs', [])
    stats_hc_s1 = get_stats_from_list(hc_runs_s1)
    stats_hc_s2 = get_stats_from_list(hc_runs_s2)

    print(f"  Configuración HC (MaxIter={HC_MAX_ITER}):") # Label simplificado

    print(f"    1 Pista : Válidas={stats_hc_s1['valid_runs']}/{len(hc_runs_s1)}")
    if stats_hc_s1['valid_runs'] > 0:
        print(f"      Costo Final HC: Min={format_cost(stats_hc_s1['min_cost'])}, Max={format_cost(stats_hc_s1['max_cost'])}, Prom={format_cost(stats_hc_s1['avg_cost'])}, StdDev={stats_hc_s1['stdev_cost']:.2f}")
        print(f"      Tiempo HC Ejec: Prom={format_time(stats_hc_s1['avg_time'])}, Total={format_time(stats_hc_s1['total_time'])}")
        best_res_hc_s1 = stats_hc_s1.get('best_result')
        if best_res_hc_s1:
            initial_c = best_res_hc_s1.get('initial_cost', 'N/A')
            impr = calculate_improvement(initial_c, best_res_hc_s1.get('cost'))
            print(f"      Mejor Run (Stoch Seed {best_res_hc_s1.get('initial_seed', 'N/A')}): Costo HC={format_cost(best_res_hc_s1.get('cost'))}, Mejora vs Stoch Ini={impr}")
    else: print("      Costo/Tiempo HC: N/A (Ninguna ejecución válida)")

    print(f"    2 Pistas: Válidas={stats_hc_s2['valid_runs']}/{len(hc_runs_s2)}")
    if stats_hc_s2['valid_runs'] > 0:
        print(f"      Costo Final HC: Min={format_cost(stats_hc_s2['min_cost'])}, Max={format_cost(stats_hc_s2['max_cost'])}, Prom={format_cost(stats_hc_s2['avg_cost'])}, StdDev={stats_hc_s2['stdev_cost']:.2f}")
        print(f"      Tiempo HC Ejec: Prom={format_time(stats_hc_s2['avg_time'])}, Total={format_time(stats_hc_s2['total_time'])}")
        best_res_hc_s2 = stats_hc_s2.get('best_result')
        if best_res_hc_s2:
            initial_c = best_res_hc_s2.get('initial_cost', 'N/A')
            impr = calculate_improvement(initial_c, best_res_hc_s2.get('cost'))
            print(f"      Mejor Run (Stoch Seed {best_res_hc_s2.get('initial_seed', 'N/A')}): Costo HC={format_cost(best_res_hc_s2.get('cost'))}, Mejora vs Stoch Ini={impr}")
    else: print("      Costo/Tiempo HC: N/A (Ninguna ejecución válida)")


    # --- 5. SA desde Determinista ---
    print(f"\n--- 5. Simulated Annealing desde Greedy Determinista ---")
    for T_init in SA_INITIAL_TEMPS:
        res_sa_d1 = results.get(f'sa_T{T_init}_from_det_1r', {}) # Corrección: Clave ajustada
        res_sa_d2 = results.get(f'sa_T{T_init}_from_det_2r', {}) # Corrección: Clave ajustada
        cost_sa_d1 = res_sa_d1.get('cost')
        time_sa_d1 = res_sa_d1.get('time')
        cost_sa_d2 = res_sa_d2.get('cost')
        time_sa_d2 = res_sa_d2.get('time')
        impr_sa_d1 = calculate_improvement(cost_d1, cost_sa_d1)
        impr_sa_d2 = calculate_improvement(cost_d2, cost_sa_d2)
        print(f"  T_Inicial = {T_init}:")
        print(f"    1 Pista : Costo={format_cost(cost_sa_d1)}, Tiempo={format_time(time_sa_d1)}, Mejora vs Det={impr_sa_d1}")
        print(f"    2 Pistas: Costo={format_cost(cost_sa_d2)}, Tiempo={format_time(time_sa_d2)}, Mejora vs Det={impr_sa_d2}")

    # --- 6. SA desde CADA Estocástica ---
    print(f"\n--- 6. Simulated Annealing desde CADA Greedy Estocástico ({NUM_STOCHASTIC_RUNS} runs base por Temp) ---")
    for T_init in SA_INITIAL_TEMPS:
        # Recuperar lista de resultados de SA para esta Temp
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
        else: print("      Costo/Tiempo SA: N/A (Ninguna ejecución válida)")

        print(f"    2 Pistas: Válidas={stats_sa_s2['valid_runs']}/{len(sa_runs_s2)}")
        if stats_sa_s2['valid_runs'] > 0:
            print(f"      Costo Final SA: Min={format_cost(stats_sa_s2['min_cost'])}, Max={format_cost(stats_sa_s2['max_cost'])}, Prom={format_cost(stats_sa_s2['avg_cost'])}, StdDev={stats_sa_s2['stdev_cost']:.2f}")
            print(f"      Tiempo SA Ejec: Prom={format_time(stats_sa_s2['avg_time'])}, Total={format_time(stats_sa_s2['total_time'])}")
            best_res_sa_s2 = stats_sa_s2.get('best_result')
            if best_res_sa_s2:
                initial_c = best_res_sa_s2.get('initial_cost', 'N/A')
                impr = calculate_improvement(initial_c, best_res_sa_s2.get('cost'))
                print(f"      Mejor Run (Stoch Seed {best_res_sa_s2.get('initial_seed', 'N/A')}): Costo SA={format_cost(best_res_sa_s2.get('cost'))}, Mejora vs Stoch Ini={impr}")
        else: print("      Costo/Tiempo SA: N/A (Ninguna ejecución válida)")

    print(f"######### Fin Resumen: {case_name} #########")


# --- Punto de Entrada Principal ---
if __name__ == "__main__":
    # --- Configuración Directorios ---
    case_dir = './Casos'
    results_dir = './results'
    # Usar nombres de archivo consistentes para evitar sobrescrituras accidentales
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_output_filename = os.path.join(results_dir, f'results_summary_{timestamp}.csv')
    json_output_filename = os.path.join(results_dir, f'all_results_details_{timestamp}.json')

    # Crear directorio de resultados si no existe
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # --- Encontrar Archivos de Caso ---
    try:
        case_files = sorted([f for f in os.listdir(case_dir) if f.startswith('case') and f.endswith('.txt')])
        if not case_files:
            print(f"ADVERTENCIA: No se encontraron archivos 'case*.txt' en el directorio '{case_dir}'")
            # Si no hay casos, no tiene sentido continuar
            sys.exit(0)
    except FileNotFoundError:
        print(f"ERROR: El directorio de casos '{case_dir}' no existe.")
        sys.exit(1) # Salir si no se pueden encontrar los casos

    print(f"Archivos de caso encontrados para procesar: {case_files}\n")

    # Diccionario principal para guardar TODOS los detalles
    all_results_detailed = {}

    # --- Preparación CSV ---
    csv_header = [
        "Caso", "Algoritmo", "Pistas", "Parametros", "Punto Partida",
        "Estado", "Costo Final", "Tiempo_s", "Seed Inicial",
        "Costo Inicial", "Mejora_%"
    ]
    try:
        # Abrir archivo CSV para escribir
        with open(csv_output_filename, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(csv_header)

            # --- Bucle Principal por Caso ---
            for filename in case_files:
                filepath = os.path.join(case_dir, filename)
                print(f"\n=============== Procesando {filename} ================ ")
                case_results = {} # Diccionario para resultados de ESTE caso
                try:
                    # Leer datos del caso
                    D, planes, separations_matrix = read_case_data(filepath)
                    print(f"  Número de aviones: {D}")
                except Exception as e:
                    print(f"  ERROR leyendo el archivo del caso {filename}: {e}")
                    traceback.print_exc() # Imprimir stack trace para debug
                    continue # Saltar al siguiente caso

                # --- Función interna para escribir fila en CSV ---
                def write_result_to_csv(algo_name, num_runways, params, start_point, result_dict):
                    # Extraer datos del diccionario de resultado, con valores default
                    final_cost = result_dict.get('cost', INFINITO_COSTO)
                    exec_time = result_dict.get('time', 0.0)
                    initial_cost = result_dict.get('initial_cost') # Puede ser None
                    initial_seed = result_dict.get('initial_seed') # Puede ser None

                    # Determinar estado y formatear valores
                    status = "INVALID" if final_cost == INFINITO_COSTO or final_cost is None else "VALID"
                    final_cost_str = format_cost(final_cost)
                    improvement_str = calculate_improvement(initial_cost, final_cost)
                    seed_str = str(initial_seed) if initial_seed is not None else "N/A"
                    params_str = str(params) if params is not None else "N/A"
                    start_point_str = str(start_point) if start_point is not None else "N/A"
                    initial_cost_str = format_cost(initial_cost) if initial_cost is not None else "N/A"
                    time_str = format_time(exec_time)

                    # Crear fila de datos
                    row_data = [
                        filename, algo_name, num_runways, params_str, start_point_str,
                        status, final_cost_str, time_str, seed_str,
                        initial_cost_str,
                        improvement_str
                    ]
                    # Escribir en CSV con manejo de errores
                    try:
                        csv_writer.writerow(row_data)
                    except Exception as csv_e:
                        print(f"  ERROR escribiendo fila en CSV: {csv_e} - Datos: {row_data}")
                    return status == "VALID" # Retornar si la corrida fue válida


                # --- Bucle por Número de Pistas (1 o 2) ---
                for num_runways in [1, 2]:
                    print(f"\n  --- {num_runways} Pista(s) ---")
                    # Crear tags para nombres de claves en el diccionario de resultados
                    runway_tag = f"{num_runways}_runway" if num_runways == 1 else f"{num_runways}_runways"
                    results_key_suffix = f"{num_runways}r" # Sufijo como '1r' o '2r'

                    # --- 1. Greedy Determinista ---
                    print(f"    Ejecutando Greedy Determinista {results_key_suffix}...")
                    start_time = time.time()
                    schedule_d, times_d, cost_d = solve_greedy_deterministic(D, planes, separations_matrix, num_runways)
                    time_d = time.time() - start_time
                    # Guardar resultado determinista (incluye costo inicial para referencia)
                    res_d = {'cost': cost_d, 'time': time_d, 'schedule': schedule_d, 'landing_times': times_d, 'initial_cost': cost_d, 'initial_seed': None}
                    case_results[f'deterministic_{runway_tag}'] = copy.deepcopy(res_d) # Guardar copia
                    is_valid_d = write_result_to_csv("Greedy Deterministic", num_runways, None, "N/A", res_d)
                    det_sol = res_d if is_valid_d else None # Guardar solo si es válido para usar luego

                    # --- 2. Greedy Estocástico ---
                    print(f"    Ejecutando Greedy Estocástico {results_key_suffix} ({NUM_STOCHASTIC_RUNS} runs)...")
                    stochastic_runs = [] # Lista para guardar resultados de las 10 corridas
                    for seed in range(NUM_STOCHASTIC_RUNS):
                        start_time = time.time()
                        schedule_s, times_s, cost_s = solve_greedy_stochastic(D, planes, separations_matrix, num_runways, seed, RCL_SIZE)
                        time_s = time.time() - start_time
                        # Guardar detalles de cada corrida estocástica
                        res_s = {'seed': seed, 'cost': cost_s, 'time': time_s, 'schedule': schedule_s, 'landing_times': times_s, 'initial_cost': cost_s, 'initial_seed': seed}
                        stochastic_runs.append(res_s)
                        # Escribir cada corrida estocástica individual al CSV
                        write_result_to_csv("Greedy Stochastic", num_runways, f"RCL={RCL_SIZE}", "N/A", res_s)

                    # **CORRECCIÓN: Calcular y guardar stats INMEDIATAMENTE**
                    stats_s = get_stats_from_list(stochastic_runs)
                    case_results[f'stochastic_{runway_tag}_runs'] = copy.deepcopy(stochastic_runs) # Guardar lista cruda
                    case_results[f'stochastic_{runway_tag}_stats'] = copy.deepcopy(stats_s) # Guardar stats calculadas

                    # Obtener lista de corridas VÁLIDAS para usar en HC/SA
                    valid_stochastic_runs = [run for run in stochastic_runs if run.get('cost') is not None and run['cost'] != INFINITO_COSTO]


                    # --- 3. HC desde Determinista ---
                    print(f"    Ejecutando HC desde Determinista {results_key_suffix}...")
                    initial_cost_hc_d = cost_d # Costo inicial es el del determinista
                    # Valores por defecto si falla o no se ejecuta
                    res_hc_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': initial_cost_hc_d, 'initial_seed': None}
                    if det_sol: # Solo ejecutar si la solución determinista fue válida
                        start_time = time.time()
                        hc_schedule_d, hc_times_d, hc_cost_d = hill_climbing_first_improvement(
                            det_sol['schedule'], det_sol['landing_times'], det_sol['cost'],
                            D, planes, separations_matrix, num_runways, HC_MAX_ITER
                        )
                        hc_time_d = time.time() - start_time
                        res_hc_d = {'cost': hc_cost_d, 'time': hc_time_d, 'schedule': hc_schedule_d, 'landing_times': hc_times_d, 'initial_cost': initial_cost_hc_d, 'initial_seed': None}

                    case_results[f'hc_from_deterministic_{runway_tag}'] = copy.deepcopy(res_hc_d)
                    write_result_to_csv("HC", num_runways, f"HC_iter={HC_MAX_ITER}", "Deterministic", res_hc_d)

                    # --- 4. HC desde CADA Estocástica Válida ---
                    print(f"    Ejecutando HC desde CADA Estocástica {results_key_suffix}...")
                    # Usaremos un label fijo ya que no hay diferentes configs de HC
                    hc_label = 'Std'
                    hc_from_stoch_runs = []
                    print(f"      Config HC (MaxIter={HC_MAX_ITER}) para {len(valid_stochastic_runs)} soluciones Stoch válidas...")
                    params_hc = f"HC_iter={HC_MAX_ITER}" # Parámetro para CSV

                    # Iterar SOLO sobre las corridas estocásticas válidas
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
                        # Guardar resultado de HC, incluyendo info inicial
                        res_hc_s = {'cost': hc_cost_s, 'time': hc_time_s, 'schedule': hc_schedule_s, 'landing_times': hc_times_s, 'initial_cost': initial_cost_s, 'initial_seed': initial_seed}
                        hc_from_stoch_runs.append(res_hc_s)
                        # Escribir resultado de esta ejecución de HC al CSV
                        write_result_to_csv("HC", num_runways, params_hc, start_point_str, res_hc_s)

                    # Guardar la lista completa de resultados de HC desde Estocásticos
                    case_results[f'hc_from_stochastic_{hc_label}_{results_key_suffix}_runs'] = copy.deepcopy(hc_from_stoch_runs)


                    # --- 5. SA desde Determinista ---
                    print(f"    Ejecutando SA desde Determinista {results_key_suffix}...")
                    initial_cost_sa_d = cost_d # Costo inicial es el del determinista
                    for T_init in SA_INITIAL_TEMPS:
                        # Valores por defecto si falla o no se ejecuta
                        res_sa_d = {'cost': INFINITO_COSTO, 'time': 0, 'schedule': [], 'landing_times': {}, 'initial_cost': initial_cost_sa_d, 'initial_seed': None}
                        params_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                        if det_sol: # Solo ejecutar si la solución determinista fue válida
                            start_time = time.time()
                            sa_schedule_d, sa_times_d, sa_cost_d = solve_simulated_annealing(
                                D, planes, separations_matrix, num_runways, det_sol, # Pasa el dict completo
                                T_init, SA_T_MIN, SA_ALPHA, SA_ITER_PER_TEMP, SA_MAX_NEIGHBOR_ATTEMPTS
                            )
                            sa_time_d = time.time() - start_time
                            res_sa_d = {'cost': sa_cost_d, 'time': sa_time_d, 'schedule': sa_schedule_d, 'landing_times': sa_times_d, 'initial_cost': initial_cost_sa_d, 'initial_seed': None}

                        # Guardar resultado para esta Temp y escribir a CSV
                        case_results[f'sa_T{T_init}_from_det_{results_key_suffix}'] = copy.deepcopy(res_sa_d)
                        write_result_to_csv("SA", num_runways, params_sa, "Deterministic", res_sa_d)


                    # --- 6. SA desde CADA Estocástica Válida ---
                    print(f"    Ejecutando SA desde CADA Estocástica {results_key_suffix}...")
                    for T_init in SA_INITIAL_TEMPS:
                        sa_from_stoch_runs = [] # Lista para esta Temp
                        params_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                        print(f"      T_Inicial = {T_init} para {len(valid_stochastic_runs)} soluciones Stoch válidas...")

                        # Iterar SOLO sobre las corridas estocásticas válidas
                        for stoch_run in valid_stochastic_runs:
                            initial_seed = stoch_run.get('seed')
                            initial_cost_s = stoch_run.get('cost')
                            start_point_str = f"Stochastic (Seed {initial_seed})"
                            start_time = time.time()
                            sa_schedule_s, sa_times_s, sa_cost_s = solve_simulated_annealing(
                                D, planes, separations_matrix, num_runways, stoch_run, # Pasa el dict completo
                                T_init, SA_T_MIN, SA_ALPHA, SA_ITER_PER_TEMP, SA_MAX_NEIGHBOR_ATTEMPTS
                            )
                            sa_time_s = time.time() - start_time
                            # Guardar resultado de SA, incluyendo info inicial
                            res_sa_s = {'cost': sa_cost_s, 'time': sa_time_s, 'schedule': sa_schedule_s, 'landing_times': sa_times_s, 'initial_cost': initial_cost_s, 'initial_seed': initial_seed}
                            sa_from_stoch_runs.append(res_sa_s)
                            # Escribir resultado de esta ejecución de SA al CSV
                            write_result_to_csv("SA", num_runways, params_sa, start_point_str, res_sa_s)

                        # Guardar la lista completa de resultados de SA (para esta Temp) desde Estocásticos
                        case_results[f'sa_T{T_init}_from_stochastic_{results_key_suffix}_runs'] = copy.deepcopy(sa_from_stoch_runs)

                # --- Fin del Bucle por Pistas ---

                # --- Fin de Procesamiento del Caso ---
                all_results_detailed[filename] = copy.deepcopy(case_results) # Guardar copia profunda
                print(f"--------------- Fin Procesamiento {filename} ---------------")

                # --- Imprimir Resumen Detallado para este Caso ---
                print_case_summary(filename, case_results)

            # --- Fin del Bucle por Casos ---

    # Capturar excepción general en la escritura del CSV
    except IOError as e:
         print(f"\nERROR FATAL: No se pudo abrir o escribir el archivo CSV '{csv_output_filename}'. Verifica permisos o ruta.")
         print(f"Detalle: {e}")
         sys.exit(1)
    except Exception as e:
        print(f"\nERROR FATAL durante la ejecución principal o escritura de CSV: {e}")
        traceback.print_exc() # Imprimir stack trace completo para diagnóstico
        sys.exit(1)


    # --- Guardar Resultados Detallados en JSON ---
    print("\nGuardando resultados detallados en JSON...")
    try:
        # Función auxiliar para manejar tipos no serializables por JSON (como sets)
        def default_serializer(obj):
            if isinstance(obj, set):
                return list(obj) # Convertir sets a listas
            # Intentar convertir a float si es posible (para números como infinito)
            # Aunque ya usamos None/INF, esto podría ayudar con otros tipos numéricos
            # try:
            #     return float(obj)
            # except (ValueError, TypeError):
                 # Si no es float, intentar serializar como está
            try:
                json.dumps(obj) # Verificar si es serializable directamente
                return obj
            except TypeError:
                 # Si todo falla, convertir a string
                return str(obj)

        # Usar copy.deepcopy para asegurar que no haya referencias cruzadas inesperadas
        with open(json_output_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(copy.deepcopy(all_results_detailed), jsonfile, indent=2, default=default_serializer)
        print(f"--- Resultados detallados guardados en '{json_output_filename}' ---")
    except Exception as e:
        print(f"--- ERROR al guardar resultados detallados en JSON: {e}")
        traceback.print_exc() # Imprimir stack trace completo para diagnóstico

    # --- Resumen Final Rápido (Opcional) ---
    print("\n\n#############################################################")
    total_cases = len(all_results_detailed)
    print(f"Casos procesados: {total_cases}")
    # Podrías añadir un resumen global de validez aquí si lo necesitas,
    # calculándolo a partir de all_results_detailed
    print("#############################################################")

    print("\n--- Ejecución Finalizada ---")