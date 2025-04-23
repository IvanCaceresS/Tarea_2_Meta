# main.py
import os
import math
import time
import statistics
import csv
import json

# Asegúrate de que estas importaciones funcionen según tu estructura de archivos
from scripts.read_case_data import read_case_data
from scripts.greedy_deterministic import solve_greedy_deterministic
from scripts.greedy_stochastic import solve_greedy_stochastic
from scripts.calculate_cost import calculate_total_cost
from scripts.hill_climbing import hill_climbing_first_improvement
from scripts.grasp import solve_grasp # Asume que solve_grasp es para el estocástico

# --- Funciones Auxiliares de Impresión ---
def print_solution_details(label, result_dict):
    """Imprime detalles de una solución (costo, tiempo, validez)."""
    cost = result_dict.get('cost', float('inf'))
    exec_time = result_dict.get('time', 0)
    if cost == float('inf'): status = "INVÁLIDO"
    else: status = "VÁLIDO"
    cost_str = "INF" if cost == float('inf') else f"{cost:.2f}"
    # Ajustar el espacio para etiquetas más largas
    print(f"    {label:<55}: {status} - Costo: {cost_str} (Tiempo: {exec_time:.4f}s)")
    return cost != float('inf') # Retorna True si es válida

def print_stochastic_run_details(label, run_list):
    """Imprime detalles y calcula stats de una lista de ejecuciones estocásticas."""
    costs = [r['cost'] for r in run_list if r.get('cost', float('inf')) != float('inf')]
    total_runs = len(run_list)
    valid_runs = len(costs)
    invalid_runs = total_runs - valid_runs
    # Ajustar el espacio
    print(f"  {label:<55} ({valid_runs} Válidas, {invalid_runs} Inválidas de {total_runs}):")
    stats_dict = {
        'valid_count': valid_runs, 'invalid_count': invalid_runs,
        'min': float('inf'), 'max': float('inf'), 'avg': float('inf'), 'stdev': 0.0
    }
    if costs:
        stats_dict['min'] = min(costs)
        stats_dict['max'] = max(costs)
        stats_dict['avg'] = statistics.mean(costs)
        stats_dict['stdev'] = statistics.stdev(costs) if valid_runs > 1 else 0.0
        print(f"      Stats (Válidas): Min={stats_dict['min']:.2f}, Max={stats_dict['max']:.2f}, Prom={stats_dict['avg']:.2f}, StdDev={stats_dict['stdev']:.2f}")
    else:
        print(f"      Stats (Válidas): N/A")
    return stats_dict # Retornar diccionario con estadísticas

if __name__ == "__main__":
    case_dir = './Casos'
    csv_output_filename = './results/results_summary.csv' # Incrementar versión
    json_output_filename = './results/all_results_details.json'
    case_files = sorted([f for f in os.listdir(case_dir) if f.startswith('case') and f.endswith('.txt')])
    print(f"Archivos de caso encontrados para procesar: {case_files}\n")

    all_results = {}
    # --- Parámetros Configurables ---
    NUM_STOCHASTIC_RUNS = 10
    GRASP_ITERATIONS_LIST = [5, 15, 30]
    RCL_SIZE = 3
    HC_MAX_ITER = 500
    INFINITO_COSTO = float('inf')

    # --- Abrir archivo CSV y escribir cabecera ---
    with open(csv_output_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        header = [
            "Caso", "Algoritmo", "Pistas", "Seed_Iteraciones",
            "Estado", "Costo", "Tiempo_s"
        ]
        csv_writer.writerow(header)

        # --- Contadores Globales ---
        global_stats = {
            'det_valid': 0, 'det_invalid': 0, 'stoch_valid': 0, 'stoch_invalid': 0,
            # Renombrar contadores para HC desde Det
            'hc_det_valid': 0, 'hc_det_failed_or_not_run': 0,
            'grasp_stoch_valid': 0, 'grasp_stoch_invalid': 0,
        }

        # --- Bucle Principal por Archivo ---
        for filename in case_files:
            filepath = os.path.join(case_dir, filename)
            print(f"\n=============== Procesando {filename} ================")
            all_results[filename] = {}
            D, planes, separations_matrix = read_case_data(filepath)
            print(f"  Número de aviones: {D}")

            # --- Función auxiliar para escribir en CSV ---
            def write_result_to_csv(algo_name, num_runways, seed_or_iter, result_dict):
                cost = result_dict.get('cost', INFINITO_COSTO)
                exec_time = result_dict.get('time', 0)
                status = "INVALID" if cost == INFINITO_COSTO else "VALID"
                cost_str = "INF" if cost == INFINITO_COSTO else f"{cost:.2f}"
                row_data = [filename, algo_name, num_runways, seed_or_iter, status, cost_str, f"{exec_time:.4f}"]
                csv_writer.writerow(row_data)
                return status == "VALID"

            # --- 1. Greedy Determinista ---
            print("\n--- 1. Greedy Determinista ---")
            start_time_d1 = time.time(); schedule_d1, times_d1, cost_d1 = solve_greedy_deterministic(D, planes, separations_matrix, 1); time_d1 = time.time() - start_time_d1
            res_d1 = {'cost': cost_d1, 'time': time_d1, 'schedule': schedule_d1, 'landing_times': times_d1}; all_results[filename]['deterministic_1_runway'] = res_d1
            is_valid_d1 = print_solution_details("Determinista 1 Pista", res_d1); write_result_to_csv("Greedy Deterministic", 1, "N/A", res_d1)
            if is_valid_d1: 
                global_stats['det_valid'] += 1; det_sol_1r = res_d1
            else: 
                global_stats['det_invalid'] += 1; det_sol_1r = None

            start_time_d2 = time.time(); schedule_d2, times_d2, cost_d2 = solve_greedy_deterministic(D, planes, separations_matrix, 2); time_d2 = time.time() - start_time_d2
            res_d2 = {'cost': cost_d2, 'time': time_d2, 'schedule': schedule_d2, 'landing_times': times_d2}; all_results[filename]['deterministic_2_runways'] = res_d2
            is_valid_d2 = print_solution_details("Determinista 2 Pistas", res_d2); write_result_to_csv("Greedy Deterministic", 2, "N/A", res_d2)
            if is_valid_d2: 
                global_stats['det_valid'] += 1; det_sol_2r = res_d2
            else: 
                global_stats['det_invalid'] += 1; det_sol_2r = None

            # --- 1.1 Hill Climbing desde Determinista ---
            #    (Corresponde al Item 2: GRASP Determinista)
            print("\n--- 1.1 HC desde Greedy Determinista ---")
            # 1 Pista
            hc_cost_d1, hc_time_d1 = INFINITO_COSTO, 0; hc_sched_d1_final, hc_times_d1_final = [], {}
            if det_sol_1r:
                start_time_hc_d1 = time.time(); hc_schedule_d1, hc_times_d1, hc_cost_d1 = hill_climbing_first_improvement(det_sol_1r['schedule'], det_sol_1r['landing_times'], det_sol_1r['cost'], D, planes, separations_matrix, 1, HC_MAX_ITER); hc_time_d1 = time.time() - start_time_hc_d1; hc_sched_d1_final, hc_times_d1_final = hc_schedule_d1, hc_times_d1
            # *** CORREGIR CLAVE DE ALMACENAMIENTO ***
            res_hc_d1 = {'cost': hc_cost_d1, 'time': hc_time_d1, 'schedule': hc_sched_d1_final, 'landing_times': hc_times_d1_final}
            all_results[filename]['hc_from_deterministic_1_runway'] = res_hc_d1
            is_valid = print_solution_details("HC desde Det. 1 Pista", res_hc_d1); write_result_to_csv("HC from Deterministic", 1, "N/A", res_hc_d1)
            if is_valid: 
                global_stats['hc_det_valid'] += 1
            else: 
                global_stats['hc_det_failed_or_not_run'] += 1

            # 2 Pistas
            hc_cost_d2, hc_time_d2 = INFINITO_COSTO, 0; hc_sched_d2_final, hc_times_d2_final = [], {}
            if det_sol_2r:
                start_time_hc_d2 = time.time(); hc_schedule_d2, hc_times_d2, hc_cost_d2 = hill_climbing_first_improvement(det_sol_2r['schedule'], det_sol_2r['landing_times'], det_sol_2r['cost'], D, planes, separations_matrix, 2, HC_MAX_ITER); hc_time_d2 = time.time() - start_time_hc_d2; hc_sched_d2_final, hc_times_d2_final = hc_schedule_d2, hc_times_d2
            # *** CORREGIR CLAVE DE ALMACENAMIENTO ***
            res_hc_d2 = {'cost': hc_cost_d2, 'time': hc_time_d2, 'schedule': hc_sched_d2_final, 'landing_times': hc_times_d2_final}
            all_results[filename]['hc_from_deterministic_2_runways'] = res_hc_d2
            is_valid = print_solution_details("HC desde Det. 2 Pistas", res_hc_d2); write_result_to_csv("HC from Deterministic", 2, "N/A", res_hc_d2)
            if is_valid: 
                global_stats['hc_det_valid'] += 1
            else: 
                global_stats['hc_det_failed_or_not_run'] += 1

            # --- 1.2 Greedy Estocástico (Item 1) ---
            print(f"\n--- 1.2 Greedy Estocástico (SOLO Construcción, {NUM_STOCHASTIC_RUNS} runs) ---")
            # ... (código estocástico sin cambios) ...
            stochastic_runs_1r = []; stochastic_runs_2r = []
            case_stoch_valid_1r, case_stoch_invalid_1r = 0, 0; case_stoch_valid_2r, case_stoch_invalid_2r = 0, 0
            # Primer ciclo para 1 pista
            for seed in range(NUM_STOCHASTIC_RUNS):
                schedule_s1, times_s1, cost_s1 = solve_greedy_stochastic(D, planes, separations_matrix, 1, seed, RCL_SIZE)
                res_s1 = {
                    'seed': seed,
                    'cost': cost_s1,
                    'schedule': schedule_s1,
                    'landing_times': times_s1
                }
                stochastic_runs_1r.append(res_s1)
                is_valid = write_result_to_csv("Greedy Stochastic", 1, seed, res_s1)
                if is_valid:
                    case_stoch_valid_1r += 1
                else:
                    case_stoch_invalid_1r += 1

            all_results[filename]['stochastic_1_runway_runs'] = stochastic_runs_1r
            stats_s1 = print_stochastic_run_details(f"Estocástico 1 Pista ({NUM_STOCHASTIC_RUNS} runs)", stochastic_runs_1r)
            all_results[filename]['stochastic_1_runway_stats'] = stats_s1
            global_stats['stoch_valid'] += stats_s1['valid_count']
            global_stats['stoch_invalid'] += stats_s1['invalid_count']

            # Segundo ciclo para 2 pistas
            for seed in range(NUM_STOCHASTIC_RUNS):
                schedule_s2, times_s2, cost_s2 = solve_greedy_stochastic(D, planes, separations_matrix, 2, seed, RCL_SIZE)
                res_s2 = {
                    'seed': seed,
                    'cost': cost_s2,
                    'schedule': schedule_s2,
                    'landing_times': times_s2
                }
                stochastic_runs_2r.append(res_s2)
                is_valid = write_result_to_csv("Greedy Stochastic", 2, seed, res_s2)
                if is_valid:
                    case_stoch_valid_2r += 1
                else:
                    case_stoch_invalid_2r += 1

            all_results[filename]['stochastic_2_runway_runs'] = stochastic_runs_2r
            stats_s2 = print_stochastic_run_details(f"Estocástico 2 Pistas ({NUM_STOCHASTIC_RUNS} runs)", stochastic_runs_2r)
            all_results[filename]['stochastic_2_runway_stats'] = stats_s2
            global_stats['stoch_valid'] += stats_s2['valid_count']
            global_stats['stoch_invalid'] += stats_s2['invalid_count']


            # --- 2.1 GRASP + HC Estocástico (Item 2.1) ---
            print(f"\n--- 2.1 GRASP + HC Estocástico (Probando con {GRASP_ITERATIONS_LIST} restarts) ---")
            for grasp_iters in GRASP_ITERATIONS_LIST:
                algo_tag_base = f'grasp_stochastic_{grasp_iters}iters'
                # 1 Pista GRASP
                print(f"  Resolviendo GRASP Estoc. 1 Pista ({grasp_iters} restarts)...")
                start_time_g1 = time.time(); grasp_schedule_1, grasp_times_1, grasp_cost_1 = solve_grasp(D, planes, separations_matrix, 1, grasp_iters, RCL_SIZE, HC_MAX_ITER); time_g1 = time.time() - start_time_g1
                # *** CORREGIR CLAVE DE ALMACENAMIENTO ***
                res_g1 = {'cost': grasp_cost_1, 'time': time_g1, 'schedule': grasp_schedule_1, 'landing_times': grasp_times_1}; all_results[filename][f'{algo_tag_base}_1_runway'] = res_g1
                is_valid = print_solution_details(f"GRASP Stoch. 1 Pista ({grasp_iters} restarts)", res_g1); write_result_to_csv("GRASP Stochastic", 1, grasp_iters, res_g1)
                if is_valid: 
                    global_stats['grasp_stoch_valid'] += 1
                else: 
                    global_stats['grasp_stoch_invalid'] += 1

                # 2 Pistas GRASP
                print(f"  Resolviendo GRASP Estoc. 2 Pistas ({grasp_iters} restarts)...")
                start_time_g2 = time.time(); grasp_schedule_2, grasp_times_2, grasp_cost_2 = solve_grasp(D, planes, separations_matrix, 2, grasp_iters, RCL_SIZE, HC_MAX_ITER); time_g2 = time.time() - start_time_g2
                # *** CORREGIR CLAVE DE ALMACENAMIENTO ***
                res_g2 = {'cost': grasp_cost_2, 'time': time_g2, 'schedule': grasp_schedule_2, 'landing_times': grasp_times_2}; all_results[filename][f'{algo_tag_base}_2_runways'] = res_g2
                is_valid = print_solution_details(f"GRASP Stoch. 2 Pistas ({grasp_iters} restarts)", res_g2); write_result_to_csv("GRASP Stochastic", 2, grasp_iters, res_g2)
                if is_valid: 
                    global_stats['grasp_stoch_valid'] += 1
                else: 
                    global_stats['grasp_stoch_invalid'] += 1

            print(f"================ Fin {filename} ================")

    # --- Guardar Resultados Detallados en JSON ---
    try:
        with open(json_output_filename, 'w') as jsonfile:
            json.dump(all_results, jsonfile, indent=2)
        print(f"\n--- Resultados detallados guardados en '{json_output_filename}' ---")
    except Exception as e:
        print(f"\n--- ERROR al guardar resultados en JSON: {e}")


    # --- RESUMEN FINAL GLOBAL EN CONSOLA ---
    print("\n\n################ Resumen Final Global (Consola) ################")
    for filename, data in all_results.items():
         print(f"\n------ {filename} ------")
         print("  --- Greedy Determinista ---")
         print_solution_details("  * Determinista 1P", data.get('deterministic_1_runway', {}))
         print_solution_details("  * Determinista 2P", data.get('deterministic_2_runways', {}))
         print("  --- HC desde Determinista (GRASP Det.) ---")
         # *** USAR CLAVE CORREGIDA ***
         print_solution_details("  * HC/GRASP Det. 1P", data.get('hc_from_deterministic_1_runway', {}))
         print_solution_details("  * HC/GRASP Det. 2P", data.get('hc_from_deterministic_2_runways', {}))
         print("  --- Greedy Estocástico (Construcción) ---")
         print_stochastic_run_details(f"  * Estocástico 1P ({NUM_STOCHASTIC_RUNS} runs)", data.get('stochastic_1_runway_runs', []))
         print_stochastic_run_details(f"  * Estocástico 2P ({NUM_STOCHASTIC_RUNS} runs)", data.get('stochastic_2_runway_runs', []))
         print("  --- GRASP Estocástico (Construcción + HC) ---")
         for grasp_iters in GRASP_ITERATIONS_LIST:
             # *** USAR CLAVE CORREGIDA ***
             print_solution_details(f"  * GRASP Stoch. 1P ({grasp_iters} restarts)", data.get(f'grasp_stochastic_{grasp_iters}iters_1_runway', {}))
             print_solution_details(f"  * GRASP Stoch. 2P ({grasp_iters} restarts)", data.get(f'grasp_stochastic_{grasp_iters}iters_2_runways', {}))


    # Resumen Global de Validez
    print("\n------ Resumen Global de Validez ------")
    total_det_intentos = len(case_files) * 2
    total_stoch_intentos = len(case_files) * 2 * NUM_STOCHASTIC_RUNS
    # *** CORREGIR CÁLCULO DE INTENTOS HC DESDE DET ***
    total_hc_det_intentos = global_stats['det_valid'] # Solo se intenta si Det es válido
    total_grasp_stoch_intentos = len(case_files) * 2 * len(GRASP_ITERATIONS_LIST)

    print(f"Greedy Determinista:")
    print(f"  Válidas: {global_stats['det_valid']} / {total_det_intentos}, Inválidas: {global_stats['det_invalid']}")
    # *** USAR CONTADOR HC_DET CORREGIDO ***
    print(f"HC desde Determinista (GRASP Det):")
    print(f"  Válidas: {global_stats['hc_det_valid']} / {total_hc_det_intentos} (intentos), Fallos/No Ejec: {global_stats['hc_det_failed_or_not_run']}")
    print(f"Greedy Estocástico (Construcción):")
    print(f"  Válidas: {global_stats['stoch_valid']} / {total_stoch_intentos}, Inválidas: {global_stats['stoch_invalid']}")
    # *** USAR CONTADOR GRASP_STOCH CORREGIDO ***
    print(f"GRASP Estocástico (Resultado Final):")
    print(f"  Válidas: {global_stats['grasp_stoch_valid']} / {total_grasp_stoch_intentos}, Inválidas: {global_stats['grasp_stoch_invalid']}")


    print("\n--- Todos los archivos procesados ---")