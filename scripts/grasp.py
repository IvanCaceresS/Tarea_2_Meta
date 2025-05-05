# grasp.py (Modificado para usar Reparación)
import random
import time
import copy

# Importaciones necesarias
try:
    from scripts.greedy_stochastic import solve_greedy_stochastic # La versión que puede violar Lk
    from scripts.hill_climbing import hill_climbing_best_improvement, recalculate_landing_times
    # Importar la función de reparación
    from scripts.repair_schedule import repair_schedule_simple_swap
    # Importar la función de costo ORIGINAL
    from scripts.calculate_cost import calculate_total_cost
except ImportError as e:
    print(f"ERROR GRASP: No se pudieron importar módulos necesarios: {e}")
    # Define Dummies para evitar crash
    def solve_greedy_stochastic(*args, **kwargs): return [], {}, float('inf')
    def hill_climbing_best_improvement(*args, **kwargs): return [], {}, float('inf')
    def repair_schedule_simple_swap(*args, **kwargs): return [], {}, float('inf'), False
    def calculate_total_cost(*args, **kwargs): return float('inf')
    def recalculate_landing_times(*args, **kwargs): return None, float('inf'), False


# --- Constantes ---
INFINITO_COSTO = float('inf')

def solve_grasp(D, planes_data, separations, num_runways,
                grasp_iterations, rcl_size, hc_max_iter, max_repair_attempts=None): # Added repair attempts param
    """
    Implementa GRASP con una fase de reparación después de la construcción.

    Args:
        D (int): Número de aviones.
        planes_data (list): Lista de [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        grasp_iterations (int): Número de veces que se repite Construcción + Reparación + Búsqueda Local.
        rcl_size (int): Tamaño de la RCL para la fase de construcción estocástica.
        hc_max_iter (int): Límite de iteraciones para Hill Climbing.
        max_repair_attempts (int, optional): Intentos máximos para la función de reparación. Defaults to 2*D.

    Returns:
        tuple: (best_schedule, best_times, best_cost, result_info)
               - best_schedule, best_times, best_cost: La mejor solución FACTIBLE encontrada.
               - result_info (dict): {'successful_repairs': int, 'failed_repairs': int, 'valid_hc_starts': int}
               Retorna ([], {}, float('inf'), info) si ninguna iteración produce una solución factible.
    """
    best_overall_cost = INFINITO_COSTO
    best_overall_schedule = []
    best_overall_times = {}
    start_time_grasp = time.time() # Tiempo total del GRASP

    # print(f"      Iniciando GRASP ({grasp_iterations} iteraciones, con Reparación)...") # Info movida a main.py
    successful_repairs = 0
    failed_repairs = 0
    total_valid_hc_starts = 0 # Contar cuántas veces HC inició con algo factible

    for i in range(grasp_iterations):
        current_seed = i # Usar iteración como semilla para reproducibilidad

        # --- Fase 1: Construcción con Greedy Estocástico (Puede ser infactible Lk) ---
        construction_schedule, construction_times, construction_cost_ignored = \
            solve_greedy_stochastic(D, planes_data, separations, num_runways, current_seed, rcl_size)

        # Verificar si la construcción produjo algo (aunque sea infactible)
        if not construction_schedule:
            # print(f"        Iter {i+1} (Seed {current_seed}): Construcción falló completamente. Saltando.") # Debug
            failed_repairs += 1 # Contar como fallo si la construcción no produjo nada
            continue

        # --- Fase 1.5: Reparación ---
        # print(f"        Iter {i+1} (Seed {current_seed}): Intentando reparar...") # Debug
        repaired_schedule, repaired_times, repaired_cost, repair_success = repair_schedule_simple_swap(
            construction_schedule, construction_times, # Pasar tiempos aunque no se usen directamente
            D, planes_data, separations, num_runways, max_repair_attempts
        )

        if not repair_success:
            # print(f"        Iter {i+1} (Seed {current_seed}): Reparación falló. Saltando búsqueda local.") # Debug
            failed_repairs += 1
            continue # Saltar al siguiente ciclo GRASP

        # Si llegamos aquí, la reparación fue exitosa
        successful_repairs += 1
        total_valid_hc_starts += 1
        # repaired_cost es el costo factible (calculado con costo original)
        # Usamos la solución reparada como punto de partida para HC
        # print(f"        Iter {i+1} (Seed {current_seed}): Reparación exitosa. Costo post-reparación={repaired_cost:.2f}. Iniciando HC...") # Debug
        initial_hc_schedule = repaired_schedule
        initial_hc_times = repaired_times
        initial_hc_cost = repaired_cost # Costo real factible

        # --- Fase 2: Búsqueda Local (Hill Climbing) ---
        # Partir de la solución REPARADA y FACTIBLE
        hc_schedule, hc_times, hc_cost = hill_climbing_best_improvement(
            initial_hc_schedule, initial_hc_times, initial_hc_cost, # Usar datos reparados
            D, planes_data, separations, num_runways, hc_max_iter
        )

        # print(f"        Iter {i+1} (Seed {current_seed}): HC Result Costo={hc_cost:.2f}") # Debug

        # --- Actualizar la Mejor Solución Global ---
        # Comparar el costo DESPUÉS de la búsqueda local
        # Asegurarse que hc_cost no sea infinito (por si HC fallara internamente, aunque no debería)
        if hc_cost is not None and hc_cost != INFINITO_COSTO and hc_cost < best_overall_cost:
             # print(f"        Iter {i+1}: ¡Nueva mejor solución encontrada! {hc_cost:.2f} (< {best_overall_cost:.2f})") # Debug
             best_overall_cost = hc_cost
             best_overall_schedule = hc_schedule # Guardar la solución mejorada por HC
             best_overall_times = hc_times

    total_grasp_time = time.time() - start_time_grasp
    # Mover el resumen de reparación a main.py para que se imprima una sola vez por ejecución de GRASP
    # print(f"      GRASP finalizado. Reparaciones Exitosas={successful_repairs}, Fallidas={failed_repairs}.")
    # print(f"      Mejor costo FACTIBLE encontrado: {format_cost(best_overall_cost)} (Tiempo total GRASP: {total_grasp_time:.4f}s)")

    # Devolver el número de reparaciones exitosas/fallidas para informar en main.py
    result_info = {'successful_repairs': successful_repairs, 'failed_repairs': failed_repairs, 'valid_hc_starts': total_valid_hc_starts}

    # Asegurarse de retornar inf si nunca se encontró una solución válida
    if best_overall_cost == INFINITO_COSTO:
        return [], {}, INFINITO_COSTO, result_info

    return best_overall_schedule, best_overall_times, best_overall_cost, result_info


# --- Función auxiliar para ejecutar HC desde determinista (sin cambios) ---
# (Se mantiene igual, no necesita reparación)
def solve_hc_from_deterministic(D, planes_data, separations, num_runways,
                                 initial_solution_dict, hc_max_iter):
    """
    Aplica Hill Climbing partiendo de UNA solución determinista inicial.
    """
    initial_schedule = initial_solution_dict.get('schedule', [])
    initial_times = initial_solution_dict.get('landing_times', {})
    initial_cost = initial_solution_dict.get('cost', INFINITO_COSTO)

    if initial_cost == INFINITO_COSTO or not initial_schedule:
        return initial_schedule, initial_times, INFINITO_COSTO

    hc_schedule, hc_times, hc_cost = hill_climbing_best_improvement(
        initial_schedule, initial_times, initial_cost,
        D, planes_data, separations, num_runways, hc_max_iter
    )
    return hc_schedule, hc_times, hc_cost

