# grasp.py
import random
import time
from scripts.greedy_stochastic import solve_greedy_stochastic
from scripts.hill_climbing import hill_climbing_first_improvement
from scripts.calculate_cost import calculate_total_cost # Necesario si recalculamos al final

# --- Constantes ---
INFINITO_COSTO = float('inf')

def solve_grasp(D, planes_data, separations, num_runways,
                grasp_iterations, rcl_size, hc_max_iter):
    """
    Implementa el algoritmo GRASP (Greedy Randomized Adaptive Search Procedure).

    Args:
        D (int): Número de aviones.
        planes_data (list): Lista de [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        grasp_iterations (int): Número de veces que se repite Construcción + Búsqueda Local.
        rcl_size (int): Tamaño de la RCL para la fase de construcción estocástica.
        hc_max_iter (int): Límite de iteraciones para Hill Climbing.

    Returns:
        tuple: (best_schedule, best_times, best_cost)
               La mejor solución encontrada a lo largo de todas las iteraciones GRASP.
               Retorna ([], {}, float('inf')) si ninguna iteración produce una solución válida.
    """
    best_overall_cost = INFINITO_COSTO
    best_overall_schedule = []
    best_overall_times = {}
    start_time_grasp = time.time() # Tiempo total del GRASP

    print(f"      Iniciando GRASP ({grasp_iterations} iteraciones)...")

    for i in range(grasp_iterations):
        # Usar el número de iteración como semilla asegura que cada construcción
        # sea diferente pero reproducible si se ejecuta GRASP de nuevo con los mismos parámetros.
        current_seed = i

        # --- Fase 1: Construcción con Greedy Estocástico ---
        construction_schedule, construction_times, construction_cost = \
            solve_greedy_stochastic(D, planes_data, separations, num_runways, current_seed, rcl_size)

        # Si la construcción falló, saltar a la siguiente iteración GRASP
        if construction_cost == INFINITO_COSTO:
            # print(f"        Iter {i+1}/{grasp_iterations} (Seed {current_seed}): Construcción falló.")
            continue # Saltar a la siguiente iteración

        # print(f"        Iter {i+1}/{grasp_iterations} (Seed {current_seed}): Construcción Costo={construction_cost:.2f}")

        # --- Fase 2: Búsqueda Local (Hill Climbing) ---
        # Partir de la solución construida
        hc_schedule, hc_times, hc_cost = hill_climbing_first_improvement(
            construction_schedule, construction_times, construction_cost,
            D, planes_data, separations, num_runways, hc_max_iter
        )

        # print(f"        Iter {i+1}/{grasp_iterations} (Seed {current_seed}): HC Result Costo={hc_cost:.2f}")

        # --- Actualizar la Mejor Solución Global ---
        # Comparar el costo DESPUÉS de la búsqueda local
        if hc_cost < best_overall_cost:
             # print(f"        Iter {i+1}/{grasp_iterations}: ¡Nueva mejor solución encontrada! {hc_cost:.2f} (< {best_overall_cost:.2f})")
             best_overall_cost = hc_cost
             best_overall_schedule = hc_schedule # Guardar la solución mejorada por HC
             best_overall_times = hc_times

    total_grasp_time = time.time() - start_time_grasp
    print(f"      GRASP finalizado. Mejor costo encontrado: {best_overall_cost:.2f} (Tiempo total GRASP: {total_grasp_time:.4f}s)")

    # Asegurarse de retornar inf si nunca se encontró una solución válida
    if best_overall_cost == INFINITO_COSTO:
        return [], {}, INFINITO_COSTO

    return best_overall_schedule, best_overall_times, best_overall_cost

def solve_hc_from_deterministic(D, planes_data, separations, num_runways,
                                 initial_solution_dict, hc_max_iter):
    """
    Aplica Hill Climbing partiendo de UNA solución determinista inicial.
    Esto simula un 'GRASP con 1 iteración y construcción determinista'.

    Args:
        D (int): Número de aviones.
        planes_data (list): Lista de [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        initial_solution_dict (dict): Diccionario con la solución determinista:
                                     {'schedule': list, 'landing_times': dict, 'cost': float}
        hc_max_iter (int): Límite de iteraciones para Hill Climbing.

    Returns:
        tuple: (best_schedule, best_times, best_cost)
               La solución después de aplicar HC.
               Retorna la entrada o fallo si la entrada es inválida.
    """
    # Extraer datos de la solución inicial
    initial_schedule = initial_solution_dict.get('schedule', [])
    initial_times = initial_solution_dict.get('landing_times', {})
    initial_cost = initial_solution_dict.get('cost', INFINITO_COSTO)

    # Si la solución inicial ya es inválida, no hacer nada
    if initial_cost == INFINITO_COSTO or not initial_schedule:
        # print("      HC desde Det: Solución inicial inválida, no se aplica HC.")
        return initial_schedule, initial_times, INFINITO_COSTO

    # Aplicar Hill Climbing
    # print("      Aplicando HC a solución determinista...")
    start_time_hc = time.time()
    hc_schedule, hc_times, hc_cost = hill_climbing_first_improvement(
        initial_schedule, initial_times, initial_cost,
        D, planes_data, separations, num_runways, hc_max_iter
    )
    hc_time = time.time() - start_time_hc

    # Retornar el resultado de HC
    # Devolver también el tiempo de ejecución de HC por si se quiere registrar
    # Nota: La función original de HC no devolvía el tiempo, se asume que se añade
    # o se mide externamente como se hace en main.py
    return hc_schedule, hc_times, hc_cost