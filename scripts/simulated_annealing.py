# simulated_annealing.py
import random
import math
import copy
import time

# Importar funciones necesarias (ajusta la ruta si es necesario)
try:
    # Si hill_climbing.py está en el mismo directorio o en PYTHONPATH
    from scripts.hill_climbing import recalculate_landing_times
except ImportError:
    print("ADVERTENCIA: No se pudo importar 'recalculate_landing_times' desde 'hill_climbing'.")
    print("Asegúrate que 'hill_climbing.py' esté accesible.")
    # Podrías copiar la función aquí como fallback si la importación falla
    # O manejar el error de forma más robusta
    def recalculate_landing_times(*args, **kwargs): # Placeholder
        print("ERROR: recalculate_landing_times no disponible!")
        return None, float('inf'), False

from scripts.calculate_cost import calculate_total_cost


# --- Constantes ---
INFINITO_COSTO = float('inf')

def solve_simulated_annealing(D, planes_data, separations, num_runways,
                              initial_solution_dict,
                              T_initial, T_min, alpha, iter_per_temp,
                              max_neighbor_attempts=100): # Límite para encontrar vecino factible
    """
    Implementa Simulated Annealing (SA).

    Args:
        D (int): Número de aviones.
        planes_data (list): Lista de [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        initial_solution_dict (dict): Solución inicial {'schedule': [], 'landing_times': {}, 'cost': float}.
        T_initial (float): Temperatura inicial.
        T_min (float): Temperatura mínima (criterio de parada).
        alpha (float): Factor de enfriamiento (e.g., 0.95).
        iter_per_temp (int): Iteraciones (generación de vecinos) por nivel de temperatura.
        max_neighbor_attempts (int): Intentos máximos para generar un vecino factible.


    Returns:
        tuple: (best_schedule, best_times, best_cost)
               La mejor solución encontrada durante el proceso SA.
               Retorna ([], {}, float('inf')) si la entrada es inválida o falla.
    """
    # --- Validación y Setup Inicial ---
    current_schedule = initial_solution_dict.get('schedule')
    current_times = initial_solution_dict.get('landing_times')
    current_cost = initial_solution_dict.get('cost', INFINITO_COSTO)

    if current_cost == INFINITO_COSTO or not current_schedule or D < 2:
        # No se puede ejecutar SA sin una solución inicial válida o si hay menos de 2 aviones
        return initial_solution_dict.get('schedule', []), initial_solution_dict.get('landing_times', {}), INFINITO_COSTO

    # Empezar con copias profundas para no modificar la original
    best_schedule = copy.deepcopy(current_schedule)
    best_times = copy.deepcopy(current_times)
    best_cost = current_cost

    current_T = T_initial
    # print(f"      SA Iniciando: T_init={T_initial}, T_min={T_min}, alpha={alpha}, iter={iter_per_temp}, Costo Ini={current_cost:.2f}")

    # --- Bucle Principal SA ---
    while current_T > T_min:
        for _ in range(iter_per_temp):
            # --- Generar Vecino Factible (2-opt swap) ---
            neighbor_schedule = None
            neighbor_times = None
            neighbor_cost = INFINITO_COSTO
            found_feasible_neighbor = False

            # Intentar generar un vecino factible hasta 'max_neighbor_attempts' veces
            for attempt in range(max_neighbor_attempts):
                # Elegir dos índices distintos al azar
                if D <= 1: break # No se puede hacer swap
                idx1, idx2 = random.sample(range(D), 2)

                # Crear copia y hacer el swap
                temp_schedule = list(current_schedule)
                temp_schedule[idx1], temp_schedule[idx2] = temp_schedule[idx2], temp_schedule[idx1]

                # Recalcular y verificar factibilidad
                temp_times, temp_cost, is_feasible = recalculate_landing_times(
                    temp_schedule, D, planes_data, separations, num_runways
                )

                if is_feasible:
                    neighbor_schedule = temp_schedule
                    neighbor_times = temp_times
                    neighbor_cost = temp_cost
                    found_feasible_neighbor = True
                    break # Salir del bucle de intentos

            # Si no se encontró vecino factible después de varios intentos, continuar
            if not found_feasible_neighbor:
                # print(f"        SA Advertencia: No se encontró vecino factible en {max_neighbor_attempts} intentos a T={current_T:.2f}")
                continue # Pasar a la siguiente iteración a esta temperatura

            # --- Decisión de Aceptación ---
            delta_cost = neighbor_cost - current_cost

            if delta_cost < 0: # Mejor solución encontrada
                accept = True
            else:
                # Aceptar peor solución con probabilidad
                probability = math.exp(-delta_cost / current_T) if current_T > 1e-9 else 0 # Evitar división por cero
                accept = random.random() < probability

            # --- Actualizar Solución Actual si se acepta ---
            if accept:
                current_schedule = neighbor_schedule
                current_times = neighbor_times
                current_cost = neighbor_cost

                # --- Actualizar Mejor Solución Global ---
                if current_cost < best_cost:
                    best_schedule = copy.deepcopy(current_schedule)
                    best_times = copy.deepcopy(current_times)
                    best_cost = current_cost
                    # print(f"        SA Nueva Mejor: {best_cost:.2f} a T={current_T:.2f}") # Debug opcional

        # --- Enfriamiento ---
        current_T *= alpha

    # print(f"      SA Finalizado. Mejor costo: {best_cost:.2f}")
    return best_schedule, best_times, best_cost