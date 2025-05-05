# repair_schedule.py
import copy

# Necesitamos la función para recalcular tiempos y verificar factibilidad
# Y la función de costo ORIGINAL para evaluar soluciones reparadas
try:
    # Adjust path if your scripts are in a different structure
    from scripts.hill_climbing import recalculate_landing_times
    from scripts.calculate_cost import calculate_total_cost # El original
except ImportError:
    try:
        # Fallback to local import if not in 'scripts'
        from hill_climbing import recalculate_landing_times
        from calculate_cost import calculate_total_cost
        # print("Advertencia: repair_schedule usó importación local para hill_climbing/calculate_cost.")
    except ImportError:
        print("ERROR FATAL: repair_schedule no puede importar funciones necesarias de hill_climbing o calculate_cost.")
        # Define Dummies para evitar crash inmediato, pero no funcionará
        def recalculate_landing_times(*args, **kwargs): return None, float('inf'), False
        def calculate_total_cost(*args, **kwargs): return float('inf')

INFINITO_COSTO = float('inf')

def repair_schedule_simple_swap(schedule, landing_times_in, D, planes_data, separations, num_runways, max_repair_attempts=None):
    """
    Intenta reparar una solución COMPLETA que puede tener violaciones de L_k
    mediante intercambios locales simples (swap con predecesor).

    Args:
        schedule (list): Secuencia completa inicial (puede ser infactible por L_k).
        landing_times_in (dict): Tiempos iniciales (pueden violar L_k, pero no se usan directamente).
        D (int): Número de aviones.
        planes_data (list): Datos originales [E, P, L, Ce, Cl].
        separations (list): Matriz de separación DxD.
        num_runways (int): Número de pistas (1 o 2).
        max_repair_attempts (int, optional): Límite de iteraciones de reparación.
                                            Defaults to 2*D if None.

    Returns:
        tuple: (final_schedule, final_times, final_cost, repair_successful)
               - final_schedule, final_times, final_cost: La solución después de intentar reparar.
                 Si la reparación es exitosa, el costo es el factible (sin penalización L_k).
                 Si falla, devuelve la secuencia original y costo infinito.
               - repair_successful (bool): True si la solución final es factible, False si no.
    """
    if not schedule:
        return [], {}, 0.0, True # Vacío es factible

    if max_repair_attempts is None:
        max_repair_attempts = 5 * D # Default limit

    current_schedule = list(schedule) # Trabajar con una copia

    # Verificar si la solución de entrada ya es factible
    current_times, current_cost_check, is_feasible = recalculate_landing_times(
         current_schedule, D, planes_data, separations, num_runways
    )

    if is_feasible:
         # Si ya es factible, calcular el costo real y devolver
         final_cost = calculate_total_cost(planes_data, current_schedule, current_times)
         # print("  Repair: Solución inicial ya era factible.") # Debug
         return current_schedule, current_times, final_cost, True

    # print(f"  Repair: Iniciando reparación para solución infactible (Max {max_repair_attempts} intentos)...") # Debug

    for attempt in range(max_repair_attempts):
        # 1. Recalcular tiempos y factibilidad actual (necesario en cada intento)
        current_times, _, is_feasible = recalculate_landing_times(
            current_schedule, D, planes_data, separations, num_runways
        )

        # 2. Si se volvió factible, éxito!
        if is_feasible:
            final_cost = calculate_total_cost(planes_data, current_schedule, current_times) # Calcular costo real
            # print(f"  Repair: Solución factible encontrada en intento {attempt+1}. Costo={final_cost:.2f}") # Debug
            return current_schedule, current_times, final_cost, True

        # 3. Si sigue infactible, encontrar la PRIMERA violación L_k
        violating_idx = -1
        if current_times: # Asegurarse que recalculate no falló completamente
            for i, plane_id in enumerate(current_schedule):
                # Ensure plane_id is valid index for planes_data
                if not (0 <= plane_id < len(planes_data)):
                    print(f"ERROR Repair: ID inválido {plane_id} en schedule.")
                    continue # Skip this ID

                E, P, L, Ce, Cl = planes_data[plane_id]
                # Ensure plane_id is in current_times
                if plane_id not in current_times:
                    print(f"ERROR Repair: ID {plane_id} no encontrado en current_times.")
                    # This indicates a deeper issue, maybe return failure
                    return schedule, landing_times_in, INFINITO_COSTO, False

                # Check if landing time exceeds L
                if current_times[plane_id] > L:
                    violating_idx = i
                    # print(f"    Repair Attempt {attempt+1}: Violación Lk encontrada en índice {violating_idx} (Avión {plane_id}, T={current_times[plane_id]:.1f} > L={L})") # Debug
                    break # Intentar arreglar la primera que se encuentre

        # 4. Si no hay violaciones L_k o es el primer avión, no podemos usar este swap
        if violating_idx <= 0:
            # print(f"  Repair: No se encontró violación Lk reparable por swap en intento {attempt+1} (idx={violating_idx}). Deteniendo.") # Debug
            break # Salir del bucle de intentos, no podemos reparar con este método

        # 5. Intentar swap con el predecesor
        # print(f"    Repair Attempt {attempt+1}: Intentando swap entre índice {violating_idx} y {violating_idx - 1}...") # Debug
        neighbor_schedule = list(current_schedule)
        try:
            # Swap elements at violating_idx and violating_idx - 1
            neighbor_schedule[violating_idx], neighbor_schedule[violating_idx - 1] = \
                neighbor_schedule[violating_idx - 1], neighbor_schedule[violating_idx]
        except IndexError:
             print(f"ERROR Repair: Índice {violating_idx} o {violating_idx - 1} fuera de rango para swap.")
             break # Cannot perform swap

        # 6. Evaluar el vecino
        neighbor_times, _, neighbor_is_feasible = recalculate_landing_times(
            neighbor_schedule, D, planes_data, separations, num_runways
        )

        # 7. Decidir si mantener el swap: SOLO si el vecino se vuelve factible
        if neighbor_is_feasible:
            final_cost = calculate_total_cost(planes_data, neighbor_schedule, neighbor_times)
            # print(f"    Repair: Swap en {violating_idx} produjo solución factible! Costo={final_cost:.2f}") # Debug
            return neighbor_schedule, neighbor_times, final_cost, True
        else:
            # Si el swap no lo hizo factible, *mantenemos el swap* y continuamos
            # La esperanza es que mover el avión infractor antes pueda
            # dar más espacio a los siguientes y eventualmente resolverse.
            # print(f"    Repair: Swap en {violating_idx} NO produjo solución factible. Manteniendo swap y continuando...") # Debug
            current_schedule = neighbor_schedule # Keep the swapped schedule

    # Si salimos del bucle sin éxito, retornamos la secuencia original y marcamos fallo
    # print(f"  Repair: Falló después de {max_repair_attempts} intentos.") # Debug
    return schedule, landing_times_in, INFINITO_COSTO, False # Devolver original y costo infinito
