# calculate_cost.py (Original - Sin Penalización L_k)

INFINITO_COSTO = float('inf') # Added for consistency if needed

def calculate_total_cost(planes_data, schedule_plane_ids, landing_times):
    """
    Calcula el costo total de una programación, basado en desviaciones
    respecto al tiempo preferente P. Asume que la entrada es factible
    respecto a E, L y separaciones (aunque no lo verifica aquí).

    Args:
        planes_data (list): Lista de [E, P, L, Ce, Cl] para cada avión.
        schedule_plane_ids (list): Lista ordenada de IDs de aviones.
        landing_times (dict): Diccionario {plane_id: tiempo_aterrizaje}.

    Returns:
        float: Costo total calculado (solo penalizaciones P).
               Retorna 0.0 si la secuencia está vacía.
               Retorna INFINITO_COSTO si hay un error grave (ID faltante).
    """
    total_cost = 0.0
    if not schedule_plane_ids:
        return 0.0

    # Check if landing_times covers all planes in schedule
    if not all(pid in landing_times for pid in schedule_plane_ids):
        print(f"ERROR CCOST (Orig): Faltan tiempos en landing_times para aviones en schedule.")
        # Find missing IDs for better debugging (optional)
        # missing_ids = [pid for pid in schedule_plane_ids if pid not in landing_times]
        # print(f"  Missing IDs: {missing_ids}")
        return INFINITO_COSTO

    for plane_id in schedule_plane_ids:
        # Validate plane_id index (redundant if schedule generation is correct, but safe)
        if not (0 <= plane_id < len(planes_data)):
            print(f"ERROR CCOST (Orig): ID de avión inválido {plane_id} fuera de rango [0, {len(planes_data)-1}]")
            return INFINITO_COSTO

        # Ensure landing_times[plane_id] exists and is valid
        if plane_id not in landing_times or landing_times[plane_id] is None:
            print(f"ERROR CCOST (Orig): Tiempo de aterrizaje faltante o inválido para avión {plane_id}.")
            return INFINITO_COSTO

        try:
            E, P, L, Ce, Cl = planes_data[plane_id]
            time_k = landing_times[plane_id] # Tiempo de aterrizaje asignado

            # Penalización por desviación respecto a P
            early_penalty = Ce * max(0, P - time_k)
            late_penalty = Cl * max(0, time_k - P)
            total_cost += (early_penalty + late_penalty)
        except IndexError:
             print(f"ERROR CCOST (Orig): IndexError al acceder a planes_data para ID {plane_id}.")
             return INFINITO_COSTO
        except TypeError:
             print(f"ERROR CCOST (Orig): TypeError al calcular costo para ID {plane_id}, time_k={time_k}.")
             return INFINITO_COSTO


    return total_cost
