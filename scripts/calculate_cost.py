def calculate_total_cost(planes_data, schedule_plane_ids, landing_times):
    total_cost = 0.0
    if not schedule_plane_ids:
        return 0.0

    for plane_id in schedule_plane_ids:
        E, P, L, Ce, Cl = planes_data[plane_id]
        time_k = landing_times[plane_id] # Tiempo de aterrizaje asignado
        early_penalty = Ce * max(0, P - time_k) # Penalización por aterrizaje temprano
        late_penalty = Cl * max(0, time_k - P) # Penalización por aterrizaje tardío
        total_cost += (early_penalty + late_penalty) # Suma de penalizaciones
    return total_cost