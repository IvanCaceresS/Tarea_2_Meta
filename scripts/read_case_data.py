def read_case_data(filepath):
    planes_data = []
    all_separations_flat = []
    with open(filepath, 'r') as f:
        D = int(f.readline().strip())
        for _ in range(D):
            parts = f.readline().split()
            planes_data.append([int(parts[0]), int(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])])
            current_plane_separations = []
            while len(current_plane_separations) < D:
                current_plane_separations.extend(map(int, f.readline().split()))
            all_separations_flat.extend(current_plane_separations)
    separations = [[0] * D for _ in range(D)]
    k = 0
    for i in range(D):
        for j in range(D):
            separations[i][j] = all_separations_flat[k]
            k += 1
    return D, planes_data, separations
