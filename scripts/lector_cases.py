def read_case(nombre_archivo):
    """
    Lee un archivo de caso y devuelve los datos estructurados.

    Args:
        nombre_archivo (str): La ruta al archivo .txt del caso.

    Returns:
        dict: Un diccionario con los datos del caso, None si hay error.
              Ejemplo de estructura:
              {
                  'num_aviones': D,
                  'aviones': [
                      {'id': 0, 'E': E0, 'P': P0, 'L': L0, 'cost_temprano': C0, 'cost_tardio': C0_prime},
                      ...
                  ],
                  'tiempos_separacion': [[t00, t01, ...], [t10, t11, ...], ...]
              }
    """
    try:
        with open(nombre_archivo, 'r') as f:
            lineas = f.readlines()

        lineas = [linea.strip() for linea in lineas if linea.strip()] # Eliminar líneas vacías y espacios extra

        if not lineas:
            print(f"Error: El archivo {nombre_archivo} está vacío o no contiene datos.")
            return None

        cursor = 0 # Para llevar la cuenta de qué línea estamos leyendo

        # 1. Leer el número de aviones (D)
        num_aviones = int(lineas[cursor])
        cursor += 1

        aviones_data = []
        tiempos_separacion_matriz = []

        # 2. Para cada avión i (i=0,...,D-1)
        for i in range(num_aviones):
            # 2.a. Leer E_k, P_k, L_k, C_k, C'_k
            if cursor >= len(lineas):
                print(f"Error: Faltan datos para el avión {i} (parámetros de aterrizaje) en {nombre_archivo}.")
                return None
            
            partes_avion = lineas[cursor].split()
            if len(partes_avion) != 5:
                print(f"Error: Formato incorrecto para los parámetros del avión {i} en {nombre_archivo}. Se esperaban 5 valores.")
                return None
            
            E_k = int(partes_avion[0])
            P_k = int(partes_avion[1])
            L_k = int(partes_avion[2])
            cost_temprano_k = float(partes_avion[3])
            cost_tardio_k = float(partes_avion[4])
            
            aviones_data.append({
                'id': i,
                'E': E_k,
                'P': P_k,
                'L': L_k,
                'cost_temprano': cost_temprano_k,
                'cost_tardio': cost_tardio_k
            })
            cursor += 1

            # 2.b. Leer los tiempos de separación tau_ij para el avión i actual
            # Estos tiempos pueden estar en una o más líneas. Necesitamos leer num_aviones valores.
            tiempos_separacion_avion_i = []
            while len(tiempos_separacion_avion_i) < num_aviones:
                if cursor >= len(lineas):
                    print(f"Error: Faltan datos de tiempos de separación para el avión {i} en {nombre_archivo}.")
                    return None
                
                partes_separacion = lineas[cursor].split()
                for val_str in partes_separacion:
                    tiempos_separacion_avion_i.append(int(val_str))
                cursor += 1 # Avanzamos a la siguiente línea del archivo
            
            if len(tiempos_separacion_avion_i) != num_aviones:
                 print(f"Error: No se pudieron leer todos los {num_aviones} tiempos de separación para el avión {i} en {nombre_archivo}.")
                 return None
            tiempos_separacion_matriz.append(tiempos_separacion_avion_i)

        # Verificación final: ¿Se leyeron todos los datos esperados?
        if len(aviones_data) != num_aviones or len(tiempos_separacion_matriz) != num_aviones:
            print(f"Error: Inconsistencia en la cantidad de datos leídos para {nombre_archivo}.")
            return None

        return {
            'num_aviones': num_aviones,
            'aviones': aviones_data,
            'tiempos_separacion': tiempos_separacion_matriz
        }

    except FileNotFoundError:
        print(f"Error: El archivo {nombre_archivo} no fue encontrado.")
        return None
    except ValueError as e:
        print(f"Error: Problema al convertir un valor en {nombre_archivo}. Detalle: {e}")
        return None
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer {nombre_archivo}: {e}")
        return None

if __name__ == '__main__':
    # Ejemplo de uso (puedes probarlo con tus archivos de caso)
    datos_case1 = read_case('case1.txt')
    if datos_case1:
        print("\nDatos de case1.txt:")
        print(f"Número de aviones: {datos_case1['num_aviones']}")
        # print("Datos de los aviones:")
        # for avion in datos_case1['aviones']:
        #     print(avion)
        # print("Matriz de tiempos de separación:")
        # for fila in datos_case1['tiempos_separacion']:
        #     print(fila)
        print(f"Primer avión: {datos_case1['aviones'][0]}")
        print(f"Tiempos de separación para el primer avión: {datos_case1['tiempos_separacion'][0]}")
        print(f"Último avión: {datos_case1['aviones'][-1]}")
        print(f"Tiempos de separación para el último avión: {datos_case1['tiempos_separacion'][-1]}")


    datos_case4 = read_case('case4.txt')
    if datos_case4:
        print("\nDatos de case4.txt:")
        print(f"Número de aviones: {datos_case4['num_aviones']}")
        print(f"Primer avión: {datos_case4['aviones'][0]}")
        print(f"Tiempos de separación para el primer avión: {datos_case4['tiempos_separacion'][0]}")
        print(f"Avión 50 (índice 49): {datos_case4['aviones'][49]}") # Para verificar uno intermedio en un caso grande
        print(f"Tiempos de separación para el avión 50: {datos_case4['tiempos_separacion'][49]}")
        print(f"Último avión: {datos_case4['aviones'][-1]}")
        print(f"Tiempos de separación para el último avión: {datos_case4['tiempos_separacion'][-1]}")