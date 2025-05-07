import csv
import os
import re
from collections import defaultdict
import statistics # Para mean

# Necesitamos pandas para pd.isna en parse_detalles_parametros, si se usa fuera del bloque main
try:
    import pandas as pd
except ImportError:
    pass # Se importará en main o se manejará el error allí si es estrictamente necesario.

# Mapeo global para el número de aviones por caso.
AVIONES_MAP = {
    'case1.txt': 15,
    'case2.txt': 20,
    'case3.txt': 44,
    'case4.txt': 100
}

def parse_detalles_parametros(detalles_str):
    """Parsea la cadena 'DetallesParametros' en un diccionario."""
    params = {}
    is_na_like = False
    if 'pd' in globals() and callable(pd.isna):
        is_na_like = pd.isna(detalles_str)
    
    if is_na_like or detalles_str == 'N/A' or not detalles_str :
        return params
        
    parts = detalles_str.split(';')
    for part in parts:
        if ':' in part:
            key_val = part.split(':', 1)
            if len(key_val) == 2:
                params[key_val[0].strip()] = key_val[1].strip()
    return params

def sort_key_caso_pistas_params(item_key):
    """
    Clave para ordenar nombres de caso o tuplas (nombre_caso, num_pistas, ...).
    """
    nombre_caso_str = ''
    num_pistas_val = 0 
    otros_params_vals = []

    if isinstance(item_key, tuple):
        nombre_caso_str = item_key[0]
        if len(item_key) > 1:
            try:
                num_pistas_val = int(item_key[1])
            except (ValueError, TypeError):
                num_pistas_val = 0 
        
        if len(item_key) > 2: 
            for p_val in item_key[2:]:
                try:
                    otros_params_vals.append(int(p_val))
                except (ValueError, TypeError):
                    otros_params_vals.append(str(p_val))
    else: 
        nombre_caso_str = str(item_key) 
        num_pistas_val = 0 
    
    match_caso = re.search(r'\d+', nombre_caso_str)
    num_caso_val = int(match_caso.group(0)) if match_caso else float('inf')

    return (num_caso_val, num_pistas_val, *otros_params_vals)

def procesar_csv(filepath):
    """Lee el archivo CSV y separa las filas por algoritmo."""
    datos_por_algoritmo = defaultdict(list)
    fieldnames = []
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            if not reader.fieldnames:
                print(f"Error: El archivo CSV '{filepath}' está vacío o no tiene encabezados.")
                return None, None
            fieldnames = reader.fieldnames
            
            required_columns = [
                'NombreCaso', 'Algoritmo', 'NumPistas', 'DetallesParametros',
                'CostoPenalizadoSolucion', 'EsEstrictamenteFactible', 'TiempoComputacional_seg'
            ]
            for col in required_columns:
                if col not in fieldnames: 
                    print(f"Error: Falta la columna esperada '{col}' en el archivo CSV '{filepath}'.")
                    print(f"Columnas encontradas: {fieldnames}")
                    return None, None

            for row in reader:
                algo = row.get('Algoritmo')
                if algo:
                    datos_por_algoritmo[algo].append(row)
        return datos_por_algoritmo, fieldnames
    except FileNotFoundError:
        print(f"Error: El archivo '{filepath}' no fue encontrado.")
        return None, None
    except Exception as e:
        print(f"Ocurrió un error inesperado al leer el archivo CSV: {e}")
        return None, None

def generar_tabla_gd(datos_list, titulo="Greedy Determinista (GD)"):
    if not datos_list:
        print(f"\nNo se encontraron datos del algoritmo '{titulo.split('(')[0].strip()}' para generar la tabla.")
        return

    resultados_agrupados = {}
    for row in datos_list:
        caso = row.get('NombreCaso')
        num_pistas = row.get('NumPistas')

        if not caso or not num_pistas:
            print(f"Advertencia ({titulo.split('(')[0].strip()}): Fila omitida por datos faltantes: {row}")
            continue

        try:
            costo_str = row.get('CostoPenalizadoSolucion', 'N/A')
            tiempo_str = row.get('TiempoComputacional_seg', 'N/A')
            factible = row.get('EsEstrictamenteFactible') == 'True'

            costo_val = float(costo_str) if costo_str not in ['N/A', 'Error', 'INFACTIBLE'] else costo_str
            
            if not factible and isinstance(costo_val, float):
                 costo_display = f"{costo_val:.2f} (NF)"
            elif isinstance(costo_val, float):
                 costo_display = f"{costo_val:.2f}"
            elif not factible and costo_str != 'INFACTIBLE': 
                 costo_display = f"{costo_str} (NF)"
            else: 
                 costo_display = costo_str

            tiempo_val = float(tiempo_str) if tiempo_str not in ['N/A', 'Error'] else tiempo_str
            tiempo_display = f"{tiempo_val:.4f}s" if isinstance(tiempo_val, float) else tiempo_str

        except ValueError:
            costo_display = "ErrorConv"
            tiempo_display = "ErrorConv"
        
        if caso not in resultados_agrupados:
            resultados_agrupados[caso] = {
                'aviones': AVIONES_MAP.get(caso, 'N/A'),
                'costo_1p': 'N/A', 'tiempo_1p': 'N/A',
                'costo_2p': 'N/A', 'tiempo_2p': 'N/A'
            }
        
        if num_pistas == '1':
            resultados_agrupados[caso]['costo_1p'] = costo_display
            resultados_agrupados[caso]['tiempo_1p'] = tiempo_display
        elif num_pistas == '2':
            resultados_agrupados[caso]['costo_2p'] = costo_display
            resultados_agrupados[caso]['tiempo_2p'] = tiempo_display

    print(f"\nTabla Resumen: {titulo}")
    line_length = 82
    print("=" * line_length)
    header = (f"{'Caso':<15} | {'Aviones':>7} | {'Costo (1P)':>10} | {'Tiempo (1P)':>11} | "
              f"{'Costo (2P)':>10} | {'Tiempo (2P)':>11}")
    print(header)
    print("-" * line_length)

    sorted_casos = sorted(resultados_agrupados.keys(), key=sort_key_caso_pistas_params)
    for caso_nombre in sorted_casos:
        res = resultados_agrupados[caso_nombre]
        fila = (f"{caso_nombre:<15} | {str(res['aviones']):>7} | {str(res['costo_1p']):>10} | {str(res['tiempo_1p']):>11} | "
                f"{str(res['costo_2p']):>10} | {str(res['tiempo_2p']):>11}")
        print(fila)
    print("-" * line_length)

def generar_tabla_ge_solo(datos_list):
    if not datos_list:
        print("\nNo se encontraron datos del algoritmo 'GE_Solo' para generar la tabla.")
        return

    resultados_ge_temp = defaultdict(lambda: {'costos_factibles': [], 'tiempos': [], 'no_factibles': 0, 'total_ejecuciones': 0, 'k_rcl_set': set()})

    for row in datos_list:
        caso = row.get('NombreCaso')
        num_pistas = row.get('NumPistas')
        detalles_str = row.get('DetallesParametros', '')
        params = parse_detalles_parametros(detalles_str)
        k_rcl = params.get('K_RCL', 'N/A') 

        if not caso or not num_pistas:
            print(f"Advertencia (GE_Solo): Fila omitida por datos faltantes: {row}")
            continue
        
        key = (caso, num_pistas) 
        resultados_ge_temp[key]['total_ejecuciones'] += 1
        resultados_ge_temp[key]['k_rcl_set'].add(k_rcl)

        factible = row.get('EsEstrictamenteFactible') == 'True'
        costo_penalizado_str = row.get('CostoPenalizadoSolucion', '0')
        tiempo_str = row.get('TiempoComputacional_seg', '0')

        try:
            tiempo_val = float(tiempo_str)
            resultados_ge_temp[key]['tiempos'].append(tiempo_val)
        except ValueError:
            print(f"Advertencia (GE_Solo): Tiempo no numérico '{tiempo_str}' para {key} en fila {row}. Omitiendo tiempo.")

        if factible:
            try:
                costo_val = float(costo_penalizado_str)
                resultados_ge_temp[key]['costos_factibles'].append(costo_val)
            except ValueError:
                print(f"Advertencia (GE_Solo): Costo no numérico '{costo_penalizado_str}' para solución factible {key} en fila {row}. Contado como no factible.")
                resultados_ge_temp[key]['no_factibles'] += 1
        else:
            resultados_ge_temp[key]['no_factibles'] += 1
            
    print("\nTabla Resumen: Greedy Estocástico (GE_Solo)")
    line_length = 115
    print("=" * line_length)
    header = (f"{'Caso':<15} | {'Aviones':>7} | {'Pistas':>6} | {'K_RCL':>5} | {'Mejor Costo':>12} | {'Costo Prom.':>12} | "
              f"{'Peor Costo':>12} | {'Tiem. Prom.(s)':>14} | {'No Fact.':>8} | {'Tot. Ejec.':>10}")
    print(header)
    print("-" * line_length)

    sorted_keys_ge = sorted(resultados_ge_temp.keys(), key=sort_key_caso_pistas_params)

    for key in sorted_keys_ge:
        caso_nombre, num_pistas_str = key
        data = resultados_ge_temp[key]
        
        aviones = AVIONES_MAP.get(caso_nombre, 'N/A')
        costos_fact = data['costos_factibles']
        tiempos = data['tiempos']
        
        k_rcl_display = list(data['k_rcl_set'])[0] if len(data['k_rcl_set']) == 1 else "Mixto"

        mejor_costo = f"{min(costos_fact):.2f}" if costos_fact else "N/A"
        peor_costo = f"{max(costos_fact):.2f}" if costos_fact else "N/A"
        costo_prom = f"{statistics.mean(costos_fact):.2f}" if costos_fact else "N/A"
        tiempo_prom = f"{statistics.mean(tiempos):.4f}" if tiempos else "N/A"
        
        no_factibles = data['no_factibles']
        total_ejecuciones = data['total_ejecuciones']

        fila = (f"{caso_nombre:<15} | {str(aviones):>7} | {num_pistas_str:>6} | {str(k_rcl_display):>5} | {mejor_costo:>12} | {costo_prom:>12} | "
                f"{peor_costo:>12} | {tiempo_prom:>14} | {str(no_factibles):>8} | {str(total_ejecuciones):>10}")
        print(fila)
    print("-" * line_length)

def generar_tabla_grasp_con_restarts(datos_list_dict):
    """
    Genera tablas para las variantes de GRASP_HC_Estoc_XRestarts y GRASP_HC_Det_0Restarts.
    datos_list_dict: {'GRASP_HC_Estoc_10Restarts': [rows], ...}
    """
    for algo_nombre_completo, datos_list in datos_list_dict.items():
        if not datos_list:
            continue 

        resultados_grasp_temp = defaultdict(lambda: {
            'costos_factibles': [], 'tiempos': [], 'no_factibles': 0, 
            'total_ejecuciones': 0, 'detalles_comunes': {}
        })

        for row in datos_list:
            caso = row.get('NombreCaso')
            num_pistas = row.get('NumPistas')
            detalles_str = row.get('DetallesParametros', '')
            params = parse_detalles_parametros(detalles_str)

            if not caso or not num_pistas:
                print(f"Advertencia ({algo_nombre_completo}): Fila omitida por datos faltantes: {row}")
                continue
            
            key = (caso, num_pistas) 
            resultados_grasp_temp[key]['total_ejecuciones'] += 1
            if not resultados_grasp_temp[key]['detalles_comunes']: 
                # Guardar parámetros comunes (sin 'Inicio' o 'Semilla' que varían por ejecución)
                resultados_grasp_temp[key]['detalles_comunes'] = {
                    k:v for k,v in params.items() if k not in ['Inicio', 'Semilla'] 
                }

            factible = row.get('EsEstrictamenteFactible') == 'True'
            costo_penalizado_str = row.get('CostoPenalizadoSolucion', '0')
            tiempo_str = row.get('TiempoComputacional_seg', '0')

            try:
                tiempo_val = float(tiempo_str)
                resultados_grasp_temp[key]['tiempos'].append(tiempo_val)
            except ValueError:
                print(f"Advertencia ({algo_nombre_completo}): Tiempo no numérico '{tiempo_str}' para {key} en fila {row}. Omitiendo tiempo.")

            if factible:
                try:
                    costo_val = float(costo_penalizado_str)
                    resultados_grasp_temp[key]['costos_factibles'].append(costo_val)
                except ValueError:
                    print(f"Advertencia ({algo_nombre_completo}): Costo no numérico '{costo_penalizado_str}' para solución factible {key} en fila {row}. Contado como no factible.")
                    resultados_grasp_temp[key]['no_factibles'] += 1
            else:
                resultados_grasp_temp[key]['no_factibles'] += 1
                
        print(f"\nTabla Resumen: {algo_nombre_completo.replace('_', ' ')}")
        
        iter_g_disp = "N/A"
        k_rcl_disp = "N/A"
        iter_hc_disp = "N/A"
        inicio_disp = "N/A" # Para GRASP_HC_Det_0Restarts

        if resultados_grasp_temp:
            first_key_data = next(iter(resultados_grasp_temp.values()))
            common_params = first_key_data['detalles_comunes']
            iter_g_disp = common_params.get('IterGRASP', 'N/A')
            k_rcl_disp = common_params.get('K_RCL', 'N/A')
            iter_hc_disp = common_params.get('IterHC', 'N/A')
            if algo_nombre_completo == 'GRASP_HC_Det_0Restarts':
                inicio_disp = common_params.get('Inicio', 'N/A')


        line_length = 135 if "Estoc" in algo_nombre_completo else 125
        header_parts = [
            f"{'Caso':<15}", f"{'Aviones':>7}", f"{'Pistas':>6}",
            f"{'IterGRASP':>9}", f"{'IterHC':>6}"
        ]
        if "Estoc" in algo_nombre_completo:
            header_parts.append(f"{'K_RCL':>5}")
        elif algo_nombre_completo == 'GRASP_HC_Det_0Restarts':
             header_parts.append(f"{'Inicio':>8}")


        header_parts.extend([
            f"{'Mejor Costo':>12}", f"{'Costo Prom.':>12}", f"{'Peor Costo':>12}",
            f"{'Tiem. Prom.(s)':>14}", f"{'No Fact.':>8}", f"{'Ejec.':>6}"
        ])
        header = " | ".join(header_parts)
        
        print("=" * len(header))
        print(header)
        print("-" * len(header))

        sorted_keys_grasp = sorted(resultados_grasp_temp.keys(), key=sort_key_caso_pistas_params)

        for key in sorted_keys_grasp:
            caso_nombre, num_pistas_str = key
            data = resultados_grasp_temp[key]
            
            aviones = AVIONES_MAP.get(caso_nombre, 'N/A')
            costos_fact = data['costos_factibles']
            tiempos = data['tiempos']
            
            current_common_params = data['detalles_comunes']
            iter_g_val = current_common_params.get('IterGRASP', 'N/A')
            k_rcl_val = current_common_params.get('K_RCL', 'N/A')
            iter_hc_val = current_common_params.get('IterHC', 'N/A')
            inicio_val = current_common_params.get('Inicio', 'N/A')


            mejor_costo = f"{min(costos_fact):.2f}" if costos_fact else "N/A"
            peor_costo = f"{max(costos_fact):.2f}" if costos_fact else "N/A"
            costo_prom = f"{statistics.mean(costos_fact):.2f}" if costos_fact else "N/A"
            tiempo_prom = f"{statistics.mean(tiempos):.4f}" if tiempos else "N/A"
            
            no_factibles = data['no_factibles']
            total_ejecuciones = data['total_ejecuciones']

            fila_parts = [
                f"{caso_nombre:<15}", f"{str(aviones):>7}", f"{num_pistas_str:>6}",
                f"{str(iter_g_val):>9}", f"{str(iter_hc_val):>6}"
            ]
            if "Estoc" in algo_nombre_completo:
                fila_parts.append(f"{str(k_rcl_val):>5}")
            elif algo_nombre_completo == 'GRASP_HC_Det_0Restarts':
                fila_parts.append(f"{str(inicio_val):>8}")


            fila_parts.extend([
                f"{mejor_costo:>12}", f"{costo_prom:>12}", f"{peor_costo:>12}",
                f"{tiempo_prom:>14}", f"{str(no_factibles):>8}", f"{str(total_ejecuciones):>6}"
            ])
            print(" | ".join(fila_parts))
        print("-" * len(header))


if __name__ == '__main__':
    try:
        import pandas as pd 
    except ImportError:
        print("La librería pandas es necesaria. Por favor, instálala: pip install pandas")
        exit()

    csv_file_path = os.path.join('.', 'results', 'resultado.csv')
    
    if not os.path.exists(csv_file_path):
        print(f"Error CRÍTICO: El archivo '{csv_file_path}' no se encuentra.")
        print("Asegúrate de que el archivo existe en la ruta especificada y tiene datos.")
        exit()

    datos_por_algoritmo, fieldnames = procesar_csv(csv_file_path)

    if datos_por_algoritmo:
        generar_tabla_gd(datos_por_algoritmo.get('GD', []), titulo="Greedy Determinista (GD)")
        generar_tabla_ge_solo(datos_por_algoritmo.get('GE_Solo', []))
        
        grasp_det_datos = {'GRASP_HC_Det_0Restarts': datos_por_algoritmo.get('GRASP_HC_Det_0Restarts', [])}
        generar_tabla_grasp_con_restarts(grasp_det_datos)

        grasp_estoc_variantes_datos = {
            'GRASP_HC_Estoc_10Restarts': datos_por_algoritmo.get('GRASP_HC_Estoc_10Restarts', []),
            'GRASP_HC_Estoc_50Restarts': datos_por_algoritmo.get('GRASP_HC_Estoc_50Restarts', []),
            'GRASP_HC_Estoc_100Restarts': datos_por_algoritmo.get('GRASP_HC_Estoc_100Restarts', [])
        }
        generar_tabla_grasp_con_restarts(grasp_estoc_variantes_datos)
        
        print("\nNotas Generales:")
        print("  - El número de 'Aviones' se basa en un mapeo predefinido en el script.")
        print("  - 'Costo' en las tablas se refiere a 'CostoPenalizadoSolucion' del CSV.")
        print("  - Para GE_Solo y GRASP Estocástico, las estadísticas de costo (Mejor, Prom., Peor) son sobre soluciones estrictamente factibles.")
        print("  - (NF) junto a un costo en la tabla GD indica que la solución no fue 'EsEstrictamenteFactible=True'.")

