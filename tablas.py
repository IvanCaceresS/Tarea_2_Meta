import csv
import os
import re
from collections import defaultdict
import statistics 
import pandas as pd

AVIONES_MAP = {
    'case1.txt': 15,
    'case2.txt': 20,
    'case3.txt': 44,
    'case4.txt': 100
}

def is_na(value):
    return pd.isna(value)

def parse_detalles_parametros(detalles_str):
    """Parsea la cadena 'DetallesParametros' en un diccionario."""
    params = {}
    if is_na(detalles_str) or detalles_str == 'N/A' or not detalles_str :
        return params
        
    parts = detalles_str.split(';')
    for part in parts:
        if ':' in part:
            key_val = part.split(':', 1)
            if len(key_val) == 2:
                key = key_val[0].strip()
                value = key_val[1].strip()
                try:
                    num_val = float(value)
                    if num_val == int(num_val):
                        params[key] = int(num_val)
                    else:
                        params[key] = num_val
                except ValueError:
                    if key == 'AspirationCriteria':
                         params[key] = value.lower() == 'true' if value.lower() in ['true', 'false'] else value
                    else:
                        params[key] = value
    match_ten = re.search(r'Ten:(\d+)', detalles_str)
    if match_ten:
        try:
            params['Ten'] = int(match_ten.group(1))
        except ValueError:
            pass 
    match_cfg = re.search(r'CfgID:([^;]+)', detalles_str)
    if match_cfg:
        params['CfgID'] = match_cfg.group(1).strip()
        
    match_inicio = re.search(r'SolInicial:([^;]+)', detalles_str) 
    if match_inicio:
        params['SolInicial'] = match_inicio.group(1).strip()
    
    match_max_iter = re.search(r'MaxIter:(\d+)', detalles_str)
    if match_max_iter and 'MaxIter' not in params:
        try:
            params['MaxIter'] = int(match_max_iter.group(1))
        except ValueError:
            pass
    
    match_k_rcl = re.search(r'K_RCL:(\d+)', detalles_str)
    if match_k_rcl and 'K_RCL' not in params:
        try:
            params['K_RCL'] = int(match_k_rcl.group(1))
        except ValueError:
            pass

    match_iter_grasp = re.search(r'IterGRASP:(\d+)', detalles_str)
    if match_iter_grasp and 'IterGRASP' not in params:
        try:
            params['IterGRASP'] = int(match_iter_grasp.group(1))
        except ValueError:
            pass

    match_iter_hc = re.search(r'IterHC:(\d+)', detalles_str)
    if match_iter_hc and 'IterHC' not in params:
        try:
            params['IterHC'] = int(match_iter_hc.group(1))
        except ValueError:
            pass
            
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
                    try:
                       numeric_val = float(p_val)
                       if numeric_val == int(numeric_val):
                           otros_params_vals.append(int(numeric_val))
                       else:
                           otros_params_vals.append(numeric_val)
                    except ValueError:
                         otros_params_vals.append(str(p_val)) 
                except (ValueError, TypeError):
                     otros_params_vals.append(str(p_val))
    else: 
        nombre_caso_str = str(item_key) 
        num_pistas_val = 0 
    
    match_caso = re.search(r'\d+', nombre_caso_str)
    num_caso_val = int(match_caso.group(0)) if match_caso else float('inf')

    ten_val = float('inf')
    if isinstance(item_key, tuple) and len(item_key) > 2:
         numeric_params_after_pistas = [p for p in item_key[2:] if isinstance(p, (int, float))]
         if numeric_params_after_pistas: 
             ten_val = numeric_params_after_pistas[0]


    return (num_caso_val, num_pistas_val, ten_val, *otros_params_vals) 


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
            missing_cols = [col for col in required_columns if col not in fieldnames]
            if missing_cols:
                print(f"Error: Faltan las columnas esperadas {missing_cols} en el archivo CSV '{filepath}'.")
                print(f"Columnas encontradas: {fieldnames}")
                return None, None

            for i, row in enumerate(reader):
                algo = row.get('Algoritmo')
                row['parsed_details'] = parse_detalles_parametros(row.get('DetallesParametros'))
                factible_str = str(row.get('EsEstrictamenteFactible', 'False')).strip().lower()
                row['EsEstrictamenteFactible_Bool'] = factible_str == 'true'

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
            factible = row.get('EsEstrictamenteFactible_Bool') # Usar valor booleano pre-procesado

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
        
        if str(num_pistas) == '1':
            resultados_agrupados[caso]['costo_1p'] = costo_display
            resultados_agrupados[caso]['tiempo_1p'] = tiempo_display
        elif str(num_pistas) == '2':
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
        params = row.get('parsed_details', {}) 
        k_rcl = params.get('K_RCL', 'N/A') 

        if not caso or not num_pistas:
            print(f"Advertencia (GE_Solo): Fila omitida por datos faltantes: {row}")
            continue
        
        key = (caso, num_pistas) 
        resultados_ge_temp[key]['total_ejecuciones'] += 1
        resultados_ge_temp[key]['k_rcl_set'].add(k_rcl)

        factible = row.get('EsEstrictamenteFactible_Bool') # Usar valor booleano
        costo_penalizado_str = row.get('CostoPenalizadoSolucion', '0')
        tiempo_str = row.get('TiempoComputacional_seg', '0')

        try:
            tiempo_val = float(tiempo_str)
            resultados_ge_temp[key]['tiempos'].append(tiempo_val)
        except (ValueError, TypeError):
            pass 
        
        if factible:
            try:
                costo_val = float(costo_penalizado_str)
                resultados_ge_temp[key]['costos_factibles'].append(costo_val)
            except (ValueError, TypeError):
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
        costo_prom = f"{statistics.mean(costos_fact):.2f}" if len(costos_fact) > 0 else "N/A"
        tiempo_prom = f"{statistics.mean(tiempos):.4f}" if len(tiempos) > 0 else "N/A"
        
        no_factibles = data['no_factibles']
        total_ejecuciones = data['total_ejecuciones']

        fila = (f"{caso_nombre:<15} | {str(aviones):>7} | {num_pistas_str:>6} | {str(k_rcl_display):>5} | {mejor_costo:>12} | {costo_prom:>12} | "
                f"{peor_costo:>12} | {tiempo_prom:>14} | {str(no_factibles):>8} | {str(total_ejecuciones):>10}")
        print(fila)
    print("-" * line_length)

def get_grasp_restarts_from_algo_name(algo_name):
    """Extrae el número de restarts del nombre del algoritmo GRASP."""
    if algo_name == 'GRASP_HC_Det_0Restarts':
        return 0
    match = re.search(r'_R(\d+)_', algo_name)
    return int(match.group(1)) if match else -1 

def generar_tabla_grasp_con_restarts(datos_list_dict):
    """
    Genera tablas para las variantes de GRASP_HC_Estoc_R{X}_SGE{Y} y GRASP_HC_Det_0Restarts.
    """
    datos_agrupados_por_restart = defaultdict(list)
    
    if 'GRASP_HC_Det_0Restarts' in datos_list_dict:
         datos_agrupados_por_restart[0].extend(datos_list_dict['GRASP_HC_Det_0Restarts'])

    for algo_nombre_completo, datos_list in datos_list_dict.items():
         restarts = get_grasp_restarts_from_algo_name(algo_nombre_completo)
         if restarts > 0: 
             datos_agrupados_por_restart[restarts].extend(datos_list)

    for restarts, datos_list in sorted(datos_agrupados_por_restart.items()):
        if not datos_list:
            continue 

        algo_titulo = f"GRASP HC {'Determinista (0 Restarts)' if restarts == 0 else f'Estocástico ({restarts} Restarts)'}"
        
        resultados_grasp_temp = defaultdict(lambda: {
            'costos_factibles': [], 'tiempos': [], 'no_factibles': 0, 
            'total_ejecuciones': 0, 'detalles_comunes': {} 
        })

        for row in datos_list:
            caso = row.get('NombreCaso')
            num_pistas = row.get('NumPistas')
            params = row.get('parsed_details', {})

            if not caso or not num_pistas:
                print(f"Advertencia ({algo_titulo}): Fila omitida por datos faltantes: {row}")
                continue
            
            key = (caso, num_pistas) 
            resultados_grasp_temp[key]['total_ejecuciones'] += 1
            
            if not resultados_grasp_temp[key]['detalles_comunes']: 
                keys_to_exclude = ['Semilla', 'CfgID'] 
                resultados_grasp_temp[key]['detalles_comunes'] = {
                    k:v for k,v in params.items() if k not in keys_to_exclude
                }

            factible = row.get('EsEstrictamenteFactible_Bool') # Usar valor booleano
            costo_penalizado_str = row.get('CostoPenalizadoSolucion', '0')
            tiempo_str = row.get('TiempoComputacional_seg', '0')

            try:
                tiempo_val = float(tiempo_str)
                resultados_grasp_temp[key]['tiempos'].append(tiempo_val)
            except (ValueError, TypeError):
                pass

            if factible:
                try:
                    costo_val = float(costo_penalizado_str)
                    resultados_grasp_temp[key]['costos_factibles'].append(costo_val)
                except (ValueError, TypeError):
                    resultados_grasp_temp[key]['no_factibles'] += 1
            else:
                resultados_grasp_temp[key]['no_factibles'] += 1
                
        print(f"\nTabla Resumen: {algo_titulo}")

        header_parts = [
            f"{'Caso':<15}", f"{'Aviones':>7}", f"{'Pistas':>6}",
            f"{'IterGRASP':>9}", f"{'IterHC':>6}"
        ]
        if restarts > 0: # Estocástico
            header_parts.append(f"{'K_RCL':>5}")
            header_parts.extend([
                f"{'Mejor Costo':>12}", f"{'Costo Prom.':>12}", f"{'Peor Costo':>12}",
                f"{'Tiem. Prom.(s)':>14}", f"{'No Fact.':>8}", f"{'Ejec.':>6}"
            ])
        else: # Determinista
             header_parts.append(f"{'Inicio':>8}") # 'SolInicial' es el parámetro
             header_parts.extend([
                f"{'Costo':>12}", 
                f"{'Tiempo (s)':>14}", 
                f"{'No Fact.':>8}", 
                f"{'Ejec.':>6}" 
            ])

        header = " | ".join(header_parts)
        line_length = len(header)
        
        print("=" * line_length)
        print(header)
        print("-" * line_length)

        sorted_keys_grasp = sorted(resultados_grasp_temp.keys(), key=sort_key_caso_pistas_params)

        for key in sorted_keys_grasp:
            caso_nombre, num_pistas_str = key
            data = resultados_grasp_temp[key] # data es resultados_grasp_temp[key]
            
            aviones = AVIONES_MAP.get(caso_nombre, 'N/A')
            costos_fact = data['costos_factibles']
            tiempos = data['tiempos']
            
            current_common_params = data['detalles_comunes'] # Usar los detalles comunes del grupo
            iter_g_val = current_common_params.get('IterGRASP', 'N/A')
            k_rcl_val = current_common_params.get('K_RCL', 'N/A')
            iter_hc_val = current_common_params.get('IterHC', 'N/A')
            inicio_val = current_common_params.get('SolInicial', 'N/A') 

            tiempo_prom = f"{statistics.mean(tiempos):.4f}" if len(tiempos) > 0 else "N/A"
            no_factibles = data['no_factibles']
            total_ejecuciones = data['total_ejecuciones']

            fila_parts = [
                f"{caso_nombre:<15}", f"{str(aviones):>7}", f"{num_pistas_str:>6}",
                f"{str(iter_g_val):>9}", f"{str(iter_hc_val):>6}"
            ]
            
            if restarts > 0: # Estocástico
                mejor_costo = f"{min(costos_fact):.2f}" if costos_fact else "N/A"
                peor_costo = f"{max(costos_fact):.2f}" if costos_fact else "N/A"
                costo_prom = f"{statistics.mean(costos_fact):.2f}" if len(costos_fact) > 0 else "N/A"
                fila_parts.append(f"{str(k_rcl_val):>5}")
                fila_parts.extend([
                    f"{mejor_costo:>12}", f"{costo_prom:>12}", f"{peor_costo:>12}",
                    f"{tiempo_prom:>14}", f"{str(no_factibles):>8}", f"{str(total_ejecuciones):>6}"
                ])
            else: # Determinista
                costo_display = f"{costos_fact[0]:.2f}" if costos_fact else "N/A" 
                fila_parts.append(f"{str(inicio_val):>8}") # Usar inicio_val
                fila_parts.extend([
                    f"{costo_display:>12}", 
                    f"{tiempo_prom:>14}", 
                    f"{str(no_factibles):>8}", 
                    f"{str(total_ejecuciones):>6}"
                ])

            print(" | ".join(fila_parts))
        print("-" * line_length)


def generar_tabla_ts_det(datos_list):
    """Genera tabla para TS Determinista (TS_GD), mostrando cada configuración."""
    if not datos_list:
        print("\nNo se encontraron datos del algoritmo 'TS_GD' para generar la tabla.")
        return
        
    params_ts_det = {
        'SolInicial': ('Inicio', 8), 
        'Ten': ('Ten', 5), 
        'MaxIter': ('IterTS', 7), 
        'MaxNoImp': ('NoImp', 8),
    }
    generar_tabla_algoritmo_individual(datos_list, "Tabu Search Determinista (Inicio GD)", params_ts_det)


def generar_tabla_ts_estoc(datos_list):
    """Genera tabla resumen para TS Estocástico (TS_GE), agrupando por Ten."""
    if not datos_list:
        print("\nNo se encontraron datos del algoritmo 'TS_GE' para generar la tabla.")
        return

    resultados_ts_temp = defaultdict(lambda: {
        'costos_factibles': [], 'tiempos': [], 'no_factibles': 0, 
        'total_ejecuciones': 0, 'detalles_comunes': {}
    })

    for row in datos_list:
        caso = row.get('NombreCaso')
        num_pistas = row.get('NumPistas')
        params = row.get('parsed_details', {})
        tenure = params.get('Ten', 'N/A')

        if not caso or not num_pistas or tenure == 'N/A':
            continue
        
        key = (caso, num_pistas, tenure) 
        resultados_ts_temp[key]['total_ejecuciones'] += 1
        if not resultados_ts_temp[key]['detalles_comunes']:
            keys_to_exclude = ['Inicio', 'Semilla', 'CfgID', 'SolInicial'] 
            resultados_ts_temp[key]['detalles_comunes'] = {
                k:v for k,v in params.items() if k not in keys_to_exclude 
            }


        factible = row.get('EsEstrictamenteFactible_Bool') 
        costo_penalizado_str = row.get('CostoPenalizadoSolucion', '0')
        tiempo_str = row.get('TiempoComputacional_seg', '0')

        try:
            tiempo_val = float(tiempo_str)
            resultados_ts_temp[key]['tiempos'].append(tiempo_val)
        except (ValueError, TypeError):
             pass 

        if factible:
            try:
                costo_val = float(costo_penalizado_str)
                resultados_ts_temp[key]['costos_factibles'].append(costo_val)
            except (ValueError, TypeError):
                resultados_ts_temp[key]['no_factibles'] += 1
        else:
            resultados_ts_temp[key]['no_factibles'] += 1

    print("\nTabla Resumen: Tabu Search Estocástico (Inicio GE)")
    
    header_parts = [
        f"{'Caso':<15}", f"{'Aviones':>7}", f"{'Pistas':>6}", 
        f"{'Ten':>5}", f"{'IterTS':>7}", f"{'MaxNoImp':>8}", # IterTS es el nombre de la columna
        f"{'Mejor Costo':>12}", f"{'Costo Prom.':>12}", f"{'Peor Costo':>12}",
        f"{'Tiem. Prom.(s)':>14}", f"{'No Fact.':>8}", f"{'Ejec.':>6}"
    ]
    header = " | ".join(header_parts)
    line_length = len(header)
    print("=" * line_length)
    print(header)
    print("-" * line_length)

    sorted_keys_ts = sorted(resultados_ts_temp.keys(), key=sort_key_caso_pistas_params)

    for key in sorted_keys_ts:
        caso_nombre, num_pistas_str, ten_val = key
        data = resultados_ts_temp[key]
        
        aviones = AVIONES_MAP.get(caso_nombre, 'N/A')
        costos_fact = data['costos_factibles']
        tiempos = data['tiempos']
        
        common_params = data['detalles_comunes']
        iter_ts_val = common_params.get('MaxIter', 'N/A') 
        max_no_imp_val = common_params.get('MaxNoImp', 'N/A')

        mejor_costo = f"{min(costos_fact):.2f}" if costos_fact else "N/A"
        peor_costo = f"{max(costos_fact):.2f}" if costos_fact else "N/A"
        costo_prom = f"{statistics.mean(costos_fact):.2f}" if len(costos_fact) > 0 else "N/A"
        tiempo_prom = f"{statistics.mean(tiempos):.4f}" if len(tiempos) > 0 else "N/A"
        
        no_factibles = data['no_factibles']
        total_ejecuciones = data['total_ejecuciones']

        fila_parts = [
            f"{caso_nombre:<15}", f"{str(aviones):>7}", f"{num_pistas_str:>6}", 
            f"{str(ten_val):>5}", f"{str(iter_ts_val):>7}", f"{str(max_no_imp_val):>8}", 
            f"{mejor_costo:>12}", f"{costo_prom:>12}", f"{peor_costo:>12}",
            f"{tiempo_prom:>14}", f"{str(no_factibles):>8}", f"{str(total_ejecuciones):>6}"
        ]
        print(" | ".join(fila_parts))
    print("-" * line_length)


def generar_tabla_algoritmo_individual(datos_list, algoritmo_nombre, param_keys_to_show_dict):
    if not datos_list:
        print(f"\nNo se encontraron datos del algoritmo '{algoritmo_nombre}' para generar la tabla.")
        return

    print(f"\nTabla Resumen: {algoritmo_nombre}")
    
    header_parts = [f"{'Caso':<15}", f"{'Aviones':>7}", f"{'Pistas':>6}"]
    for display_name, width in param_keys_to_show_dict.values(): # param_keys_to_show_dict es {'CSV_Key': ('DisplayName', width)}
        header_parts.append(f"{display_name:>{width}}")
        
    header_parts.extend([f"{'Costo':>10}", f"{'Tiempo (s)':>11}", f"{'Factible':>8}"])
    header_str = " | ".join(header_parts)
    line_length = len(header_str)
    print("=" * line_length)
    print(header_str)
    print("-" * line_length)
    
    def get_sort_key_for_individual(row):
        caso = row.get('NombreCaso')
        pistas = row.get('NumPistas')
        parsed_params = row.get('parsed_details', {})
        
        sort_tuple_elements = list(sort_key_caso_pistas_params((caso,pistas))) 
        for pk_csv in param_keys_to_show_dict.keys(): # Iterar sobre las claves CSV del diccionario
            val = parsed_params.get(pk_csv, 'N/A')
            try:
                 try:
                     num_val = float(val)
                     if num_val == int(num_val):
                         sort_tuple_elements.append(int(num_val))
                     else:
                         sort_tuple_elements.append(num_val)
                 except ValueError:
                     sort_tuple_elements.append(str(val)) 
            except (ValueError, TypeError):
                 sort_tuple_elements.append(str(val))
        return tuple(sort_tuple_elements)

    sorted_datos = sorted(datos_list, key=get_sort_key_for_individual)

    for row in sorted_datos:
        caso = row.get('NombreCaso')
        aviones = AVIONES_MAP.get(caso, 'N/A')
        num_pistas = row.get('NumPistas')
        parsed_params = row.get('parsed_details', {})

        costo_str = row.get('CostoPenalizadoSolucion', 'N/A')
        tiempo_str = row.get('TiempoComputacional_seg', 'N/A')
        factible_str = "Sí" if row.get('EsEstrictamenteFactible_Bool') else "No"
        
        try:
            costo_val = float(costo_str) if costo_str not in ['N/A', 'Error', 'INFACTIBLE'] else costo_str
            tiempo_val = float(tiempo_str) if tiempo_str not in ['N/A', 'Error'] else tiempo_str
        except ValueError:
            costo_val = "ErrorConv"
            tiempo_val = "ErrorConv"

        costo_display = f"{costo_val:.2f}" if isinstance(costo_val, float) else costo_val
        tiempo_display = f"{tiempo_val:.4f}" if isinstance(tiempo_val, float) else tiempo_val
        
        fila_parts = [f"{caso:<15}", f"{str(aviones):>7}", f"{num_pistas:>6}"]
        for pk_csv, (_, width) in param_keys_to_show_dict.items(): # pk_csv es la clave del CSV
            param_val = parsed_params.get(pk_csv, 'N/A')
            if pk_csv == 'AspirationCriteria' and isinstance(param_val, bool):
                 param_display = 'Si' if param_val else 'No'
            else:
                 param_display = str(param_val)
            fila_parts.append(f"{param_display:>{width}}")
        
        fila_parts.extend([f"{costo_display:>10}", f"{tiempo_display:>11}", f"{factible_str:>8}"])
        print(" | ".join(fila_parts))
        
    print("-" * line_length)


if __name__ == '__main__':

    script_dir = os.path.dirname(__file__) # Directorio del script actual
    csv_file_path = os.path.join(script_dir, 'results', 'resultado.csv')

    if not os.path.exists(csv_file_path):
        print(f"Error CRÍTICO: El archivo '{csv_file_path}' no se encuentra.")
        print(f"Buscado en: {os.path.abspath(csv_file_path)}")
        print("Asegúrate de que el archivo existe en la ruta especificada y tiene datos.")
        # Intentar una ruta alternativa si la primera falla (común en algunos entornos de ejecución)
        alternative_path = os.path.join(os.getcwd(), 'results', 'resultado.csv')
        if os.path.exists(alternative_path):
            print(f"Intentando con ruta alternativa: {alternative_path}")
            csv_file_path = alternative_path
        else:
            print(f"Ruta alternativa también falló: {alternative_path}")
            exit()


    datos_por_algoritmo, fieldnames = procesar_csv(csv_file_path)

    if datos_por_algoritmo:
        # --- Tablas Greedy ---
        generar_tabla_gd(datos_por_algoritmo.get('GD', []), titulo="Greedy Determinista (GD)")
        generar_tabla_ge_solo(datos_por_algoritmo.get('GE_Solo', []))
        
        # --- Tablas GRASP ---
        # GRASP Determinista
        grasp_det_datos_dict = {'GRASP_HC_Det_0Restarts': datos_por_algoritmo.get('GRASP_HC_Det_0Restarts', [])}
        if grasp_det_datos_dict['GRASP_HC_Det_0Restarts']: # Solo generar si hay datos
            generar_tabla_grasp_con_restarts(grasp_det_datos_dict) 
        else:
            print("\nNo se encontraron datos para GRASP HC Determinista (0 Restarts).")


        # GRASP Estocástico (agrupando por número de restarts)
        grasp_estoc_variantes_datos = defaultdict(list)
        for algo_name, data_list in datos_por_algoritmo.items():
            if algo_name.startswith('GRASP_HC_Estoc_R'):
                restarts = get_grasp_restarts_from_algo_name(algo_name)
                if restarts > 0:
                     grasp_estoc_variantes_datos[restarts].extend(data_list)
        
        # Generar tabla para cada grupo de restarts
        for r_count, r_data_list in sorted(grasp_estoc_variantes_datos.items()):
            pass

        grasp_estoc_all_variants_dict = {
            algo_name: data
            for algo_name, data in datos_por_algoritmo.items()
            if algo_name.startswith('GRASP_HC_Estoc_R') and get_grasp_restarts_from_algo_name(algo_name) > 0
        }
        if grasp_estoc_all_variants_dict:
             generar_tabla_grasp_con_restarts(grasp_estoc_all_variants_dict)
        else:
             print("\nNo se encontraron datos para las variantes GRASP HC Estocástico.")


        # --- Tablas Tabu Search ---
        params_ts_det = { # Definición de columnas para TS Determinista
            'SolInicial': ('Inicio', 8), 
            'Ten': ('Ten', 5), 
            'MaxIter': ('IterTS', 7), # Clave CSV 'MaxIter', se muestra como 'IterTS'
            'MaxNoImp': ('NoImp', 8),
        }
        
        ts_gd_datos_all = []
        for algo, data_list in datos_por_algoritmo.items():
            if algo.startswith('TS_GD_'): # Asumiendo que todos los TS_GD son deterministas
                ts_gd_datos_all.extend(data_list) 

        if ts_gd_datos_all:
             generar_tabla_algoritmo_individual(ts_gd_datos_all, "Tabu Search Determinista (Inicio GD)", params_ts_det)
        else:
             print("\nNo se encontraron datos para Tabu Search Determinista (TS_GD).")

        ts_ge_datos_all = []
        for algo, data_list in datos_por_algoritmo.items():
             if algo.startswith('TS_GE_'): # Asumiendo que todos los TS_GE son estocásticos
                 ts_ge_datos_all.extend(data_list) 

        if ts_ge_datos_all:
            generar_tabla_ts_estoc(ts_ge_datos_all) # Esta función agrupa y resume
        else:
             print("\nNo se encontraron datos para Tabu Search Estocástico (TS_GE).")

        
        print("\nNotas Generales:")
        print("  - El número de 'Aviones' se basa en un mapeo predefinido en el script.")
        print("  - 'Costo' en las tablas se refiere a 'CostoPenalizadoSolucion' del CSV.")
        print("  - Para GE_Solo y GRASP/TS Estocástico, las estadísticas de costo (Mejor, Prom., Peor) son sobre soluciones estrictamente factibles.")
        print("  - (NF) junto a un costo indica que la solución no fue 'EsEstrictamenteFactible=True'.")

