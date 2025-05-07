# main.py (Versión Paralelizada - Corregida)
import os 
import csv
from datetime import datetime
import time 
import copy 
import multiprocessing # Importar multiprocessing
import itertools # Necesario para starmap si se usa pool.map con starmapstar

# Asegúrate que estas rutas sean correctas para tu estructura de carpetas
from scripts.lector_cases import read_case 
from scripts.greedy_deterministic import resolver as resolver_gd 
from scripts.greedy_stochastic import resolver as resolver_ge
from scripts.grasp_hc import grasp_resolver, evaluar_solucion_penalizada 
from scripts.tabu_search import tabu_search_resolver 

# --- CONFIGURACIÓN PARA CSV RESUMIDO ---
CARPETA_RESULTADOS = "results" 
NOMBRE_BASE_CSV_RESUMEN = f"resumen_soluciones_aterrizajes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
PATH_COMPLETO_CSV_RESUMEN = "" # Se definirá en main()

CABECERA_CSV_RESUMEN = [
    'NombreCaso', 'Algoritmo', 'NumPistas', 'DetallesParametros', 
    'CostoBaseSolucion', 'CostoPenalizadoSolucion', 'EsEstrictamenteFactible',
    'ViolacionesLk', 'ViolacionesSep', 'ViolacionesNoProg',
    'TiempoComputacional_seg', 'OrdenAterrizajeIDs', 'AvionesNoProgramados'
]

# --- PARÁMETROS PARA LOS ALGORITMOS ---
NUM_EJECUCIONES_GE_SOLO = 10
K_RCL_PARAM = 3 
MAX_ITER_SIN_MEJORA_HC_GLOBAL = 20 # Usado en GRASP

RESTARTS_GRASP_SOBRE_GE_CONFIGS = [10, 25, 50] 

MAX_ITERATIONS_TS_FIJO = 200 
MAX_ITER_NO_IMPROVE_TS_FIJO = 40 
TABU_TENURES_A_PROBAR = [5, 7, 10, 15, 20] 

TABU_SEARCH_CONFIGS = []
for i, tenure in enumerate(TABU_TENURES_A_PROBAR):
    TABU_SEARCH_CONFIGS.append({
        'id_config': f'TS_Ten{tenure}', 
        'tabu_tenure': tenure, 
        'max_iterations_ts': MAX_ITERATIONS_TS_FIJO, 
        'max_iter_no_improve_ts': MAX_ITER_NO_IMPROVE_TS_FIJO
    })

PENALIDAD_LK = 1000.0
PENALIDAD_SEP = 5000.0
PENALIDAD_NO_PROG = 100000.0

# --- Funciones de Tarea para Paralelización ---

def ejecutar_ge_tarea(args):
    """Ejecuta una instancia de GE. Llamada con pool.map, recibe la tupla."""
    datos_del_caso, num_pistas_actual, semilla_ge, k_rcl = args # Desempaquetar aquí
    tiempo_inicio = time.perf_counter()
    solucion = resolver_ge(datos_del_caso, num_pistas_actual, semilla_ge, k_rcl)
    tiempo_fin = time.perf_counter()
    tiempo_comp = tiempo_fin - tiempo_inicio
    return {'solucion': solucion, 'tiempo': tiempo_comp, 'semilla_origen': semilla_ge} 

# CORREGIDO: Aceptar argumentos directamente para starmap
def ejecutar_grasp_tarea(datos_del_caso, num_pistas_actual, restarts_cfg, semilla_inicial, k_rcl, max_iter_hc, pen_lk, pen_sep, pen_no_prog, sol_inicial):
    """Ejecuta una instancia de GRASP. Llamada con pool.starmap."""
    tiempo_inicio = time.perf_counter()
    solucion = grasp_resolver(
        datos_del_caso, num_pistas_actual, restarts_cfg, semilla_inicial, k_rcl, 
        max_iter_hc, pen_lk, pen_sep, pen_no_prog, sol_inicial
    )
    tiempo_fin = time.perf_counter()
    tiempo_comp = tiempo_fin - tiempo_inicio
    return {'solucion': solucion, 'tiempo': tiempo_comp, 'restarts_cfg': restarts_cfg, 'semilla_origen': semilla_inicial if sol_inicial else None}

# CORREGIDO: Aceptar argumentos directamente para starmap
def ejecutar_ts_tarea(datos_del_caso, sol_inicial_ts, num_pistas_actual, ts_cfg, pen_lk, pen_sep, pen_no_prog, origen_inicial):
    """Ejecuta una instancia de Tabu Search. Llamada con pool.starmap."""
    tiempo_inicio = time.perf_counter()
    solucion = tabu_search_resolver(
        datos_del_caso, copy.deepcopy(sol_inicial_ts), num_pistas_actual,
        ts_cfg, pen_lk, pen_sep, pen_no_prog
    )
    tiempo_fin = time.perf_counter()
    tiempo_comp = tiempo_fin - tiempo_inicio
    return {'solucion': solucion, 'tiempo': tiempo_comp, 'ts_cfg': ts_cfg, 'origen_inicial': origen_inicial}


def escribir_resumen_solucion_csv(path_completo_csv, nombre_caso, algoritmo_nombre, num_pistas, detalles_parametros,
                                  eval_solucion, tiempo_comp, secuencia_aterrizajes_lista, aviones_no_programados_lista_ids):
    # (Sin cambios en esta función)
    costo_base_str = f"{eval_solucion['costo_base']:.2f}"
    costo_penalizado_str = f"{eval_solucion['costo_penalizado']:.2f}"
    es_factible_str = str(eval_solucion['es_estrictamente_factible'])
    
    orden_ids = [aterrizaje['avion_id'] for aterrizaje in secuencia_aterrizajes_lista]
    orden_ids_str = "-".join(map(str, orden_ids)) 
    aviones_no_programados_str = "-".join(map(str, aviones_no_programados_lista_ids)) if aviones_no_programados_lista_ids else ""

    fila_datos = [
        nombre_caso, algoritmo_nombre, num_pistas, detalles_parametros,
        costo_base_str, costo_penalizado_str, es_factible_str,
        eval_solucion['violaciones_lk_count'], eval_solucion['violaciones_sep_count'], eval_solucion['violaciones_no_prog_count'],
        f"{tiempo_comp:.4f}", orden_ids_str, aviones_no_programados_str
    ]
    escribir_cabecera = not os.path.exists(path_completo_csv)
    try:
        with open(path_completo_csv, 'a', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            if escribir_cabecera: writer.writerow(CABECERA_CSV_RESUMEN)
            writer.writerow(fila_datos)
    except IOError as e:
        print(f"      ERROR al escribir en CSV (Resumen): {e}")

def main():
    global PATH_COMPLETO_CSV_RESUMEN
    nombre_carpeta_casos = 'cases'
    nombres_base_casos = ['case1.txt', 'case2.txt', 'case3.txt', 'case4.txt']
    # nombres_base_casos = ['case1.txt'] 

    # --- Configuración de Paralelización ---
    num_workers = os.cpu_count() 
    print(f"Usando {num_workers} workers para paralelización.")

    if not os.path.exists(CARPETA_RESULTADOS):
        try:
            os.makedirs(CARPETA_RESULTADOS)
            print(f"Carpeta '{CARPETA_RESULTADOS}' creada.")
        except OSError as e:
            print(f"Error al crear la carpeta '{CARPETA_RESULTADOS}': {e}")
            return 
            
    PATH_COMPLETO_CSV_RESUMEN = os.path.join(CARPETA_RESULTADOS, NOMBRE_BASE_CSV_RESUMEN)
    if os.path.exists(PATH_COMPLETO_CSV_RESUMEN):
        try:
            os.remove(PATH_COMPLETO_CSV_RESUMEN)
            print(f"Archivo CSV resumen anterior '{PATH_COMPLETO_CSV_RESUMEN}' eliminado.")
        except OSError as e:
            print(f"Error al eliminar CSV resumen anterior: {e}")

    archivos_casos = [os.path.join(nombre_carpeta_casos, nombre) for nombre in nombres_base_casos]

    for ruta_archivo_caso in archivos_casos:
        nombre_base_del_caso = os.path.basename(ruta_archivo_caso) 
        print(f"\nProcesando: {ruta_archivo_caso} ")
        datos_del_caso = read_case(ruta_archivo_caso)

        if not datos_del_caso:
            print(f"No se pudieron cargar los datos para {ruta_archivo_caso}. Saltando.")
            continue
        
        num_aviones_total_caso = datos_del_caso['num_aviones']
        print(f"Número de aviones: {num_aviones_total_caso}")
            
        for num_pistas_actual in [1, 2]:
            print(f"\n  --- Para {num_pistas_actual} pista(s) ---")
            datos_del_caso['num_pistas_original'] = num_pistas_actual 

            # --- Ejecuciones Secuenciales (rápidas) ---
            # 1. Greedy Determinista (GD)
            print("\n    1. Ejecutando Greedy Determinista (GD):")
            algoritmo_nombre_gd = "GD" 
            tiempo_inicio_gd = time.perf_counter()
            solucion_gd = resolver_gd(datos_del_caso, num_pistas=num_pistas_actual)
            tiempo_fin_gd = time.perf_counter()
            tiempo_comp_gd = tiempo_fin_gd - tiempo_inicio_gd
            
            if solucion_gd:
                eval_gd = evaluar_solucion_penalizada(solucion_gd['secuencia_aterrizajes'], solucion_gd.get('aviones_no_programados', []), datos_del_caso, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG)
                print(f"      GD: CostoBase={eval_gd['costo_base']:.2f}, Penalizado={eval_gd['costo_penalizado']:.2f}, Factible={eval_gd['es_estrictamente_factible']} (T: {tiempo_comp_gd:.4f}s)")
                escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, algoritmo_nombre_gd, num_pistas_actual, "N/A", eval_gd, tiempo_comp_gd, solucion_gd['secuencia_aterrizajes'], solucion_gd.get('aviones_no_programados', []))
            else: print(f"      GD: No se obtuvo solución.")

            # --- Preparación para Paralelización ---
            # Crear el Pool DENTRO del bucle de pistas para que los datos del caso estén actualizados
            with multiprocessing.Pool(processes=num_workers) as pool:
            
                # 2. Greedy Estocástico (GE_Solo) - 10 ejecuciones (Paralelizado)
                print(f"\n    2. Ejecutando Greedy Estocástico (GE_Solo) - {NUM_EJECUCIONES_GE_SOLO} ejecuciones (Paralelo):")
                # Args para GE (cada elemento es una tupla, para pool.map)
                ge_args_list = [(datos_del_caso, num_pistas_actual, i, K_RCL_PARAM) for i in range(NUM_EJECUCIONES_GE_SOLO)]
                resultados_ge_paralelo = pool.map(ejecutar_ge_tarea, ge_args_list)
                
                soluciones_ge_originales = [] 
                ge_solo_costos_base_factibles = [] 
                ge_solo_costos_penalizados_factibles = []
                ge_solo_count_factibles = 0
                for res_ge in resultados_ge_paralelo:
                    sol_ge_actual = res_ge['solucion']
                    tiempo_comp_ge_solo = res_ge['tiempo']
                    semilla_ge = res_ge['semilla_origen']
                    if sol_ge_actual:
                        soluciones_ge_originales.append({'solucion': copy.deepcopy(sol_ge_actual), 'semilla_origen': semilla_ge})
                        eval_ge_solo = evaluar_solucion_penalizada(sol_ge_actual['secuencia_aterrizajes'], sol_ge_actual.get('aviones_no_programados', []), datos_del_caso, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG)
                        if eval_ge_solo['es_estrictamente_factible']:
                            ge_solo_count_factibles += 1
                            ge_solo_costos_base_factibles.append(eval_ge_solo['costo_base'])
                            ge_solo_costos_penalizados_factibles.append(eval_ge_solo['costo_penalizado'])
                        detalles_ge_solo = f"Semilla:{semilla_ge};K_RCL:{K_RCL_PARAM}"
                        escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, "GE_Solo", num_pistas_actual, detalles_ge_solo, eval_ge_solo, tiempo_comp_ge_solo, sol_ge_actual['secuencia_aterrizajes'], sol_ge_actual.get('aviones_no_programados', []))
                    else: print(f"      GE_Solo (sem {semilla_ge}): No se obtuvo solución.")
                
                print(f"      GE_Solo: {len(soluciones_ge_originales)} soluciones generadas.")
                if ge_solo_count_factibles > 0:
                    print(f"        Resultados GE_Solo ({ge_solo_count_factibles}/{NUM_EJECUCIONES_GE_SOLO} factibles):")
                    print(f"          Mejor Costo Base (factible): {min(ge_solo_costos_base_factibles):.2f} (Penalizado: {min(ge_solo_costos_penalizados_factibles):.2f})")
                    print(f"          Costo Base Promedio (factibles): {sum(ge_solo_costos_base_factibles)/ge_solo_count_factibles:.2f}")
                    print(f"          Peor Costo Base (factible): {max(ge_solo_costos_base_factibles):.2f}")
                else:
                    print(f"        Resultados GE_Solo: No se obtuvieron soluciones estrictamente factibles en {NUM_EJECUCIONES_GE_SOLO} ejecuciones.")

                # 3. GRASP con GD como inicio (Secuencial)
                print("\n    3. Ejecutando GRASP (inicio GD, 0 restarts GRASP efectivos -> GD+HC):")
                if solucion_gd and solucion_gd.get('secuencia_aterrizajes'):
                    algoritmo_nombre_grasp_gd = "GRASP_HC_Det_0Restarts"
                    tiempo_inicio_grasp_gd = time.perf_counter()
                    # Llamar a GRASP directamente, no necesita la función tarea aquí
                    solucion_grasp_gd = grasp_resolver(
                        datos_del_caso, num_pistas_actual, 
                        num_iteraciones_grasp=1, semilla_inicial_grasp=0, 
                        parametro_rcl_ge=K_RCL_PARAM, max_iter_sin_mejora_hc=MAX_ITER_SIN_MEJORA_HC_GLOBAL, 
                        penalidad_lk=PENALIDAD_LK, penalidad_sep=PENALIDAD_SEP, penalidad_no_prog=PENALIDAD_NO_PROG,
                        solucion_inicial_para_primera_iter=solucion_gd
                    )
                    tiempo_fin_grasp_gd = time.perf_counter()
                    tiempo_comp_grasp_gd = tiempo_fin_grasp_gd - tiempo_inicio_grasp_gd
                    print(f"      {algoritmo_nombre_grasp_gd}: CostoBase={solucion_grasp_gd['costo_total']:.2f}, Penalizado={solucion_grasp_gd['costo_penalizado']:.2f}, Factible={solucion_grasp_gd['es_factible']} (T: {tiempo_comp_grasp_gd:.4f}s)")
                    detalles_grasp_gd = f"Inicio:GD;IterGRASP:1;IterHC:{MAX_ITER_SIN_MEJORA_HC_GLOBAL}"
                    escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, algoritmo_nombre_grasp_gd, num_pistas_actual, detalles_grasp_gd, 
                                                {'costo_base': solucion_grasp_gd['costo_total'], 'costo_penalizado': solucion_grasp_gd['costo_penalizado'],'es_estrictamente_factible': solucion_grasp_gd['es_factible'],'violaciones_lk_count': solucion_grasp_gd.get('violaciones_lk_count',0),'violaciones_sep_count': solucion_grasp_gd.get('violaciones_sep_count',0),'violaciones_no_prog_count': solucion_grasp_gd.get('violaciones_no_prog_count',0)}, 
                                                tiempo_comp_grasp_gd, solucion_grasp_gd['secuencia_aterrizajes'], solucion_grasp_gd.get('aviones_no_programados', []))
                else: print("      Skipping GRASP (inicio GD) porque GD no generó una secuencia válida.")

                # 4. GRASP con GE como inicio (Paralelizado)
                print(f"\n    4. Ejecutando GRASP (inicio GE) con diferentes restarts GRASP (Paralelo):")
                if not soluciones_ge_originales:
                    print("      Skipping GRASP (inicio GE) porque no hay soluciones GE válidas.")
                else:
                    grasp_ge_args_list = []
                    for restarts_cfg in RESTARTS_GRASP_SOBRE_GE_CONFIGS:
                        for ge_data in soluciones_ge_originales:
                            # Crear tupla de argumentos para starmap
                            grasp_ge_args_list.append((
                                datos_del_caso, num_pistas_actual, restarts_cfg, 
                                ge_data['semilla_origen'], K_RCL_PARAM, MAX_ITER_SIN_MEJORA_HC_GLOBAL,
                                PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG, ge_data['solucion']
                            ))
                    
                    resultados_grasp_ge_paralelo = pool.starmap(ejecutar_grasp_tarea, grasp_ge_args_list)

                    # Procesar y resumir resultados
                    for restarts_cfg in RESTARTS_GRASP_SOBRE_GE_CONFIGS:
                        print(f"      Config GRASP: {restarts_cfg} restarts internos")
                        grasp_ge_costos_base_factibles = []
                        grasp_ge_costos_penalizados_factibles = []
                        grasp_ge_count_factibles = 0
                        for res_grasp in resultados_grasp_ge_paralelo:
                            if res_grasp['restarts_cfg'] == restarts_cfg:
                                solucion_grasp_ge = res_grasp['solucion']
                                tiempo_comp_grasp_ge = res_grasp['tiempo']
                                semilla_origen_ge = res_grasp['semilla_origen']
                                algoritmo_nombre_grasp_ge = f"GRASP_HC_Estoc_R{restarts_cfg}_SGE{semilla_origen_ge}"
                                
                                if solucion_grasp_ge['es_factible']:
                                    grasp_ge_count_factibles +=1
                                    grasp_ge_costos_base_factibles.append(solucion_grasp_ge['costo_total'])
                                    grasp_ge_costos_penalizados_factibles.append(solucion_grasp_ge['costo_penalizado'])
                                
                                detalles_grasp_ge = f"Inicio:GE_sem{semilla_origen_ge};IterGRASP:{restarts_cfg};K_RCL:{K_RCL_PARAM};IterHC:{MAX_ITER_SIN_MEJORA_HC_GLOBAL}"
                                escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, algoritmo_nombre_grasp_ge, num_pistas_actual, detalles_grasp_ge,
                                                            {'costo_base': solucion_grasp_ge['costo_total'],'costo_penalizado': solucion_grasp_ge['costo_penalizado'],'es_estrictamente_factible': solucion_grasp_ge['es_factible'],'violaciones_lk_count': solucion_grasp_ge.get('violaciones_lk_count',0),'violaciones_sep_count': solucion_grasp_ge.get('violaciones_sep_count',0),'violaciones_no_prog_count': solucion_grasp_ge.get('violaciones_no_prog_count',0)},
                                                            tiempo_comp_grasp_ge, solucion_grasp_ge['secuencia_aterrizajes'], solucion_grasp_ge.get('aviones_no_programados', []))
                        
                        if grasp_ge_count_factibles > 0:
                            print(f"        Resumen GRASP_R{restarts_cfg} (sobre {len(soluciones_ge_originales)} inicios GE, {grasp_ge_count_factibles} factibles post-GRASP):")
                            print(f"          Mejor Costo Base (factible): {min(grasp_ge_costos_base_factibles):.2f} (Penalizado: {min(grasp_ge_costos_penalizados_factibles):.2f})")
                            print(f"          Costo Base Promedio (factibles): {sum(grasp_ge_costos_base_factibles)/grasp_ge_count_factibles:.2f}")
                            print(f"          Peor Costo Base (factible): {max(grasp_ge_costos_base_factibles):.2f}")
                        else:
                            print(f"        Resumen GRASP_R{restarts_cfg}: No se obtuvieron soluciones estrictamente factibles.")

                # 5. Tabu Search sobre Solución de Greedy Determinista (Paralelizado)
                print("\n    5. Ejecutando Tabu Search sobre GD (TS_GD) (Paralelo):")
                if solucion_gd and solucion_gd.get('secuencia_aterrizajes'):
                    ts_gd_args_list = []
                    for ts_cfg in TABU_SEARCH_CONFIGS:
                        ts_gd_args_list.append((
                            datos_del_caso, solucion_gd, num_pistas_actual, ts_cfg, 
                            PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG, "GD"
                        ))
                    
                    resultados_ts_gd_paralelo = pool.starmap(ejecutar_ts_tarea, ts_gd_args_list)

                    for res_ts in resultados_ts_gd_paralelo:
                        solucion_ts_gd = res_ts['solucion']
                        tiempo_comp_ts_gd = res_ts['tiempo']
                        ts_cfg = res_ts['ts_cfg']
                        current_tenure = ts_cfg['tabu_tenure']
                        alg_nombre_ts_gd = f"TS_GD_Ten{current_tenure}"
                        print(f"      Config TS: Tenure={current_tenure} (MaxIter:{ts_cfg['max_iterations_ts']}, MaxNoImp:{ts_cfg['max_iter_no_improve_ts']})")
                        print(f"        {alg_nombre_ts_gd}: CostoBase={solucion_ts_gd['costo_total']:.2f}, Penalizado={solucion_ts_gd['costo_penalizado']:.2f}, Factible={solucion_ts_gd['es_factible']} (T: {tiempo_comp_ts_gd:.4f}s)")
                        detalles_ts_gd = f"Inicio:GD;Ten:{current_tenure};MaxIter:{ts_cfg['max_iterations_ts']};MaxNoImp:{ts_cfg['max_iter_no_improve_ts']}"
                        escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, alg_nombre_ts_gd, num_pistas_actual, detalles_ts_gd, 
                                                    {'costo_base': solucion_ts_gd['costo_total'], 'costo_penalizado': solucion_ts_gd['costo_penalizado'],'es_estrictamente_factible': solucion_ts_gd['es_factible'],'violaciones_lk_count': solucion_ts_gd.get('violaciones_lk_count',0),'violaciones_sep_count': solucion_ts_gd.get('violaciones_sep_count',0),'violaciones_no_prog_count': solucion_ts_gd.get('violaciones_no_prog_count',0)}, 
                                                    tiempo_comp_ts_gd, solucion_ts_gd['secuencia_aterrizajes'], solucion_ts_gd.get('aviones_no_programados', []))
                else: print("      Skipping TS sobre GD porque GD no generó una secuencia válida.")

                # 6. Tabu Search sobre las 10 Soluciones de Greedy Estocástico (Paralelizado)
                print(f"\n    6. Ejecutando Tabu Search sobre las {len(soluciones_ge_originales)} soluciones de GE (TS_GE) (Paralelo):")
                if not soluciones_ge_originales:
                    print("      Skipping TS sobre GE porque no hay soluciones GE válidas.")
                else:
                    ts_ge_args_list = []
                    for ts_cfg in TABU_SEARCH_CONFIGS:
                        for ge_data in soluciones_ge_originales:
                            if ge_data['solucion'] and ge_data['solucion'].get('secuencia_aterrizajes'):
                                ts_ge_args_list.append((
                                    datos_del_caso, ge_data['solucion'], num_pistas_actual, ts_cfg,
                                    PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG, f"GE_Sem{ge_data['semilla_origen']}"
                                ))
                    
                    resultados_ts_ge_paralelo = pool.starmap(ejecutar_ts_tarea, ts_ge_args_list)

                    # Procesar y resumir resultados
                    for ts_cfg in TABU_SEARCH_CONFIGS:
                        current_tenure = ts_cfg['tabu_tenure']
                        print(f"      Config TS: Tenure={current_tenure} (MaxIter:{ts_cfg['max_iterations_ts']}, MaxNoImp:{ts_cfg['max_iter_no_improve_ts']})")
                        ts_ge_costos_base_factibles = []
                        ts_ge_costos_penalizados_factibles = []
                        ts_ge_count_factibles = 0
                        
                        for res_ts in resultados_ts_ge_paralelo:
                            if res_ts['ts_cfg']['id_config'] == ts_cfg['id_config']: 
                                solucion_ts_ge = res_ts['solucion']
                                tiempo_comp_ts_ge = res_ts['tiempo']
                                origen_info = res_ts['origen_inicial'] 
                                semilla_origen_ge_ts = int(origen_info.split("Sem")[1]) 
                                alg_nombre_ts_ge = f"TS_GE_Ten{current_tenure}_SGE{semilla_origen_ge_ts}" 

                                if solucion_ts_ge['es_factible']:
                                    ts_ge_count_factibles +=1
                                    ts_ge_costos_base_factibles.append(solucion_ts_ge['costo_total'])
                                    ts_ge_costos_penalizados_factibles.append(solucion_ts_ge['costo_penalizado'])

                                detalles_ts_ge = f"Inicio:{origen_info};Ten:{current_tenure};MaxIter:{ts_cfg['max_iterations_ts']};MaxNoImp:{ts_cfg['max_iter_no_improve_ts']}"
                                escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, alg_nombre_ts_ge, num_pistas_actual, detalles_ts_ge,
                                                            {'costo_base': solucion_ts_ge['costo_total'],'costo_penalizado': solucion_ts_ge['costo_penalizado'],'es_estrictamente_factible': solucion_ts_ge['es_factible'],'violaciones_lk_count': solucion_ts_ge.get('violaciones_lk_count',0),'violaciones_sep_count': solucion_ts_ge.get('violaciones_sep_count',0),'violaciones_no_prog_count': solucion_ts_ge.get('violaciones_no_prog_count',0)},
                                                            tiempo_comp_ts_ge, solucion_ts_ge['secuencia_aterrizajes'], solucion_ts_ge.get('aviones_no_programados', []))
                        
                        if ts_ge_count_factibles > 0:
                            print(f"        Resumen TS_Ten{current_tenure} (sobre {len(soluciones_ge_originales)} inicios GE, {ts_ge_count_factibles} factibles post-TS):")
                            print(f"          Mejor Costo Base (factible): {min(ts_ge_costos_base_factibles):.2f} (Penalizado: {min(ts_ge_costos_penalizados_factibles):.2f})")
                            print(f"          Costo Base Promedio (factibles): {sum(ts_ge_costos_base_factibles)/ts_ge_count_factibles:.2f}")
                            print(f"          Peor Costo Base (factible): {max(ts_ge_costos_base_factibles):.2f}")
                        else:
                            print(f"        Resumen TS_Ten{current_tenure}: No se obtuvieron soluciones estrictamente factibles.")

            # Cerrar el pool al final del bucle de pistas
            # pool.close() # No es necesario con 'with'
            # pool.join() # No es necesario con 'with'

        print("----------------------------------\n") # Separador entre casos

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    main()
