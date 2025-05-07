# main.py
import os 
import csv
from datetime import datetime
import time 
import copy # Para deepcopy de soluciones GE

# Asegúrate que estas rutas sean correctas para tu estructura de carpetas
from scripts.lector_cases import read_case 
from scripts.greedy_deterministic import resolver as resolver_gd 
from scripts.greedy_stochastic import resolver as resolver_ge
from scripts.grasp_hc import grasp_resolver, evaluar_solucion_penalizada 
#QUITAMOS LA IMPORTACION DE HILL CLIMBING INDIVIDUAL PORQUE GRASP YA LO INCLUYE Y TS NO LO NECESITA DIRECTAMENTE
# from scripts.grasp_hc import hill_climbing_mejor_mejora 
from scripts.tabu_search import tabu_search_resolver # NUEVA IMPORTACIÓN PARA TABU SEARCH

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

# Configuraciones de restarts para GRASP cuando parte de soluciones GE
RESTARTS_GRASP_SOBRE_GE_CONFIGS = [10, 50, 100] 

# --- PARÁMETROS PARA TABU SEARCH ---
TABU_SEARCH_CONFIGS = [
    {'id_config': 'TS_Cfg1', 'tabu_tenure': 5,  'max_iterations_ts': 100, 'max_iter_no_improve_ts': 20},
    {'id_config': 'TS_Cfg2', 'tabu_tenure': 10, 'max_iterations_ts': 100, 'max_iter_no_improve_ts': 20},
    {'id_config': 'TS_Cfg3', 'tabu_tenure': 7,  'max_iterations_ts': 200, 'max_iter_no_improve_ts': 40},
    {'id_config': 'TS_Cfg4', 'tabu_tenure': 15, 'max_iterations_ts': 200, 'max_iter_no_improve_ts': 40},
    {'id_config': 'TS_Cfg5', 'tabu_tenure': 10, 'max_iterations_ts': 300, 'max_iter_no_improve_ts': 50},
]

# Penalizaciones
PENALIDAD_LK = 1000.0
PENALIDAD_SEP = 5000.0
PENALIDAD_NO_PROG = 100000.0


def escribir_resumen_solucion_csv(path_completo_csv, nombre_caso, algoritmo_nombre, num_pistas, detalles_parametros,
                                  eval_solucion, tiempo_comp, secuencia_aterrizajes_lista, aviones_no_programados_lista_ids):
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
    # nombres_base_casos = ['case1.txt'] # Para pruebas rápidas

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

            # 2. Greedy Estocástico (GE_Solo) - 10 ejecuciones
            print(f"\n    2. Ejecutando Greedy Estocástico (GE_Solo) - {NUM_EJECUCIONES_GE_SOLO} ejecuciones:")
            soluciones_ge_originales = []
            # Para estadísticas de GE_Solo en consola
            ge_solo_costos_base_factibles = [] 
            ge_solo_costos_penalizados_factibles = []
            ge_solo_count_factibles = 0

            for i_ge in range(NUM_EJECUCIONES_GE_SOLO):
                semilla_ge = i_ge
                tiempo_inicio_ge_solo = time.perf_counter()
                sol_ge_actual = resolver_ge(datos_del_caso, num_pistas_actual, semilla_ge, K_RCL_PARAM)
                tiempo_fin_ge_solo = time.perf_counter()
                tiempo_comp_ge_solo = tiempo_fin_ge_solo - tiempo_inicio_ge_solo
                
                if sol_ge_actual:
                    soluciones_ge_originales.append({'solucion': copy.deepcopy(sol_ge_actual), 'semilla_origen': semilla_ge})
                    eval_ge_solo = evaluar_solucion_penalizada(sol_ge_actual['secuencia_aterrizajes'], sol_ge_actual.get('aviones_no_programados', []), datos_del_caso, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG)
                    
                    if eval_ge_solo['es_estrictamente_factible']:
                        ge_solo_count_factibles += 1
                        ge_solo_costos_base_factibles.append(eval_ge_solo['costo_base'])
                        ge_solo_costos_penalizados_factibles.append(eval_ge_solo['costo_penalizado'])
                    
                    # print(f"      GE_Solo (sem {semilla_ge}): CostoBase={eval_ge_solo['costo_base']:.2f}, Penalizado={eval_ge_solo['costo_penalizado']:.2f}, Factible={eval_ge_solo['es_estrictamente_factible']} (T: {tiempo_comp_ge_solo:.4f}s)")
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


            # 3. GRASP con GD como inicio (0 restarts efectivos para GRASP, solo HC sobre GD)
            print("\n    3. Ejecutando GRASP (inicio GD, 0 restarts GRASP efectivos -> GD+HC):")
            if solucion_gd and solucion_gd.get('secuencia_aterrizajes'):
                algoritmo_nombre_grasp_gd = "GRASP_HC_Det_0Restarts"
                tiempo_inicio_grasp_gd = time.perf_counter()
                solucion_grasp_gd = grasp_resolver(
                    datos_del_caso, num_pistas_actual, 
                    num_iteraciones_grasp=1, 
                    semilla_inicial_grasp=0, 
                    parametro_rcl_ge=K_RCL_PARAM, 
                    max_iter_sin_mejora_hc=MAX_ITER_SIN_MEJORA_HC_GLOBAL, 
                    penalidad_lk=PENALIDAD_LK, penalidad_sep=PENALIDAD_SEP, penalidad_no_prog=PENALIDAD_NO_PROG,
                    solucion_inicial_para_primera_iter=solucion_gd
                )
                tiempo_fin_grasp_gd = time.perf_counter()
                tiempo_comp_grasp_gd = tiempo_fin_grasp_gd - tiempo_inicio_grasp_gd
                
                print(f"      {algoritmo_nombre_grasp_gd}: CostoBase={solucion_grasp_gd['costo_total']:.2f}, Penalizado={solucion_grasp_gd['costo_penalizado']:.2f}, Factible={solucion_grasp_gd['es_factible']} (T: {tiempo_comp_grasp_gd:.4f}s)")
                detalles_grasp_gd = f"Inicio:GD;IterGRASP:1;IterHC:{MAX_ITER_SIN_MEJORA_HC_GLOBAL}"
                escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, algoritmo_nombre_grasp_gd, num_pistas_actual, detalles_grasp_gd, 
                                            {'costo_base': solucion_grasp_gd['costo_total'], 
                                             'costo_penalizado': solucion_grasp_gd['costo_penalizado'],
                                             'es_estrictamente_factible': solucion_grasp_gd['es_factible'],
                                             'violaciones_lk_count': solucion_grasp_gd.get('violaciones_lk_count',0),
                                             'violaciones_sep_count': solucion_grasp_gd.get('violaciones_sep_count',0),
                                             'violaciones_no_prog_count': solucion_grasp_gd.get('violaciones_no_prog_count',0)}, 
                                            tiempo_comp_grasp_gd, solucion_grasp_gd['secuencia_aterrizajes'], solucion_grasp_gd.get('aviones_no_programados', []))
            else:
                print("      Skipping GRASP (inicio GD) porque GD no generó una secuencia válida.")

            # 4. GRASP con GE como inicio (diferentes números de restarts GRASP)
            print(f"\n    4. Ejecutando GRASP (inicio GE) con diferentes restarts GRASP:")
            if not soluciones_ge_originales:
                print("      Skipping GRASP (inicio GE) porque no hay soluciones GE válidas.")
            else:
                for restarts_cfg in RESTARTS_GRASP_SOBRE_GE_CONFIGS:
                    print(f"      Config GRASP: {restarts_cfg} restarts internos (sobre cada una de las {len(soluciones_ge_originales)} soluciones GE)")
                    
                    grasp_ge_costos_base_factibles = []
                    grasp_ge_costos_penalizados_factibles = []
                    grasp_ge_count_factibles = 0
                    
                    for idx_sol_ge, ge_data in enumerate(soluciones_ge_originales):
                        sol_ge_inicial = ge_data['solucion']
                        semilla_origen_ge = ge_data['semilla_origen']
                        algoritmo_nombre_grasp_ge = f"GRASP_HC_Estoc_R{restarts_cfg}_SGE{semilla_origen_ge}"
                        
                        tiempo_inicio_grasp_ge = time.perf_counter()
                        solucion_grasp_ge = grasp_resolver(
                            datos_del_caso, num_pistas_actual,
                            num_iteraciones_grasp=restarts_cfg,
                            semilla_inicial_grasp=semilla_origen_ge, 
                            parametro_rcl_ge=K_RCL_PARAM,
                            max_iter_sin_mejora_hc=MAX_ITER_SIN_MEJORA_HC_GLOBAL,
                            penalidad_lk=PENALIDAD_LK, penalidad_sep=PENALIDAD_SEP, penalidad_no_prog=PENALIDAD_NO_PROG,
                            solucion_inicial_para_primera_iter=sol_ge_inicial 
                        )
                        tiempo_fin_grasp_ge = time.perf_counter()
                        tiempo_comp_grasp_ge = tiempo_fin_grasp_ge - tiempo_inicio_grasp_ge

                        if solucion_grasp_ge['es_factible']:
                            grasp_ge_count_factibles +=1
                            grasp_ge_costos_base_factibles.append(solucion_grasp_ge['costo_total'])
                            grasp_ge_costos_penalizados_factibles.append(solucion_grasp_ge['costo_penalizado'])
                        
                        # print(f"        {algoritmo_nombre_grasp_ge}: CostoBase={solucion_grasp_ge['costo_total']:.2f}, Penalizado={solucion_grasp_ge['costo_penalizado']:.2f}, Factible={solucion_grasp_ge['es_factible']} (T: {tiempo_comp_grasp_ge:.4f}s)")
                        detalles_grasp_ge = f"Inicio:GE_sem{semilla_origen_ge};IterGRASP:{restarts_cfg};K_RCL:{K_RCL_PARAM};IterHC:{MAX_ITER_SIN_MEJORA_HC_GLOBAL}"
                        escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, algoritmo_nombre_grasp_ge, num_pistas_actual, detalles_grasp_ge,
                                                    {'costo_base': solucion_grasp_ge['costo_total'],
                                                     'costo_penalizado': solucion_grasp_ge['costo_penalizado'],
                                                     'es_estrictamente_factible': solucion_grasp_ge['es_factible'],
                                                     'violaciones_lk_count': solucion_grasp_ge.get('violaciones_lk_count',0),
                                                     'violaciones_sep_count': solucion_grasp_ge.get('violaciones_sep_count',0),
                                                     'violaciones_no_prog_count': solucion_grasp_ge.get('violaciones_no_prog_count',0)},
                                                    tiempo_comp_grasp_ge, solucion_grasp_ge['secuencia_aterrizajes'], solucion_grasp_ge.get('aviones_no_programados', []))
                    
                    if grasp_ge_count_factibles > 0:
                        print(f"        Resumen GRASP_R{restarts_cfg} (sobre {len(soluciones_ge_originales)} inicios GE, {grasp_ge_count_factibles} factibles post-GRASP):")
                        print(f"          Mejor Costo Base (factible): {min(grasp_ge_costos_base_factibles):.2f} (Penalizado: {min(grasp_ge_costos_penalizados_factibles):.2f})")
                        print(f"          Costo Base Promedio (factibles): {sum(grasp_ge_costos_base_factibles)/grasp_ge_count_factibles:.2f}")
                        print(f"          Peor Costo Base (factible): {max(grasp_ge_costos_base_factibles):.2f}")
                    else:
                        print(f"        Resumen GRASP_R{restarts_cfg}: No se obtuvieron soluciones estrictamente factibles tras aplicar GRASP a las {len(soluciones_ge_originales)} soluciones de GE.")


            # 5. Tabu Search sobre Solución de Greedy Determinista
            print("\n    5. Ejecutando Tabu Search sobre GD (TS_GD):")
            if solucion_gd and solucion_gd.get('secuencia_aterrizajes'):
                for ts_cfg in TABU_SEARCH_CONFIGS:
                    alg_nombre_ts_gd = f"TS_GD_{ts_cfg['id_config']}"
                    print(f"      Config TS: {ts_cfg['id_config']} (Tenure:{ts_cfg['tabu_tenure']}, MaxIter:{ts_cfg['max_iterations_ts']}, MaxNoImp:{ts_cfg['max_iter_no_improve_ts']})")
                    
                    tiempo_inicio_ts_gd = time.perf_counter()
                    solucion_ts_gd = tabu_search_resolver(
                        datos_del_caso, copy.deepcopy(solucion_gd), num_pistas_actual,
                        ts_cfg, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG
                    )
                    tiempo_fin_ts_gd = time.perf_counter()
                    tiempo_comp_ts_gd = tiempo_fin_ts_gd - tiempo_inicio_ts_gd

                    print(f"        {alg_nombre_ts_gd}: CostoBase={solucion_ts_gd['costo_total']:.2f}, Penalizado={solucion_ts_gd['costo_penalizado']:.2f}, Factible={solucion_ts_gd['es_factible']} (T: {tiempo_comp_ts_gd:.4f}s)")
                    detalles_ts_gd = f"Inicio:GD;CfgID:{ts_cfg['id_config']};Ten:{ts_cfg['tabu_tenure']};MaxIter:{ts_cfg['max_iterations_ts']};MaxNoImp:{ts_cfg['max_iter_no_improve_ts']}"
                    escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, alg_nombre_ts_gd, num_pistas_actual, detalles_ts_gd, 
                                                {'costo_base': solucion_ts_gd['costo_total'], 
                                                 'costo_penalizado': solucion_ts_gd['costo_penalizado'],
                                                 'es_estrictamente_factible': solucion_ts_gd['es_factible'],
                                                 'violaciones_lk_count': solucion_ts_gd.get('violaciones_lk_count',0),
                                                 'violaciones_sep_count': solucion_ts_gd.get('violaciones_sep_count',0),
                                                 'violaciones_no_prog_count': solucion_ts_gd.get('violaciones_no_prog_count',0)}, 
                                                tiempo_comp_ts_gd, solucion_ts_gd['secuencia_aterrizajes'], solucion_ts_gd.get('aviones_no_programados', []))
            else:
                print("      Skipping TS sobre GD porque GD no generó una secuencia válida.")

            # 6. Tabu Search sobre las 10 Soluciones de Greedy Estocástico
            print(f"\n    6. Ejecutando Tabu Search sobre las {len(soluciones_ge_originales)} soluciones de GE (TS_GE):")
            if not soluciones_ge_originales:
                print("      Skipping TS sobre GE porque no hay soluciones GE válidas.")
            else:
                for ts_cfg in TABU_SEARCH_CONFIGS:
                    print(f"      Config TS: {ts_cfg['id_config']} (Tenure:{ts_cfg['tabu_tenure']}, MaxIter:{ts_cfg['max_iterations_ts']}, MaxNoImp:{ts_cfg['max_iter_no_improve_ts']})")
                    
                    ts_ge_costos_base_factibles = []
                    ts_ge_costos_penalizados_factibles = []
                    ts_ge_count_factibles = 0
                    
                    for idx_sol_ge, ge_data in enumerate(soluciones_ge_originales):
                        sol_ge_inicial_ts = ge_data['solucion']
                        semilla_origen_ge_ts = ge_data['semilla_origen']
                        alg_nombre_ts_ge = f"TS_GE_{ts_cfg['id_config']}_SGE{semilla_origen_ge_ts}"
                        
                        if not sol_ge_inicial_ts or not sol_ge_inicial_ts.get('secuencia_aterrizajes'):
                            continue
                        
                        tiempo_inicio_ts_ge = time.perf_counter()
                        solucion_ts_ge = tabu_search_resolver(
                            datos_del_caso, copy.deepcopy(sol_ge_inicial_ts), num_pistas_actual,
                            ts_cfg, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG
                        )
                        tiempo_fin_ts_ge = time.perf_counter()
                        tiempo_comp_ts_ge = tiempo_fin_ts_ge - tiempo_inicio_ts_ge
                        
                        if solucion_ts_ge['es_factible']:
                            ts_ge_count_factibles +=1
                            ts_ge_costos_base_factibles.append(solucion_ts_ge['costo_total'])
                            ts_ge_costos_penalizados_factibles.append(solucion_ts_ge['costo_penalizado'])

                        # print(f"        {alg_nombre_ts_ge}: CostoBase={solucion_ts_ge['costo_total']:.2f}, Penalizado={solucion_ts_ge['costo_penalizado']:.2f}, Factible={solucion_ts_ge['es_factible']} (T: {tiempo_comp_ts_ge:.4f}s)")
                        detalles_ts_ge = f"Inicio:GE_Sem{semilla_origen_ge_ts};CfgID:{ts_cfg['id_config']};Ten:{ts_cfg['tabu_tenure']};MaxIter:{ts_cfg['max_iterations_ts']};MaxNoImp:{ts_cfg['max_iter_no_improve_ts']}"
                        escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, alg_nombre_ts_ge, num_pistas_actual, detalles_ts_ge,
                                                    {'costo_base': solucion_ts_ge['costo_total'],
                                                     'costo_penalizado': solucion_ts_ge['costo_penalizado'],
                                                     'es_estrictamente_factible': solucion_ts_ge['es_factible'],
                                                     'violaciones_lk_count': solucion_ts_ge.get('violaciones_lk_count',0),
                                                     'violaciones_sep_count': solucion_ts_ge.get('violaciones_sep_count',0),
                                                     'violaciones_no_prog_count': solucion_ts_ge.get('violaciones_no_prog_count',0)},
                                                    tiempo_comp_ts_ge, solucion_ts_ge['secuencia_aterrizajes'], solucion_ts_ge.get('aviones_no_programados', []))
                    
                    if ts_ge_count_factibles > 0:
                        print(f"        Resumen TS_Cfg {ts_cfg['id_config']} (sobre {len(soluciones_ge_originales)} inicios GE, {ts_ge_count_factibles} factibles post-TS):")
                        print(f"          Mejor Costo Base (factible): {min(ts_ge_costos_base_factibles):.2f} (Penalizado: {min(ts_ge_costos_penalizados_factibles):.2f})")
                        print(f"          Costo Base Promedio (factibles): {sum(ts_ge_costos_base_factibles)/ts_ge_count_factibles:.2f}")
                        print(f"          Peor Costo Base (factible): {max(ts_ge_costos_base_factibles):.2f}")
                    else:
                        print(f"        Resumen TS_Cfg {ts_cfg['id_config']}: No se obtuvieron soluciones estrictamente factibles tras aplicar TS a las {len(soluciones_ge_originales)} soluciones de GE.")

        else: 
            print(f"No se pudieron cargar los datos para {ruta_archivo_caso}.")
        print("----------------------------------\n")

if __name__ == '__main__':
    main()
