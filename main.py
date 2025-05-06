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
from scripts.grasp_hc import grasp_resolver, hill_climbing_mejor_mejora, evaluar_solucion_penalizada
# from scripts.verificador import verificar_solucion # Mantengo esto comentado como lo tienes

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
MAX_ITER_SIN_MEJORA_HC_GLOBAL = 20 

# Configuraciones de restarts para GRASP cuando parte de soluciones GE
RESTARTS_GRASP_SOBRE_GE_CONFIGS = [10, 50, 100] 

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
                eval_gd = evaluar_solucion_penalizada(solucion_gd['secuencia_aterrizajes'], solucion_gd['aviones_no_programados'], datos_del_caso, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG)
                print(f"      GD: CostoBase={eval_gd['costo_base']:.2f}, Penalizado={eval_gd['costo_penalizado']:.2f}, Factible={eval_gd['es_estrictamente_factible']} (T: {tiempo_comp_gd:.4f}s)")
                escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, algoritmo_nombre_gd, num_pistas_actual, "N/A", eval_gd, tiempo_comp_gd, solucion_gd['secuencia_aterrizajes'], solucion_gd['aviones_no_programados'])
            else: print(f"      GD: No se obtuvo solución.")

            # 2. Greedy Estocástico (GE_Solo) - 10 ejecuciones
            print(f"\n    2. Ejecutando Greedy Estocástico (GE_Solo) - {NUM_EJECUCIONES_GE_SOLO} ejecuciones:")
            soluciones_ge_originales = []
            for i_ge in range(NUM_EJECUCIONES_GE_SOLO):
                semilla_ge = i_ge
                tiempo_inicio_ge_solo = time.perf_counter()
                sol_ge_actual = resolver_ge(datos_del_caso, num_pistas_actual, semilla_ge, K_RCL_PARAM)
                tiempo_fin_ge_solo = time.perf_counter()
                tiempo_comp_ge_solo = tiempo_fin_ge_solo - tiempo_inicio_ge_solo
                
                if sol_ge_actual:
                    soluciones_ge_originales.append({'solucion': copy.deepcopy(sol_ge_actual), 'semilla_origen': semilla_ge})
                    eval_ge_solo = evaluar_solucion_penalizada(sol_ge_actual['secuencia_aterrizajes'], sol_ge_actual['aviones_no_programados'], datos_del_caso, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG)
                    # print(f"      GE_Solo (sem {semilla_ge}): CostoBase={eval_ge_solo['costo_base']:.2f}, Penalizado={eval_ge_solo['costo_penalizado']:.2f}, Factible={eval_ge_solo['es_estrictamente_factible']} (T: {tiempo_comp_ge_solo:.4f}s)")
                    detalles_ge_solo = f"Semilla:{semilla_ge};K_RCL:{K_RCL_PARAM}"
                    escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, "GE_Solo", num_pistas_actual, detalles_ge_solo, eval_ge_solo, tiempo_comp_ge_solo, sol_ge_actual['secuencia_aterrizajes'], sol_ge_actual['aviones_no_programados'])
                else: print(f"      GE_Solo (sem {semilla_ge}): No se obtuvo solución.")
            print(f"      GE_Solo: {len(soluciones_ge_originales)} soluciones generadas.")

            # 3. GRASP con GD como inicio (0 restarts efectivos para GRASP, solo HC sobre GD)
            print("\n    3. Ejecutando GRASP (inicio GD, 0 restarts GRASP efectivos -> GD+HC):")
            if solucion_gd and solucion_gd.get('secuencia_aterrizajes'):
                algoritmo_nombre_grasp_gd = "GRASP_HC_Det_0Restarts"
                tiempo_inicio_grasp_gd = time.perf_counter()
                # Llamamos a grasp_resolver con num_iteraciones_grasp=1 y la solución GD como inicial
                solucion_grasp_gd = grasp_resolver(
                    datos_del_caso, num_pistas_actual, 
                    num_iteraciones_grasp=1, # Solo procesa la solución inicial con HC
                    semilla_inicial_grasp=0, # No relevante si num_iteraciones_grasp=1 y se da sol_inicial
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
                    print(f"      Config GRASP: {restarts_cfg} restarts internos (para cada una de las {len(soluciones_ge_originales)} soluciones GE)")
                    algoritmo_nombre_grasp_ge = f"GRASP_HC_Estoc_{restarts_cfg}Restarts"
                    
                    for idx_sol_ge, ge_data in enumerate(soluciones_ge_originales):
                        sol_ge_inicial = ge_data['solucion']
                        semilla_origen_ge = ge_data['semilla_origen']
                        # print(f"        Procesando con GRASP (inicio GE semilla {semilla_origen_ge}), {restarts_cfg} restarts GRASP...")

                        tiempo_inicio_grasp_ge = time.perf_counter()
                        solucion_grasp_ge = grasp_resolver(
                            datos_del_caso, num_pistas_actual,
                            num_iteraciones_grasp=restarts_cfg,
                            semilla_inicial_grasp=semilla_origen_ge, # Usar semilla de GE para diversificar inicios de GRASP
                            parametro_rcl_ge=K_RCL_PARAM,
                            max_iter_sin_mejora_hc=MAX_ITER_SIN_MEJORA_HC_GLOBAL,
                            penalidad_lk=PENALIDAD_LK, penalidad_sep=PENALIDAD_SEP, penalidad_no_prog=PENALIDAD_NO_PROG,
                            solucion_inicial_para_primera_iter=sol_ge_inicial # Pasar la solución GE
                        )
                        tiempo_fin_grasp_ge = time.perf_counter()
                        tiempo_comp_grasp_ge = tiempo_fin_grasp_ge - tiempo_inicio_grasp_ge

                        # print(f"          Resultado: CostoBase={solucion_grasp_ge['costo_total']:.2f}, Penalizado={solucion_grasp_ge['costo_penalizado']:.2f}, Factible={solucion_grasp_ge['es_factible']} (T: {tiempo_comp_grasp_ge:.4f}s)")
                        detalles_grasp_ge = f"Inicio:GE_sem{semilla_origen_ge};IterGRASP:{restarts_cfg};K_RCL:{K_RCL_PARAM};IterHC:{MAX_ITER_SIN_MEJORA_HC_GLOBAL}"
                        escribir_resumen_solucion_csv(PATH_COMPLETO_CSV_RESUMEN, nombre_base_del_caso, algoritmo_nombre_grasp_ge, num_pistas_actual, detalles_grasp_ge,
                                                    {'costo_base': solucion_grasp_ge['costo_total'],
                                                     'costo_penalizado': solucion_grasp_ge['costo_penalizado'],
                                                     'es_estrictamente_factible': solucion_grasp_ge['es_factible'],
                                                     'violaciones_lk_count': solucion_grasp_ge.get('violaciones_lk_count',0),
                                                     'violaciones_sep_count': solucion_grasp_ge.get('violaciones_sep_count',0),
                                                     'violaciones_no_prog_count': solucion_grasp_ge.get('violaciones_no_prog_count',0)},
                                                    tiempo_comp_grasp_ge, solucion_grasp_ge['secuencia_aterrizajes'], solucion_grasp_ge.get('aviones_no_programados', []))
        else: 
            print(f"No se pudieron cargar los datos para {ruta_archivo_caso}.")
        print("----------------------------------\n")

if __name__ == '__main__':
    main()
