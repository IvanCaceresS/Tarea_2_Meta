# main.py
import os 
import csv
from datetime import datetime
import time 

# Asegúrate que estas rutas sean correctas para tu estructura de carpetas
from scripts.lector_cases import read_case 
from scripts.greedy_deterministic import resolver as resolver_gd 
from scripts.greedy_stochastic import resolver as resolver_ge
from scripts.grasp_hc import grasp_resolver, hill_climbing_mejor_mejora, evaluar_solucion_penalizada
# from scripts.verificador import verificar_solucion # Mantengo esto comentado como lo tienes

# --- CONFIGURACIÓN PARA CSV RESUMIDO ---
CARPETA_RESULTADOS = "results" 
NOMBRE_BASE_CSV_RESUMEN = f"resumen_soluciones_aterrizajes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# El path completo se construirá después de asegurar que la carpeta exista
# PATH_COMPLETO_CSV_RESUMEN se definirá en main()

CABECERA_CSV_RESUMEN = [
    'NombreCaso', 'Algoritmo', 'NumPistas', 'DetallesParametros', 
    'CostoBaseSolucion', 'CostoPenalizadoSolucion', 'EsEstrictamenteFactible',
    'ViolacionesLk', 'ViolacionesSep', 'ViolacionesNoProg',
    'TiempoComputacional_seg', 'OrdenAterrizajeIDs', 'AvionesNoProgramados'
]

# --- PARÁMETROS PARA LOS ALGORITMOS ---
# Para Greedy Estocástico (GE)
NUM_EJECUCIONES_GE = 10
K_RCL_GE = 3 # Usado también como parametro_rcl_ge para la construcción en GRASP

# Para GRASP + Hill Climbing
ITERACIONES_GRASP_CONFIGS = [10, 25, 50]  # Diferentes cantidades de reinicio (iteraciones GRASP)
SEMILLA_INICIAL_GRASP = 0 # Semilla base para las iteraciones de GRASP
MAX_ITER_SIN_MEJORA_HC = 20 # Límite de iteraciones sin mejora para Hill Climbing

# Penalizaciones (ajustar según experimentación)
PENALIDAD_LK = 1000.0
PENALIDAD_SEP = 5000.0
PENALIDAD_NO_PROG = 100000.0


def escribir_resumen_solucion_csv(path_completo_csv, nombre_caso, algoritmo_nombre, num_pistas, detalles_parametros,
                                  eval_solucion, tiempo_comp, secuencia_aterrizajes_lista, aviones_no_programados_lista_ids):
    """
    Escribe una línea de resumen de la solución en el archivo CSV especificado,
    utilizando la evaluación precalculada.
    """
    costo_base_str = f"{eval_solucion['costo_base']:.2f}"
    costo_penalizado_str = f"{eval_solucion['costo_penalizado']:.2f}"
    es_factible_str = str(eval_solucion['es_estrictamente_factible'])
    
    orden_ids = [aterrizaje['avion_id'] for aterrizaje in secuencia_aterrizajes_lista]
    orden_ids_str = "-".join(map(str, orden_ids)) 

    aviones_no_programados_str = "-".join(map(str, aviones_no_programados_lista_ids)) if aviones_no_programados_lista_ids else ""

    fila_datos = [
        nombre_caso,
        algoritmo_nombre,
        num_pistas,
        detalles_parametros,
        costo_base_str,
        costo_penalizado_str,
        es_factible_str,
        eval_solucion['violaciones_lk_count'],
        eval_solucion['violaciones_sep_count'],
        eval_solucion['violaciones_no_prog_count'],
        f"{tiempo_comp:.4f}", 
        orden_ids_str,
        aviones_no_programados_str
    ]

    escribir_cabecera = not os.path.exists(path_completo_csv)
    
    try:
        with open(path_completo_csv, 'a', newline='', encoding='utf-8') as f_csv:
            writer = csv.writer(f_csv)
            if escribir_cabecera:
                writer.writerow(CABECERA_CSV_RESUMEN)
            writer.writerow(fila_datos)
    except IOError as e:
        print(f"      ERROR al escribir en CSV (Resumen): {e}")


# --- (La función original escribir_resultados_csv detallada permanece comentada) ---
# NOMBRE_ARCHIVO_CSV_DETALLADO = f"resultados_aterrizajes_DETALLADO_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# CABECERA_CSV_DETALLADO = [
#     'NombreCaso', 'Algoritmo', 'NumPistas', 'SemillaGE', 
#     'CostoTotalSolucion', 'AvionID', 'TiempoAterrizajeProgramado', 
#     'PistaAsignada', 'CostoIndividualAvion',
#     'Ek_Avion', 'Pk_Avion', 'Lk_Avion'
# ]
# def escribir_resultados_csv_detallado(nombre_caso, algoritmo_nombre, num_pistas, semilla_ge, solucion, datos_entrada_aviones): # Nombre original de tu función detallada
#     """
#     Escribe los detalles de una solución en el archivo CSV.
#     """
#     if not solucion or not solucion.get('secuencia_aterrizajes'):
#         print(f"           ADVERTENCIA CSV: No hay secuencia de aterrizajes para {nombre_caso}, {algoritmo_nombre}, {num_pistas}p, semilla {semilla_ge}.")
#         return

#     costo_total_solucion = solucion['costo_total'] # Esto sería el costo base
    
#     # Crear un diccionario para acceder fácilmente a E_k, P_k, L_k por id
#     info_aviones_dict = {avion['id']: avion for avion in datos_entrada_aviones}

#     filas_para_escribir = []
#     for aterrizaje in solucion['secuencia_aterrizajes']:
#         avion_id = aterrizaje['avion_id']
#         avion_original_info = info_aviones_dict.get(avion_id, {}) 

#         fila = [
#             nombre_caso,
#             algoritmo_nombre,
#             num_pistas,
#             semilla_ge if algoritmo_nombre == 'GE' else 'N/A', # Semilla solo para GE
#             f"{costo_total_solucion:.2f}", 
#             avion_id,
#             aterrizaje['tiempo'],
#             aterrizaje['pista'],
#             f"{aterrizaje.get('costo_individual', 0.0):.2f}", 
#             avion_original_info.get('E', ''),
#             avion_original_info.get('P', ''),
#             avion_original_info.get('L', '')
#         ]
#         filas_para_escribir.append(fila)

#     escribir_cabecera = not os.path.exists(NOMBRE_ARCHIVO_CSV_DETALLADO) 
    
#     try:
#         with open(NOMBRE_ARCHIVO_CSV_DETALLADO, 'a', newline='', encoding='utf-8') as f_csv: 
#             writer = csv.writer(f_csv)
#             if escribir_cabecera:
#                 writer.writerow(CABECERA_CSV_DETALLADO) 
#             writer.writerows(filas_para_escribir)
#     except IOError as e:
#         print(f"           ERROR al escribir en CSV: {e}")


def main():
    nombre_carpeta_casos = 'cases'
    nombres_base_casos = ['case1.txt', 'case2.txt', 'case3.txt', 'case4.txt']
    # Para una prueba más detallada:
    # nombres_base_casos = ['case1.txt'] 

    # --- CREAR CARPETA DE RESULTADOS SI NO EXISTE ---
    if not os.path.exists(CARPETA_RESULTADOS):
        try:
            os.makedirs(CARPETA_RESULTADOS)
            print(f"Carpeta '{CARPETA_RESULTADOS}' creada.")
        except OSError as e:
            print(f"Error al crear la carpeta '{CARPETA_RESULTADOS}': {e}")
            return 
            
    path_completo_csv_resumen = os.path.join(CARPETA_RESULTADOS, NOMBRE_BASE_CSV_RESUMEN)

    # Opcional: Para empezar con un CSV resumen limpio cada vez que ejecutas main.py
    if os.path.exists(path_completo_csv_resumen):
        try:
            os.remove(path_completo_csv_resumen)
            print(f"Archivo CSV resumen anterior '{path_completo_csv_resumen}' eliminado.")
        except OSError as e:
            print(f"Error al eliminar CSV resumen anterior: {e}")


    archivos_casos = [os.path.join(nombre_carpeta_casos, nombre) for nombre in nombres_base_casos]

    for ruta_archivo_caso in archivos_casos:
        nombre_base_del_caso = os.path.basename(ruta_archivo_caso) 
        print(f"\nProcesando: {ruta_archivo_caso} ")
        datos_del_caso = read_case(ruta_archivo_caso)

        if datos_del_caso:
            num_aviones_total_caso = datos_del_caso['num_aviones']
            print(f"Número de aviones: {num_aviones_total_caso}")
            
            for num_pistas_actual in [1, 2]:
                print(f"\n  --- Para {num_pistas_actual} pista(s) ---")
                datos_del_caso['num_pistas_original'] = num_pistas_actual # Para que evaluar_solucion sepa las pistas

                # --- Greedy Determinista (GD) ---
                print("\n    Ejecutando Greedy Determinista (GD):")
                algoritmo_nombre_gd = "GD" 
                
                tiempo_inicio_gd = time.perf_counter()
                solucion_gd = resolver_gd(datos_del_caso, num_pistas=num_pistas_actual)
                tiempo_fin_gd = time.perf_counter()
                tiempo_comp_gd = tiempo_fin_gd - tiempo_inicio_gd
                
                if solucion_gd:
                    eval_gd = evaluar_solucion_penalizada(
                        solucion_gd['secuencia_aterrizajes'], 
                        solucion_gd['aviones_no_programados'], 
                        datos_del_caso, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG
                    )
                    print(f"      Costo Base GD: {eval_gd['costo_base']:.2f}, Penalizado: {eval_gd['costo_penalizado']:.2f}, Factible: {eval_gd['es_estrictamente_factible']} (Tiempo: {tiempo_comp_gd:.4f}s)")
                    if eval_gd['violaciones_no_prog_count'] > 0:
                        print(f"      Aviones no programados GD: {solucion_gd['aviones_no_programados']}")
                    
                    escribir_resumen_solucion_csv(
                        path_completo_csv_resumen, nombre_base_del_caso, algoritmo_nombre_gd, num_pistas_actual, 
                        "N/A", eval_gd, tiempo_comp_gd, 
                        solucion_gd['secuencia_aterrizajes'], solucion_gd['aviones_no_programados']
                    )
                else: # No debería ocurrir si el resolver_gd siempre devuelve un dict
                    print(f"      No se obtuvo estructura de solución GD para {num_pistas_actual} pista(s).")
                    # Podrías escribir una línea de error si lo deseas

                # --- Hill Climbing sobre Greedy Determinista (GD+HC) ---
                if solucion_gd and solucion_gd['secuencia_aterrizajes']: # Solo si GD produjo algo
                    print("\n    Ejecutando Hill Climbing sobre GD (GD+HC):")
                    algoritmo_nombre_gd_hc = "GD_HC"
                    
                    tiempo_inicio_gd_hc = time.perf_counter()
                    solucion_gd_hc = hill_climbing_mejor_mejora(
                        solucion_gd, # Pasamos la solución completa del GD
                        datos_del_caso,
                        num_pistas_actual,
                        MAX_ITER_SIN_MEJORA_HC,
                        PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG
                    )
                    tiempo_fin_gd_hc = time.perf_counter()
                    tiempo_comp_gd_hc = tiempo_fin_gd_hc - tiempo_inicio_gd_hc

                    # La solución de HC ya viene con 'costo_total' (base), 'costo_penalizado', 'es_factible' (estricta)
                    # y los contadores de violaciones a través de la evaluación interna que hace.
                    # No es necesario llamar a evaluar_solucion_penalizada de nuevo externamente si HC ya lo hace.
                    # El 'eval_actual' devuelto por HC es lo que necesitamos.
                    print(f"      Costo Base GD+HC: {solucion_gd_hc['costo_total']:.2f}, Penalizado: {solucion_gd_hc['costo_penalizado']:.2f}, Factible: {solucion_gd_hc['es_factible']} (Tiempo: {tiempo_comp_gd_hc:.4f}s)")
                    
                    detalles_gd_hc = f"IterHC:{MAX_ITER_SIN_MEJORA_HC}"
                    escribir_resumen_solucion_csv(
                        path_completo_csv_resumen, nombre_base_del_caso, algoritmo_nombre_gd_hc, num_pistas_actual,
                        detalles_gd_hc, 
                        {'costo_base': solucion_gd_hc['costo_total'], # Adaptar para la función de escritura
                         'costo_penalizado': solucion_gd_hc['costo_penalizado'],
                         'es_estrictamente_factible': solucion_gd_hc['es_factible'],
                         'violaciones_lk_count': solucion_gd_hc.get('violaciones_lk_count',0), # HC debería devolver esto
                         'violaciones_sep_count': solucion_gd_hc.get('violaciones_sep_count',0),
                         'violaciones_no_prog_count': solucion_gd_hc.get('violaciones_no_prog_count',0)
                        },
                        tiempo_comp_gd_hc,
                        solucion_gd_hc['secuencia_aterrizajes'], solucion_gd_hc['aviones_no_programados']
                    )
                else:
                    print("      Skipping GD+HC porque GD no generó una secuencia válida.")


                # --- Greedy Estocástico (GE) ---
                print("\n    Ejecutando Greedy Estocástico (GE):")
                algoritmo_nombre_ge = "GE" 
                
                mejores_costos_ge = {'base': float('inf'), 'penalizado': float('inf')}
                sum_costos_ge_factibles = {'base': 0, 'penalizado': 0}
                count_ge_factibles = 0

                for i_ge in range(NUM_EJECUCIONES_GE):
                    semilla_actual_ge = i_ge # O alguna otra estrategia de semillas
                    
                    tiempo_inicio_ge = time.perf_counter()
                    sol_ge_actual = resolver_ge(
                        datos_del_caso, 
                        num_pistas=num_pistas_actual, 
                        semilla=semilla_actual_ge, 
                        parametro_rcl_alpha=K_RCL_GE
                    ) 
                    tiempo_fin_ge = time.perf_counter()
                    tiempo_comp_ge = tiempo_fin_ge - tiempo_inicio_ge
                    
                    if sol_ge_actual:
                        eval_ge = evaluar_solucion_penalizada(
                            sol_ge_actual['secuencia_aterrizajes'], 
                            sol_ge_actual['aviones_no_programados'], 
                            datos_del_caso, PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG
                        )
                        
                        # Para estadísticas internas de GE
                        if eval_ge['es_estrictamente_factible']:
                            count_ge_factibles +=1
                            sum_costos_ge_factibles['base'] += eval_ge['costo_base']
                            sum_costos_ge_factibles['penalizado'] += eval_ge['costo_penalizado']
                            if eval_ge['costo_base'] < mejores_costos_ge['base']: # O podrías basar "mejor" en penalizado
                                mejores_costos_ge['base'] = eval_ge['costo_base']
                                mejores_costos_ge['penalizado'] = eval_ge['costo_penalizado']

                        detalles_ge = f"Semilla:{semilla_actual_ge};K_RCL:{K_RCL_GE}"
                        escribir_resumen_solucion_csv(
                            path_completo_csv_resumen, nombre_base_del_caso, algoritmo_nombre_ge, num_pistas_actual,
                            detalles_ge, eval_ge, tiempo_comp_ge,
                            sol_ge_actual['secuencia_aterrizajes'], sol_ge_actual['aviones_no_programados']
                        )
                    else: # No debería ocurrir
                        print(f"      Ejecución GE (semilla {semilla_actual_ge}): No se obtuvo estructura de solución.")
                
                if count_ge_factibles > 0:
                    print(f"      Resultados GE ({num_pistas_actual} pista(s), {NUM_EJECUCIONES_GE} ejec, {count_ge_factibles} factibles):")
                    print(f"        Mejor Costo Base (factible): {mejores_costos_ge['base']:.2f} (Penalizado: {mejores_costos_ge['penalizado']:.2f})")
                    print(f"        Costo Base Promedio (factibles): {sum_costos_ge_factibles['base']/count_ge_factibles:.2f}")
                else:
                    print(f"      Resultados GE ({num_pistas_actual} pista(s)): No se obtuvieron soluciones estrictamente factibles en {NUM_EJECUCIONES_GE} ejecuciones.")

                # --- GRASP + Hill Climbing (Estocástico) ---
                print("\n    Ejecutando GRASP + Hill Climbing (GRASP_HC_Estoc):")
                algoritmo_nombre_grasp = "GRASP_HC_Estoc"

                for iter_grasp_cfg in ITERACIONES_GRASP_CONFIGS:
                    print(f"      Config GRASP: {iter_grasp_cfg} iteraciones (restarts)")
                    tiempo_inicio_grasp = time.perf_counter()
                    solucion_grasp = grasp_resolver(
                        datos_del_caso,
                        num_pistas_actual,
                        iter_grasp_cfg, # Número de iteraciones GRASP (restarts)
                        SEMILLA_INICIAL_GRASP,
                        K_RCL_GE, # Parámetro RCL para la construcción GE interna
                        MAX_ITER_SIN_MEJORA_HC,
                        PENALIDAD_LK, PENALIDAD_SEP, PENALIDAD_NO_PROG
                    )
                    tiempo_fin_grasp = time.perf_counter()
                    tiempo_comp_grasp = tiempo_fin_grasp - tiempo_inicio_grasp

                    # solucion_grasp ya contiene 'costo_total' (base), 'costo_penalizado', 'es_factible' (estricta), y violaciones
                    print(f"        Costo Base GRASP: {solucion_grasp['costo_total']:.2f}, Penalizado: {solucion_grasp['costo_penalizado']:.2f}, Factible: {solucion_grasp['es_factible']} (Tiempo: {tiempo_comp_grasp:.4f}s)")

                    detalles_grasp = f"IterGRASP:{iter_grasp_cfg};SemIni:{SEMILLA_INICIAL_GRASP};K_RCL:{K_RCL_GE};IterHC:{MAX_ITER_SIN_MEJORA_HC}"
                    escribir_resumen_solucion_csv(
                        path_completo_csv_resumen, nombre_base_del_caso, algoritmo_nombre_grasp, num_pistas_actual,
                        detalles_grasp, 
                        {'costo_base': solucion_grasp['costo_total'], # Adaptar para la función de escritura
                         'costo_penalizado': solucion_grasp['costo_penalizado'],
                         'es_estrictamente_factible': solucion_grasp['es_factible'],
                         'violaciones_lk_count': solucion_grasp.get('violaciones_lk_count',0),
                         'violaciones_sep_count': solucion_grasp.get('violaciones_sep_count',0),
                         'violaciones_no_prog_count': solucion_grasp.get('violaciones_no_prog_count',0)
                        },
                        tiempo_comp_grasp,
                        solucion_grasp['secuencia_aterrizajes'], solucion_grasp['aviones_no_programados']
                    )
            
            # --- Comentarios originales de verificación de datos del caso ---
            # if len(datos_del_caso['aviones']) == num_aviones_total_caso and len(datos_del_caso['tiempos_separacion']) == num_aviones_total_caso:
            #     print(f"  Consistencia en número de aviones y datos: OK")
            # else:
            #     print(f"  ERROR: Inconsistencia detectada después de la carga.")
            # print(f"  Primer avión (ID {datos_del_caso['aviones'][0]['id']}): {datos_del_caso['aviones'][0]}")
            # print(f"  Tiempos de separación para el primer avión: {datos_del_caso['tiempos_separacion'][0]}")
            # if len(datos_del_caso['tiempos_separacion'][0]) == num_aviones_total_caso:
            #     print(f"    Longitud de tiempos_separacion[0]: OK ({len(datos_del_caso['tiempos_separacion'][0])})")
            # else:
            #     print(f"    ERROR: Longitud de tiempos_separacion[0] incorrecta: {len(datos_del_caso['tiempos_separacion'][0])}, esperaba {num_aviones_total_caso}")
            # if num_aviones_total_caso > 1:
            #     print(f"  Último avión (ID {datos_del_caso['aviones'][-1]['id']}): {datos_del_caso['aviones'][-1]}")
            #     print(f"  Tiempos de separación para el último avión: {datos_del_caso['tiempos_separacion'][-1]}")
            #     if len(datos_del_caso['tiempos_separacion'][-1]) == num_aviones_total_caso:
            #         print(f"    Longitud de tiempos_separacion[-1]: OK ({len(datos_del_caso['tiempos_separacion'][-1])})")
            #     else:
            #         print(f"    ERROR: Longitud de tiempos_separacion[-1] incorrecta: {len(datos_del_caso['tiempos_separacion'][-1])}, esperaba {num_aviones_total_caso}")
            # if num_aviones_total_caso > 20 and num_aviones_total_caso >= 50 : # Ejemplo para case4
            #     idx_intermedio = 49 
            #     print(f"  Avión intermedio (ID {datos_del_caso['aviones'][idx_intermedio]['id']}): {datos_del_caso['aviones'][idx_intermedio]}")
            #     print(f"  Tiempos de separación para el avión intermedio: {datos_del_caso['tiempos_separacion'][idx_intermedio]}")
            #     if len(datos_del_caso['tiempos_separacion'][idx_intermedio]) == num_aviones_total_caso:
            #         print(f"    Longitud de tiempos_separacion[{idx_intermedio}]: OK ({len(datos_del_caso['tiempos_separacion'][idx_intermedio])})")
            #     else:
            #         print(f"    ERROR: Longitud de tiempos_separacion[{idx_intermedio}] incorrecta: {len(datos_del_caso['tiempos_separacion'][idx_intermedio])}, esperaba {num_aviones_total_caso}")
        else: # if datos_del_caso
            print(f"No se pudieron cargar los datos para {ruta_archivo_caso}.")
        print("----------------------------------\n")

if __name__ == '__main__':
    main()
