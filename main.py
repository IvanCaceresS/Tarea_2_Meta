# main.py
import os 
import csv # <--- IMPORTAR CSV
from datetime import datetime # Para nombre de archivo único
import time # <--- IMPORTADO PARA MEDIR TIEMPO

# Asegúrate que estas rutas sean correctas para tu estructura de carpetas
from scripts.lector_cases import read_case 
from scripts.greedy_deterministic import resolver as resolver_gd 
from scripts.greedy_stochastic import resolver as resolver_ge
# from scripts.verificador import verificar_solucion # Mantengo esto comentado como lo tienes

# --- CONFIGURACIÓN PARA CSV RESUMIDO ---
CARPETA_RESULTADOS = "results" 
NOMBRE_BASE_CSV_RESUMEN = f"resumen_soluciones_aterrizajes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
# El path completo se construirá después de asegurar que la carpeta exista
# PATH_COMPLETO_CSV_RESUMEN se definirá en main()

CABECERA_CSV_RESUMEN = [
    'NombreCaso', 'Algoritmo', 'NumPistas', 'SemillaGE', 
    'CostoTotalSolucion', 'TiempoComputacional_seg', 'OrdenAterrizajeIDs',
    'AvionesNoProgramados'
]

# --- FUNCIÓN PARA ESCRIBIR EN CSV RESUMIDO (NUEVA) ---
def escribir_resumen_solucion_csv(path_completo_csv, nombre_caso, algoritmo_nombre, num_pistas, semilla_ge, solucion, tiempo_comp):
    """
    Escribe una línea de resumen de la solución en el archivo CSV especificado.
    """
    if not solucion: # Si el algoritmo no pudo generar una solución
        print(f"      ADVERTENCIA CSV (Resumen): No hay solución para {nombre_caso}, {algoritmo_nombre}, {num_pistas}p, semilla {semilla_ge}.")
        fila_datos = [
            nombre_caso, algoritmo_nombre, num_pistas, 
            semilla_ge if algoritmo_nombre == 'GE' else 'N/A', 
            'ERROR_NO_SOLUCION', f"{tiempo_comp:.4f}", '', ''
        ]
    else:
        costo_total_solucion = solucion.get('costo_total', float('inf')) 
        
        secuencia_aterrizajes = solucion.get('secuencia_aterrizajes', [])
        orden_ids = [aterrizaje['avion_id'] for aterrizaje in secuencia_aterrizajes]
        orden_ids_str = "-".join(map(str, orden_ids)) 

        aviones_no_programados_lista = solucion.get('aviones_no_programados', [])
        aviones_no_programados_str = "-".join(map(str, aviones_no_programados_lista)) if aviones_no_programados_lista else ""

        fila_datos = [
            nombre_caso,
            algoritmo_nombre,
            num_pistas,
            semilla_ge if algoritmo_nombre == 'GE' else 'N/A',
            f"{costo_total_solucion:.2f}",
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
# def escribir_resultados_csv(nombre_caso, algoritmo_nombre, num_pistas, semilla_ge, solucion, datos_entrada_aviones):
#     """
#     Escribe los detalles de una solución en el archivo CSV.
#     """
#     if not solucion or not solucion.get('secuencia_aterrizajes'):
#         print(f"           ADVERTENCIA CSV: No hay secuencia de aterrizajes para {nombre_caso}, {algoritmo_nombre}, {num_pistas}p, semilla {semilla_ge}.")
#         return

#     costo_total_solucion = solucion['costo_total']
    
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

#     escribir_cabecera = not os.path.exists(NOMBRE_ARCHIVO_CSV_DETALLADO) # Usar el nombre del CSV detallado
    
#     try:
#         with open(NOMBRE_ARCHIVO_CSV_DETALLADO, 'a', newline='', encoding='utf-8') as f_csv: # Usar el nombre del CSV detallado
#             writer = csv.writer(f_csv)
#             if escribir_cabecera:
#                 writer.writerow(CABECERA_CSV_DETALLADO) # Usar la cabecera del CSV detallado
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
            return # Salir si no se puede crear la carpeta
            
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
            num_aviones = datos_del_caso['num_aviones']
            # aviones = datos_del_caso['aviones'] # Comentado como lo tenías
            # tiempos_separacion = datos_del_caso['tiempos_separacion'] # Comentado como lo tenías

            print(f"Número de aviones: {num_aviones}")
            
            # --- Greedy Determinista ---
            print("\n  Ejecutando Greedy Determinista:")
            algoritmo_actual_nombre_gd = "GD" 
            
            # Para 1 pista (GD)
            num_pistas_gd_1 = 1
            print(f"    Calculando para {num_pistas_gd_1} pista (GD):")
            tiempo_inicio_gd1 = time.perf_counter()
            solucion_gd_1pista = resolver_gd(datos_del_caso, num_pistas=num_pistas_gd_1)
            tiempo_fin_gd1 = time.perf_counter()
            tiempo_comp_gd1 = tiempo_fin_gd1 - tiempo_inicio_gd1
            
            if solucion_gd_1pista:
                print(f"      Costo Total ({num_pistas_gd_1} pista, GD): {solucion_gd_1pista['costo_total']:.2f} (Tiempo: {tiempo_comp_gd1:.4f}s)")
                if 'aviones_no_programados' in solucion_gd_1pista and solucion_gd_1pista['aviones_no_programados']:
                    print(f"      Aviones no programados ({num_pistas_gd_1} pista, GD): {solucion_gd_1pista['aviones_no_programados']}")
                escribir_resumen_solucion_csv(path_completo_csv_resumen, nombre_base_del_caso, algoritmo_actual_nombre_gd, num_pistas_gd_1, 'N/A', solucion_gd_1pista, tiempo_comp_gd1)
            else:
                print(f"      No se obtuvo solución GD para {num_pistas_gd_1} pista.")
                escribir_resumen_solucion_csv(path_completo_csv_resumen, nombre_base_del_caso, algoritmo_actual_nombre_gd, num_pistas_gd_1, 'N/A', None, tiempo_comp_gd1)


            # Para 2 pistas (GD)
            num_pistas_gd_2 = 2
            print(f"    Calculando para {num_pistas_gd_2} pistas (GD):")
            tiempo_inicio_gd2 = time.perf_counter()
            solucion_gd_2pistas = resolver_gd(datos_del_caso, num_pistas=num_pistas_gd_2)
            tiempo_fin_gd2 = time.perf_counter()
            tiempo_comp_gd2 = tiempo_fin_gd2 - tiempo_inicio_gd2
            
            if solucion_gd_2pistas:
                print(f"      Costo Total ({num_pistas_gd_2} pistas, GD): {solucion_gd_2pistas['costo_total']:.2f} (Tiempo: {tiempo_comp_gd2:.4f}s)")
                if 'aviones_no_programados' in solucion_gd_2pistas and solucion_gd_2pistas['aviones_no_programados']:
                    print(f"      Aviones no programados ({num_pistas_gd_2} pistas, GD): {solucion_gd_2pistas['aviones_no_programados']}")
                escribir_resumen_solucion_csv(path_completo_csv_resumen, nombre_base_del_caso, algoritmo_actual_nombre_gd, num_pistas_gd_2, 'N/A', solucion_gd_2pistas, tiempo_comp_gd2)
            else:
                print(f"      No se obtuvo solución GD para {num_pistas_gd_2} pistas.")
                escribir_resumen_solucion_csv(path_completo_csv_resumen, nombre_base_del_caso, algoritmo_actual_nombre_gd, num_pistas_gd_2, 'N/A', None, tiempo_comp_gd2)

            # --- Greedy Estocástico ---
            print("\n  Ejecutando Greedy Estocástico:")
            algoritmo_actual_nombre_ge = "GE" 
            num_ejecuciones_ge = 10
            k_rcl_ge = 3 

            for num_pista_actual_ge in [1, 2]:
                print(f"    Calculando para {num_pista_actual_ge} pista(s) (GE, {num_ejecuciones_ge} ejecuciones):")
                resultados_ge_iteraciones_costos = []
                
                for i in range(num_ejecuciones_ge):
                    semilla_actual = i 
                    tiempo_inicio_ge = time.perf_counter()
                    sol_ge_actual = resolver_ge(datos_del_caso, num_pistas=num_pista_actual_ge, semilla=semilla_actual, parametro_rcl_alpha=k_rcl_ge) 
                    tiempo_fin_ge = time.perf_counter()
                    tiempo_comp_ge = tiempo_fin_ge - tiempo_inicio_ge
                    
                    if sol_ge_actual:
                        costo_actual_ge = sol_ge_actual['costo_total']
                        resultados_ge_iteraciones_costos.append(costo_actual_ge)
                        escribir_resumen_solucion_csv(path_completo_csv_resumen, nombre_base_del_caso, algoritmo_actual_nombre_ge, num_pista_actual_ge, semilla_actual, sol_ge_actual, tiempo_comp_ge)
                    else:
                        print(f"      Ejecución GE {i+1} (semilla {semilla_actual}): No se obtuvo solución.")
                        resultados_ge_iteraciones_costos.append(float('inf')) 
                        escribir_resumen_solucion_csv(path_completo_csv_resumen, nombre_base_del_caso, algoritmo_actual_nombre_ge, num_pista_actual_ge, semilla_actual, None, tiempo_comp_ge)

                if resultados_ge_iteraciones_costos:
                    costos_validos_ge = [c for c in resultados_ge_iteraciones_costos if c != float('inf')]
                    if costos_validos_ge:
                        print(f"      Resultados GE ({num_pista_actual_ge} pista(s)):")
                        print(f"        Mejor Costo: {min(costos_validos_ge):.2f}")
                        print(f"        Costo Promedio: {sum(costos_validos_ge)/len(costos_validos_ge):.2f}")
                        print(f"        Peor Costo: {max(costos_validos_ge):.2f}")
                        # Para el informe, querrás todos los costos de las 10 ejecuciones:
                        # print(f"        Todos los costos: {[f'{c:.2f}' for c in costos_validos_ge]}") # Comentado como lo tenías
                    else:
                        print(f"      Resultados GE ({num_pista_actual_ge} pista(s)): No se obtuvieron soluciones válidas.")
                else:
                    print(f"      Resultados GE ({num_pista_actual_ge} pista(s)): No se realizaron ejecuciones.")
            
            # # Verificar consistencia general (ya lo hace tu lector, pero no está de más) # Comentado como lo tenías
            # # if len(aviones) == num_aviones and len(tiempos_separacion) == num_aviones: # Comentado como lo tenías
            # #     print(f"  Consistencia en número de aviones y datos: OK") # Comentado como lo tenías
            # # # Verificar primer avión # Comentado como lo tenías
            # # print(f"  Primer avión (ID {aviones[0]['id']}): {aviones[0]}") # Comentado como lo tenías
            # # print(f"  Tiempos de separación para el primer avión: {tiempos_separacion[0]}") # Comentado como lo tenías
            # # if len(tiempos_separacion[0]) == num_aviones: # Comentado como lo tenías
            # #     print(f"    Longitud de tiempos_separacion[0]: OK ({len(tiempos_separacion[0])})") # Comentado como lo tenías
            # # else: # Comentado como lo tenías
            # #     print(f"    ERROR: Longitud de tiempos_separacion[0] incorrecta: {len(tiempos_separacion[0])}, esperaba {num_aviones}") # Comentado como lo tenías
            # # # Verificar último avión (si hay más de uno) # Comentado como lo tenías
            # # if num_aviones > 1: # Comentado como lo tenías
            # #     print(f"  Último avión (ID {aviones[-1]['id']}): {aviones[-1]}") # Comentado como lo tenías
            # #     print(f"  Tiempos de separación para el último avión: {tiempos_separacion[-1]}") # Comentado como lo tenías
            # #     if len(tiempos_separacion[-1]) == num_aviones: # Comentado como lo tenías
            # #         print(f"    Longitud de tiempos_separacion[-1]: OK ({len(tiempos_separacion[-1])})") # Comentado como lo tenías
            # #     else: # Comentado como lo tenías
            # #         print(f"    ERROR: Longitud de tiempos_separacion[-1] incorrecta: {len(tiempos_separacion[-1])}, esperaba {num_aviones}") # Comentado como lo tenías
            # # # Verificar un avión intermedio para casos grandes # Comentado como lo tenías
            # # if num_aviones > 20 and num_aviones >= 50 : # Ejemplo para case4 # Comentado como lo tenías
            # #     idx_intermedio = 49  # Comentado como lo tenías
            # #     print(f"  Avión intermedio (ID {aviones[idx_intermedio]['id']}): {aviones[idx_intermedio]}") # Comentado como lo tenías
            # #     print(f"  Tiempos de separación para el avión intermedio: {tiempos_separacion[idx_intermedio]}") # Comentado como lo tenías
            # #     if len(tiempos_separacion[idx_intermedio]) == num_aviones: # Comentado como lo tenías
            # #         print(f"    Longitud de tiempos_separacion[{idx_intermedio}]: OK ({len(tiempos_separacion[idx_intermedio])})") # Comentado como lo tenías
            # #     else: # Comentado como lo tenías
            # #         print(f"    ERROR: Longitud de tiempos_separacion[{idx_intermedio}] incorrecta: {len(tiempos_separacion[idx_intermedio])}, esperaba {num_aviones}") # Comentado como lo tenías
            # # else: # Comentado como lo tenías
            # # print(f"  ERROR: Inconsistencia detectada después de la carga.") # Comentado como lo tenías

        else:
            print(f"No se pudieron cargar los datos para {ruta_archivo_caso}.")
        print("----------------------------------\n")

if __name__ == '__main__':
    main()