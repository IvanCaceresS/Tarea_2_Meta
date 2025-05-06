# main.py
import os 
from scripts.lector_cases import read_case 
from scripts.greedy_deterministic import resolver as resolver_gd 
from scripts.greedy_stochastic import resolver as resolver_ge
from scripts.verificador import verificar_solucion

def main():
    nombre_carpeta_casos = 'cases'
    nombres_base_casos = ['case1.txt', 'case2.txt', 'case3.txt', 'case4.txt']
    # Para una prueba más detallada:
    # nombres_base_casos = ['case1.txt'] 

    archivos_casos = [os.path.join(nombre_carpeta_casos, nombre) for nombre in nombres_base_casos]

    for ruta_archivo_caso in archivos_casos:
        print(f"\nProcesando: {ruta_archivo_caso} ")
        datos_del_caso = read_case(ruta_archivo_caso)

        if datos_del_caso:
            num_aviones = datos_del_caso['num_aviones']
            # aviones = datos_del_caso['aviones']
            # tiempos_separacion = datos_del_caso['tiempos_separacion']

            print(f"Número de aviones: {num_aviones}")
            
            # --- Greedy Determinista ---
            print("\n  Ejecutando Greedy Determinista:")
            
            # Para 1 pista (GD)
            print("    Calculando para 1 pista (GD):")
            solucion_gd_1pista = resolver_gd(datos_del_caso, num_pistas=1)
            if solucion_gd_1pista:
                print(f"      Costo Total (1 pista, GD): {solucion_gd_1pista['costo_total']:.2f}")
                if 'aviones_no_programados' in solucion_gd_1pista and solucion_gd_1pista['aviones_no_programados']:
                    print(f"      Aviones no programados (1 pista, GD): {solucion_gd_1pista['aviones_no_programados']}")
                # print("      Verificando solución (1 pista, GD):")
                # es_valida_1pista = verificar_solucion(solucion_gd_1pista, datos_del_caso, num_pistas_usadas=1)
                # if not es_valida_1pista:
                #     print("      ¡¡¡ ATENCIÓN: LA SOLUCIÓN GD PARA 1 PISTA NO ES VÁLIDA !!!")
            else:
                print("      No se obtuvo solución GD para 1 pista.")

            # Para 2 pistas (GD)
            print("    Calculando para 2 pistas (GD):")
            solucion_gd_2pistas = resolver_gd(datos_del_caso, num_pistas=2)
            if solucion_gd_2pistas:
                print(f"      Costo Total (2 pistas, GD): {solucion_gd_2pistas['costo_total']:.2f}")
                if 'aviones_no_programados' in solucion_gd_2pistas and solucion_gd_2pistas['aviones_no_programados']:
                    print(f"      Aviones no programados (2 pistas, GD): {solucion_gd_2pistas['aviones_no_programados']}")
                # print("      Verificando solución (2 pistas, GD):")
                # es_valida_2pistas = verificar_solucion(solucion_gd_2pistas, datos_del_caso, num_pistas_usadas=2)
                # if not es_valida_2pistas:
                #     print("      ¡¡¡ ATENCIÓN: LA SOLUCIÓN GD PARA 2 PISTAS NO ES VÁLIDA !!!")
            else:
                print("      No se obtuvo solución GD para 2 pistas.")

            # --- Greedy Estocástico ---
            print("\n  Ejecutando Greedy Estocástico:")
            num_ejecuciones_ge = 10
            k_rcl_ge = 3 # Ejemplo de parámetro k para la RCL (puedes ajustarlo)

            for num_pista_actual_ge in [1, 2]:
                print(f"    Calculando para {num_pista_actual_ge} pista(s) (GE, {num_ejecuciones_ge} ejecuciones):")
                resultados_ge_iteraciones_costos = []
                # mejor_costo_ge_caso_pistas = float('inf') # Si quisieras guardar la mejor solución completa

                for i in range(num_ejecuciones_ge):
                    semilla_actual = i 
                    # Asegúrate que el parámetro en resolver_ge se llame parametro_rcl_alpha o ajusta la llamada
                    sol_ge_actual = resolver_ge(datos_del_caso, num_pistas=num_pista_actual_ge, semilla=semilla_actual, parametro_rcl_alpha=k_rcl_ge) 
                    
                    if sol_ge_actual:
                        costo_actual_ge = sol_ge_actual['costo_total']
                        resultados_ge_iteraciones_costos.append(costo_actual_ge)
                        
                        # Opcional: imprimir costo de cada ejecución si quieres mucho detalle
                        # print(f"      Ejec. {i+1} (semilla {semilla_actual}): Costo = {costo_actual_ge:.2f}")

                        # Comentar la verificación por ahora
                        # es_valida_ge = verificar_solucion(sol_ge_actual, datos_del_caso, num_pistas_usadas=num_pista_actual_ge)
                        # if not es_valida_ge:
                        #     print(f"        ¡¡¡ SOLUCIÓN GE EJEC.{i+1} INVÁLIDA !!!")
                        
                        # if costo_actual_ge < mejor_costo_ge_caso_pistas: # and es_valida_ge
                        #     mejor_costo_ge_caso_pistas = costo_actual_ge
                    else:
                        print(f"      Ejecución {i+1} (semilla {semilla_actual}): No se obtuvo solución.")
                        resultados_ge_iteraciones_costos.append(float('inf')) # Para no romper min/max/promedio si falla una ejecución

                if resultados_ge_iteraciones_costos:
                    costos_validos_ge = [c for c in resultados_ge_iteraciones_costos if c != float('inf')]
                    if costos_validos_ge:
                        print(f"      Resultados GE ({num_pista_actual_ge} pista(s)):")
                        print(f"        Mejor Costo: {min(costos_validos_ge):.2f}")
                        print(f"        Costo Promedio: {sum(costos_validos_ge)/len(costos_validos_ge):.2f}")
                        print(f"        Peor Costo: {max(costos_validos_ge):.2f}")
                        # Para el informe, querrás todos los costos de las 10 ejecuciones:
                        # print(f"        Todos los costos: {[f'{c:.2f}' for c in costos_validos_ge]}")
                    else:
                        print(f"      Resultados GE ({num_pista_actual_ge} pista(s)): No se obtuvieron soluciones válidas.")
                else:
                     print(f"      Resultados GE ({num_pista_actual_ge} pista(s)): No se realizaron ejecuciones.")
            
                
                
            
            # # Verificar consistencia general (ya lo hace tu lector, pero no está de más)
            # if len(aviones) == num_aviones and len(tiempos_separacion) == num_aviones:
            #     print(f"  Consistencia en número de aviones y datos: OK")
                
            #     # Verificar primer avión
            #     print(f"  Primer avión (ID {aviones[0]['id']}): {aviones[0]}")
            #     print(f"  Tiempos de separación para el primer avión: {tiempos_separacion[0]}")
            #     if len(tiempos_separacion[0]) == num_aviones:
            #         print(f"    Longitud de tiempos_separacion[0]: OK ({len(tiempos_separacion[0])})")
            #     else:
            #         print(f"    ERROR: Longitud de tiempos_separacion[0] incorrecta: {len(tiempos_separacion[0])}, esperaba {num_aviones}")

            #     # Verificar último avión (si hay más de uno)
            #     if num_aviones > 1:
            #         print(f"  Último avión (ID {aviones[-1]['id']}): {aviones[-1]}")
            #         print(f"  Tiempos de separación para el último avión: {tiempos_separacion[-1]}")
            #         if len(tiempos_separacion[-1]) == num_aviones:
            #             print(f"    Longitud de tiempos_separacion[-1]: OK ({len(tiempos_separacion[-1])})")
            #         else:
            #             print(f"    ERROR: Longitud de tiempos_separacion[-1] incorrecta: {len(tiempos_separacion[-1])}, esperaba {num_aviones}")

            #     # Verificar un avión intermedio para casos grandes
            #     if num_aviones > 20 and num_aviones >= 50 : # Ejemplo para case4
            #         idx_intermedio = 49 
            #         print(f"  Avión intermedio (ID {aviones[idx_intermedio]['id']}): {aviones[idx_intermedio]}")
            #         print(f"  Tiempos de separación para el avión intermedio: {tiempos_separacion[idx_intermedio]}")
            #         if len(tiempos_separacion[idx_intermedio]) == num_aviones:
            #             print(f"    Longitud de tiempos_separacion[{idx_intermedio}]: OK ({len(tiempos_separacion[idx_intermedio])})")
            #         else:
            #             print(f"    ERROR: Longitud de tiempos_separacion[{idx_intermedio}] incorrecta: {len(tiempos_separacion[idx_intermedio])}, esperaba {num_aviones}")
            
            # else:
            #     print(f"  ERROR: Inconsistencia detectada después de la carga.")

        else:
            print(f"No se pudieron cargar los datos para {ruta_archivo_caso}.")
        print("----------------------------------\n")

if __name__ == '__main__':
    main()