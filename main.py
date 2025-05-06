# main.py
import os 
from scripts.lector_cases import read_case 
from scripts.greedy_deterministic import resolver as resolver_gd 
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
            
            # Para 1 pista
            print("    Calculando para 1 pista:")
            solucion_gd_1pista = resolver_gd(datos_del_caso, num_pistas=1)
            if solucion_gd_1pista:
                print(f"      Costo Total (1 pista): {solucion_gd_1pista['costo_total']:.2f}")
                if 'aviones_no_programados' in solucion_gd_1pista and solucion_gd_1pista['aviones_no_programados']:
                    print(f"      Aviones no programados (1 pista): {solucion_gd_1pista['aviones_no_programados']}")
                
                print("      Verificando solución (1 pista):")
                es_valida_1pista = verificar_solucion(solucion_gd_1pista, datos_del_caso, num_pistas_usadas=1)
                if not es_valida_1pista:
                    print("      ¡¡¡ ATENCIÓN: LA SOLUCIÓN GD PARA 1 PISTA NO ES VÁLIDA !!!")
            else:
                print("      No se obtuvo solución GD para 1 pista.")

            # Para 2 pistas
            print("    Calculando para 2 pistas:")
            solucion_gd_2pistas = resolver_gd(datos_del_caso, num_pistas=2)
            if solucion_gd_2pistas:
                print(f"      Costo Total (2 pistas): {solucion_gd_2pistas['costo_total']:.2f}")
                if 'aviones_no_programados' in solucion_gd_2pistas and solucion_gd_2pistas['aviones_no_programados']:
                    print(f"      Aviones no programados (2 pistas): {solucion_gd_2pistas['aviones_no_programados']}")

                print("      Verificando solución (2 pistas):")
                es_valida_2pistas = verificar_solucion(solucion_gd_2pistas, datos_del_caso, num_pistas_usadas=2)
                if not es_valida_2pistas:
                    print("      ¡¡¡ ATENCIÓN: LA SOLUCIÓN GD PARA 2 PISTAS NO ES VÁLIDA !!!")
            else:
                print("      No se obtuvo solución GD para 2 pistas.")
            
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