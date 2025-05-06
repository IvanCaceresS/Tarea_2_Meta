# scripts/greedy_estocastico.py
import random
import copy
# Probablemente necesites tu función calcular_costo_aterrizaje aquí también
# from .verificador import calcular_costo_aterrizaje # Si está en verificador
# o cópiala directamente

# (Puedes copiar calcular_costo_aterrizaje aquí si no la importas)
def calcular_costo_aterrizaje(tiempo_aterrizaje, avion_info):
    costo = 0
    if tiempo_aterrizaje < avion_info['P']:
        costo = (avion_info['P'] - tiempo_aterrizaje) * avion_info['cost_temprano']
    elif tiempo_aterrizaje > avion_info['P']:
        costo = (tiempo_aterrizaje - avion_info['P']) * avion_info['cost_tardio']
    return costo


def resolver(datos_del_caso, num_pistas, semilla, parametro_rcl_alpha=0.1): # parametro_rcl_alpha es un ejemplo
    """
    Resuelve usando un greedy estocástico.
    parametro_rcl_alpha: Porcentaje para construir la RCL (e.g., 0.1 para 10% del mejor)
                         o podría ser un k (tamaño fijo de RCL).
    """
    random.seed(semilla) # ¡Importante para la reproducibilidad!

    aviones_originales = datos_del_caso['aviones']
    tiempos_separacion = datos_del_caso['tiempos_separacion']
    num_aviones_total = datos_del_caso['num_aviones']

    solucion = {
        'secuencia_aterrizajes': [],
        'costo_total': 0,
        'aviones_no_programados': []
    }
    
    estado_pistas = [{'ultimo_avion_id': None, 'ultimo_tiempo_aterrizaje': -float('inf')} for _ in range(num_pistas)]
    
    # Lista de aviones aún no programados (inicialmente todos, representados por sus IDs o copias de su info)
    aviones_no_programados = [copy.deepcopy(avion) for avion in aviones_originales]

    while len(solucion['secuencia_aterrizajes']) < num_aviones_total and aviones_no_programados:
        
        # --- Construcción de la RCL ---
        if not aviones_no_programados:
            break # No quedan aviones por programar

        # 1. Evaluar todos los aviones no programados según un criterio (ej. E_k)
        #    Guardar (valor_criterio, avion_info)
        candidatos_evaluados = []
        for avion_info_iter in aviones_no_programados:
            # Aquí el criterio podría ser más complejo si se considera la "mejor inserción posible ahora"
            # Por simplicidad, usemos E_k como en el determinista para ordenar inicialmente.
            criterio_valor = avion_info_iter['P'] # Podría ser P_k, o una estimación del costo de inserción
            candidatos_evaluados.append({'avion_info': avion_info_iter, 'criterio': criterio_valor})

        if not candidatos_evaluados:
            break # No hay candidatos válidos, algo raro.

        # Ordenar candidatos por el criterio (ascendente para E_k o P_k)
        candidatos_evaluados.sort(key=lambda x: x['criterio'])
        
        mejor_criterio = candidatos_evaluados[0]['criterio']
        
        # Construir RCL: aquellos dentro de alpha % del mejor criterio
        # O un k fijo, ej. k = max(1, int(len(candidatos_evaluados) * parametro_rcl_alpha)) o un k=3, k=5...
        # Usemos el enfoque de alpha para este ejemplo:
        limite_rcl = mejor_criterio * (1 + parametro_rcl_alpha) # Si el criterio es costo, sería <=. Si es tiempo temprano, podría ser <=
        # Para E_k o P_k, el límite sería:
        # limite_rcl = mejor_criterio + (max_Ek - min_Ek) * parametro_rcl_alpha (una forma de normalizar)
        # O más simple: los 'k' mejores, o los que estén "cerca" del mejor E_k.
        # Vamos con una RCL de los candidatos cuyo E_k esté cerca del mejor E_k.
        # Por ejemplo, todos los que tengan E_k <= mejor_E_k + umbral_tiempo (ej. umbral_tiempo = 10 o 20)
        # O, si usamos alpha: limite_rcl = mejor_criterio + alpha * (datos_del_caso['aviones_originales_max_L_menos_min_E'] * alpha)
        # Tomemos un enfoque más simple para RCL: los top N o los que estén dentro de un rango del mejor.
        # Ejemplo: alpha como fracción de candidatos a considerar (ej. 0.2 = top 20%)
        
        rcl_size = max(1, int(len(candidatos_evaluados) * parametro_rcl_alpha)) if parametro_rcl_alpha <= 1 else int(parametro_rcl_alpha) # Puede ser k_fijo o alpha
        rcl = [c['avion_info'] for c in candidatos_evaluados[:rcl_size]]


        if not rcl: # Si la RCL queda vacía (no debería si rcl_size >=1)
            # Esto podría pasar si todos los aviones restantes son imposibles de programar
            # y candidatos_evaluados se vacía o el criterio no funciona.
            # Por seguridad, tomar el mejor si la RCL está vacía y hay candidatos.
            if candidatos_evaluados:
                 rcl = [candidatos_evaluados[0]['avion_info']]
            else:
                 break # No hay más aviones que intentar

        # --- Selección Aleatoria de la RCL ---
        avion_actual_info = random.choice(rcl)
        
        # El resto de la lógica es muy similar al greedy determinista:
        # encontrar la mejor pista y tiempo para 'avion_actual_info'
        avion_id_actual = avion_actual_info['id']
        mejor_opcion_para_avion_actual = {
            'pista_asignada': -1,
            'tiempo_aterrizaje_final': -1,
            'costo_penalizacion': float('inf'),
            'valida': False
        }

        for pista_idx in range(num_pistas):
            tiempo_aterrizaje_min_pista = avion_actual_info['E']
            ultimo_avion_en_pista_id = estado_pistas[pista_idx]['ultimo_avion_id']
            if ultimo_avion_en_pista_id is not None:
                tiempo_separacion_requerido = tiempos_separacion[ultimo_avion_en_pista_id][avion_id_actual]
                tiempo_aterrizaje_min_pista = max(
                    tiempo_aterrizaje_min_pista,
                    estado_pistas[pista_idx]['ultimo_tiempo_aterrizaje'] + tiempo_separacion_requerido
                )
            
            tiempo_aterrizaje_propuesto = max(tiempo_aterrizaje_min_pista, avion_actual_info['P'])

            if tiempo_aterrizaje_propuesto > avion_actual_info['L']:
                if tiempo_aterrizaje_min_pista <= avion_actual_info['L']:
                    tiempo_aterrizaje_final_pista = tiempo_aterrizaje_min_pista
                else:
                    continue 
            else:
                tiempo_aterrizaje_final_pista = tiempo_aterrizaje_propuesto

            costo_actual = calcular_costo_aterrizaje(tiempo_aterrizaje_final_pista, avion_actual_info)

            if costo_actual < mejor_opcion_para_avion_actual['costo_penalizacion']:
                mejor_opcion_para_avion_actual['pista_asignada'] = pista_idx
                mejor_opcion_para_avion_actual['tiempo_aterrizaje_final'] = tiempo_aterrizaje_final_pista
                mejor_opcion_para_avion_actual['costo_penalizacion'] = costo_actual
                mejor_opcion_para_avion_actual['valida'] = True
            elif costo_actual == mejor_opcion_para_avion_actual['costo_penalizacion']:
                # Desempate aleatorio si los costos son iguales para diferentes pistas
                if random.choice([True, False]):
                    mejor_opcion_para_avion_actual['pista_asignada'] = pista_idx
                    mejor_opcion_para_avion_actual['tiempo_aterrizaje_final'] = tiempo_aterrizaje_final_pista
                    # costo_penalizacion no cambia


        if mejor_opcion_para_avion_actual['valida']:
            p_asignada = mejor_opcion_para_avion_actual['pista_asignada']
            t_final = mejor_opcion_para_avion_actual['tiempo_aterrizaje_final']
            c_individual = mejor_opcion_para_avion_actual['costo_penalizacion']

            solucion['secuencia_aterrizajes'].append({
                'avion_id': avion_id_actual,
                'pista': p_asignada,
                'tiempo': t_final,
                'costo_individual': c_individual
            })
            solucion['costo_total'] += c_individual

            estado_pistas[p_asignada]['ultimo_avion_id'] = avion_id_actual
            estado_pistas[p_asignada]['ultimo_tiempo_aterrizaje'] = t_final
            
            # Remover el avión programado de la lista de no programados
            aviones_no_programados = [avion for avion in aviones_no_programados if avion['id'] != avion_id_actual]
        else:
            solucion['aviones_no_programados'].append(avion_id_actual)
            # Es importante removerlo de todas formas para no intentar re-programarlo infinitamente si es infactible
            aviones_no_programados = [avion for avion in aviones_no_programados if avion['id'] != avion_id_actual]


    solucion['secuencia_aterrizajes'].sort(key=lambda x: x['tiempo'])
    return solucion

# --- Bloque de prueba ---
if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
    from scripts.lector_cases import read_case 
    from scripts.verificador import verificar_solucion

    ruta_case1 = os.path.join('..', 'cases', 'case1.txt') 
    datos_case1 = read_case(ruta_case1)

    if datos_case1:
        print("\n--- Greedy Estocástico: case1.txt ---")
        
        # Parámetros para el greedy estocástico
        num_ejecuciones = 10
        # alpha_rcl = 0.2 # Tomar el 20% de los mejores candidatos para la RCL
        k_rcl = 5       # Tomar los k=5 mejores candidatos para la RCL

        for num_pista_actual in [1, 2]:
            print(f"\n  Resultados para {num_pista_actual} pista(s):")
            costos_ejecuciones = []
            mejor_solucion_estocastica = None
            mejor_costo_estocastico = float('inf')

            for i in range(num_ejecuciones):
                semilla = i # Semillas 0, 1, ..., 9
                # Pasar k_rcl como parametro_rcl_alpha (o renombrar el parámetro en la función)
                sol_actual = resolver(datos_case1, num_pistas=num_pista_actual, semilla=semilla, parametro_rcl_alpha=k_rcl)
                
                print(f"    Ejecución {i+1} (semilla {semilla}): Costo = {sol_actual['costo_total']:.2f}", end="")
                if sol_actual['aviones_no_programados']:
                    print(f", No Programados: {sol_actual['aviones_no_programados']}", end="")
                
                es_valida = verificar_solucion(sol_actual, datos_case1, num_pistas_usadas=num_pista_actual)
                if not es_valida:
                    print(" ¡¡¡SOLUCIÓN INVÁLIDA!!!", end="")
                print() # Nueva línea

                costos_ejecuciones.append(sol_actual['costo_total'])
                if sol_actual['costo_total'] < mejor_costo_estocastico and es_valida:
                    mejor_costo_estocastico = sol_actual['costo_total']
                    mejor_solucion_estocastica = sol_actual
            
            if costos_ejecuciones:
                print(f"    Mejor costo de {num_ejecuciones} ejecuciones: {min(costos_ejecuciones):.2f}")
                print(f"    Costo promedio: {sum(costos_ejecuciones)/len(costos_ejecuciones):.2f}")
                print(f"    Peor costo: {max(costos_ejecuciones):.2f}")
    else:
        print("No se pudieron cargar datos para case1.txt para la prueba del greedy estocástico.")