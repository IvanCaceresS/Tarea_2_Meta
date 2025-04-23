import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np # Para manejar infinito
import os
import time

# --- Constantes y Configuración ---
# *** USAR RUTAS CORREGIDAS ***
JSON_INPUT_FILENAME = './results/all_results_details.json' # Asegúrate que el JSON exista aquí
OUTPUT_GRAPH_DIR = './results/graficos_resultados'
INFINITO_COSTO = float('inf')
GRASP_ITERATIONS_LIST = [5, 15, 30] # Debe coincidir con main.py

# Crear directorio de salida si no existe
if not os.path.exists(OUTPUT_GRAPH_DIR):
    os.makedirs(OUTPUT_GRAPH_DIR)

# --- Funciones Auxiliares ---
def get_stats_from_runs(run_list, key='cost'):
    """Extrae estadísticas (min, avg, max, stdev) para una clave dada ('cost' o 'time')."""
    values = [r[key] for r in run_list if r.get('cost', INFINITO_COSTO) != INFINITO_COSTO and key in r] # Solo de runs válidos
    if not values:
        return {'min': None, 'avg': None, 'max': None, 'stdev': None, 'valid_count': 0}
    valid_count = len(values)
    # Usar np.nanmin/max/mean/std podría ser más robusto si hubiera NaNs, pero aquí filtramos antes
    return {
        'min': min(values),
        'avg': np.mean(values),
        'max': max(values),
        'stdev': np.std(values) if valid_count > 1 else 0.0,
        'valid_count': valid_count
    }

# --- Cargar Datos ---
try:
    with open(JSON_INPUT_FILENAME, 'r') as f:
        all_results = json.load(f)
    print(f"Datos cargados exitosamente desde '{JSON_INPUT_FILENAME}'.")
except FileNotFoundError:
    print(f"Error: El archivo '{JSON_INPUT_FILENAME}' no fue encontrado.")
    print("Asegúrate de ejecutar main.py primero para generar el archivo JSON.")
    exit()
except json.JSONDecodeError:
    print(f"Error: El archivo '{JSON_INPUT_FILENAME}' no es un JSON válido.")
    exit()

# --- Procesar Datos para Graficar (Costos y Tiempos) ---
plot_data = []
cases = sorted(all_results.keys())

for case_name in cases:
    case_results = all_results[case_name]
    data_entry = {'Caso': case_name}

    # --- Extraer Costos y Tiempos ---
    # Función auxiliar para extraer ambos
    def get_result_values(key_base):
        res = case_results.get(key_base, {})
        cost = res.get('cost', INFINITO_COSTO)
        exec_time = res.get('time', 0.0) # Usar 0.0 como default para tiempo
        return cost, exec_time

    # Determinista
    cost_d1, time_d1 = get_result_values('deterministic_1_runway')
    cost_d2, time_d2 = get_result_values('deterministic_2_runways')
    data_entry['Greedy Det 1P Costo'] = cost_d1
    data_entry['Greedy Det 1P Tiempo'] = time_d1
    data_entry['Greedy Det 2P Costo'] = cost_d2
    data_entry['Greedy Det 2P Tiempo'] = time_d2

    # HC desde Determinista
    cost_hc_d1, time_hc_d1 = get_result_values('hc_from_deterministic_1_runway')
    cost_hc_d2, time_hc_d2 = get_result_values('hc_from_deterministic_2_runways')
    data_entry['HC desde Det 1P Costo'] = cost_hc_d1
    data_entry['HC desde Det 1P Tiempo'] = time_hc_d1
    data_entry['HC desde Det 2P Costo'] = cost_hc_d2
    data_entry['HC desde Det 2P Tiempo'] = time_hc_d2

    # Estadísticas Estocásticas (Item 1)
    stoch_runs_1r = case_results.get('stochastic_1_runway_runs', [])
    stoch_runs_2r = case_results.get('stochastic_2_runway_runs', [])
    stoch_1r_cost_stats = get_stats_from_runs(stoch_runs_1r, 'cost')
    stoch_1r_time_stats = get_stats_from_runs(stoch_runs_1r, 'time') # Stats de tiempo
    stoch_2r_cost_stats = get_stats_from_runs(stoch_runs_2r, 'cost')
    stoch_2r_time_stats = get_stats_from_runs(stoch_runs_2r, 'time') # Stats de tiempo
    data_entry['Stoch Avg 1P Costo'] = stoch_1r_cost_stats['avg']
    data_entry['Stoch Min 1P Costo'] = stoch_1r_cost_stats['min']
    data_entry['Stoch Avg 1P Tiempo'] = stoch_1r_time_stats['avg'] # Tiempo promedio
    data_entry['Stoch Avg 2P Costo'] = stoch_2r_cost_stats['avg']
    data_entry['Stoch Min 2P Costo'] = stoch_2r_cost_stats['min']
    data_entry['Stoch Avg 2P Tiempo'] = stoch_2r_time_stats['avg'] # Tiempo promedio

    # Resultados GRASP Estocástico
    for iters in GRASP_ITERATIONS_LIST:
        cost_g1, time_g1 = get_result_values(f'grasp_stochastic_{iters}iters_1_runway')
        cost_g2, time_g2 = get_result_values(f'grasp_stochastic_{iters}iters_2_runways')
        data_entry[f'GRASP {iters}r 1P Costo'] = cost_g1
        data_entry[f'GRASP {iters}r 1P Tiempo'] = time_g1
        data_entry[f'GRASP {iters}r 2P Costo'] = cost_g2
        data_entry[f'GRASP {iters}r 2P Tiempo'] = time_g2

    plot_data.append(data_entry)

# Crear DataFrame de Pandas
df = pd.DataFrame(plot_data)
# Reemplazar Infinito con NaN para que matplotlib los ignore o maneje mejor
df.replace(INFINITO_COSTO, np.nan, inplace=True)
# Reemplazar None (de stats si no hubo válidos) con NaN
df.fillna(np.nan, inplace=True)
df.set_index('Caso', inplace=True)

print("\nDataFrame procesado para graficar (incluye Tiempos):")
print(f"Columnas: {df.columns.tolist()}")
print(df.head()) # Mostrar primeras filas

# --- Generación de Gráficos ---
timestamp = time.strftime("%Y%m%d_%H%M%S")

# --- Gráficos de COSTO (Iguales que antes, usando nuevas columnas) ---

# 1. Comparación de Costos Finales
print("\nGenerando Gráfico 1: Comparación de Costos Finales...")
fig1, ax1 = plt.subplots(figsize=(15, 8))
cols_costo_g1 = [
    'Greedy Det 1P Costo', 'HC desde Det 1P Costo',
    'Greedy Det 2P Costo', 'HC desde Det 2P Costo',
    f'GRASP {max(GRASP_ITERATIONS_LIST)}r 1P Costo',
    f'GRASP {max(GRASP_ITERATIONS_LIST)}r 2P Costo'
]
cols_exist_g1 = [col for col in cols_costo_g1 if col in df.columns]
if cols_exist_g1:
    df[cols_exist_g1].plot(kind='bar', ax=ax1, rot=0)
    ax1.set_title('Comparación de Costos Finales por Algoritmo y Caso')
    ax1.set_ylabel('Costo Total')
    ax1.set_xlabel('Caso de Prueba'); ax1.grid(axis='y', linestyle='--')
    if len(cases) > 4: plt.xticks(rotation=45, ha='right')
    ax1.legend(title='Algoritmo', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig1_path = os.path.join(OUTPUT_GRAPH_DIR, f'01_costos_finales_{timestamp}.png')
    plt.savefig(fig1_path); print(f"Gráfico guardado en: {fig1_path}")
else: plt.close(fig1); print("Gráfico 1 (Costos Finales) omitido - faltan columnas.")

# 2. Comparación Costos 1 Pista vs 2 Pistas
print("Generando Gráfico 2: Comparación Costos 1 Pista vs 2 Pistas...")
fig2, ax2 = plt.subplots(1, 3, figsize=(18, 6), sharey=True); fig2.subplots_adjust(wspace=0.1)
if not isinstance(ax2, np.ndarray): ax2 = [ax2]
metrics_g2 = ['Greedy Det', 'HC desde Det', f'GRASP {max(GRASP_ITERATIONS_LIST)}r']
plot_success_g2 = False
for i, metric in enumerate(metrics_g2):
    col_1p, col_2p = f'{metric} 1P Costo', f'{metric} 2P Costo'
    if col_1p in df.columns and col_2p in df.columns:
        df[[col_1p, col_2p]].plot(kind='bar', ax=ax2[i], rot=0, legend=False)
        ax2[i].set_title(f'{metric}'); ax2[i].set_xlabel('Caso'); ax2[i].grid(axis='y', linestyle='--')
        plot_success_g2 = True
    else: ax2[i].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax2[i].transAxes); ax2[i].set_title(f'{metric} (Datos Falt.)'); ax2[i].set_xlabel('Caso')
if plot_success_g2:
    ax2[0].set_ylabel('Costo Total'); fig2.legend(['1 Pista', '2 Pistas'], loc='upper right'); fig2.suptitle('Comparación de Costos: 1 Pista vs 2 Pistas'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2_path = os.path.join(OUTPUT_GRAPH_DIR, f'02_comparacion_pistas_costo_{timestamp}.png'); plt.savefig(fig2_path); print(f"Gráfico guardado en: {fig2_path}")
else: plt.close(fig2); print("Gráfico 2 (Costos Pistas) omitido - faltan columnas.")

# 3. Efecto Nº Restarts GRASP (Costo)
print("Generando Gráfico 3: Efecto Nº Restarts GRASP (Costo)...")
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 6), sharey=True); fig3.subplots_adjust(wspace=0.1)
if not isinstance(ax3, np.ndarray): ax3 = [ax3]
grasp_cols_1p_cost = [f'GRASP {i}r 1P Costo' for i in GRASP_ITERATIONS_LIST]
grasp_cols_2p_cost = [f'GRASP {i}r 2P Costo' for i in GRASP_ITERATIONS_LIST]
plot_success_g3 = False
cols_exist_g3_1p = [col for col in grasp_cols_1p_cost if col in df.columns]
if len(cols_exist_g3_1p) == len(grasp_cols_1p_cost): df[grasp_cols_1p_cost].plot(kind='bar', ax=ax3[0], rot=0); ax3[0].set_title('GRASP Costo - 1 Pista'); ax3[0].set_xlabel('Caso'); ax3[0].set_ylabel('Costo Total'); ax3[0].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax3[0].grid(axis='y', linestyle='--'); plot_success_g3 = True
else: ax3[0].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax3[0].transAxes); ax3[0].set_title('GRASP Costo - 1 Pista (Datos Falt.)')
cols_exist_g3_2p = [col for col in grasp_cols_2p_cost if col in df.columns]
if len(cols_exist_g3_2p) == len(grasp_cols_2p_cost): df[grasp_cols_2p_cost].plot(kind='bar', ax=ax3[1], rot=0); ax3[1].set_title('GRASP Costo - 2 Pistas'); ax3[1].set_xlabel('Caso'); ax3[1].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax3[1].grid(axis='y', linestyle='--'); plot_success_g3 = True
else: ax3[1].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax3[1].transAxes); ax3[1].set_title('GRASP Costo - 2 Pistas (Datos Falt.)')
if plot_success_g3:
    fig3.suptitle('Efecto del Número de Restarts en GRASP (Costo)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3_path = os.path.join(OUTPUT_GRAPH_DIR, f'03_efecto_restarts_grasp_costo_{timestamp}.png'); plt.savefig(fig3_path); print(f"Gráfico guardado en: {fig3_path}")
else: plt.close(fig3); print("Gráfico 3 (GRASP Costo) omitido - faltan columnas.")

# 4. Boxplot para Greedy Estocástico (Costo)
print("Generando Gráfico 4: Boxplot Greedy Estocástico (Costo)...")
fig4, ax4 = plt.subplots(1, len(cases), figsize=(5 * len(cases), 6), sharey=True); fig4.subplots_adjust(wspace=0.1)
if len(cases) == 1: ax4 = [ax4]
fig4.suptitle('Distribución Costos Greedy Estocástico (10 Runs)'); plot_success_g4 = False
for i, case_name in enumerate(cases):
    runs_1p = all_results[case_name].get('stochastic_1_runway_runs', []); runs_2p = all_results[case_name].get('stochastic_2_runway_runs', [])
    costs_1p = [r['cost'] for r in runs_1p if r.get('cost', INFINITO_COSTO) != INFINITO_COSTO]; costs_2p = [r['cost'] for r in runs_2p if r.get('cost', INFINITO_COSTO) != INFINITO_COSTO]
    data_to_plot = []; labels = []
    if costs_1p: data_to_plot.append(costs_1p); labels.append('1 Pista')
    if costs_2p: data_to_plot.append(costs_2p); labels.append('2 Pistas')
    if data_to_plot:
        plot_success_g4 = True; bp = ax4[i].boxplot(data_to_plot, patch_artist=True, labels=labels); colors = ['lightblue', 'lightgreen'];
        for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)
        ax4[i].set_title(f'{case_name}'); ax4[i].grid(axis='y', linestyle='--');
        if i == 0: ax4[i].set_ylabel('Costo Total')
    else: ax4[i].text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=ax4[i].transAxes); ax4[i].set_title(f'{case_name}')
if plot_success_g4:
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig4_path = os.path.join(OUTPUT_GRAPH_DIR, f'04_boxplot_estocastico_costo_{timestamp}.png'); plt.savefig(fig4_path); print(f"Gráfico guardado en: {fig4_path}")
else: print("No se generó Gráfico 4 (Boxplot Costo)."); plt.close(fig4)


# --- NUEVOS GRÁFICOS DE TIEMPO ---

# 5. Comparación de Tiempos de Ejecución (Barras Agrupadas)
print("Generando Gráfico 5: Comparación de Tiempos de Ejecución...")
fig5, ax5 = plt.subplots(figsize=(15, 8))
# Seleccionar columnas de tiempo relevantes
cols_tiempo_g5 = [
    'Greedy Det 1P Tiempo', 'HC desde Det 1P Tiempo',
    'Greedy Det 2P Tiempo', 'HC desde Det 2P Tiempo',
    # Tiempo TOTAL de GRASP (no por iteración)
    f'GRASP {max(GRASP_ITERATIONS_LIST)}r 1P Tiempo',
    f'GRASP {max(GRASP_ITERATIONS_LIST)}r 2P Tiempo',
    # Tiempo PROMEDIO de construcción Estocástica
    'Stoch Avg 1P Tiempo',
    'Stoch Avg 2P Tiempo'
]
cols_exist_g5 = [col for col in cols_tiempo_g5 if col in df.columns]
if cols_exist_g5:
    df[cols_exist_g5].plot(kind='bar', ax=ax5, rot=0)
    ax5.set_title('Comparación de Tiempos de Ejecución por Algoritmo y Caso')
    ax5.set_ylabel('Tiempo (s)')
    ax5.set_xlabel('Caso de Prueba')
    # Usar escala logarítmica si los tiempos varían mucho
    # ax5.set_yscale('log')
    # ax5.set_ylabel('Tiempo (s) - Escala Log')
    if len(cases) > 4: plt.xticks(rotation=45, ha='right')
    ax5.grid(axis='y', linestyle='--')
    ax5.legend(title='Algoritmo', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    fig5_path = os.path.join(OUTPUT_GRAPH_DIR, f'05_tiempos_ejecucion_{timestamp}.png')
    plt.savefig(fig5_path); print(f"Gráfico guardado en: {fig5_path}")
else: plt.close(fig5); print("Gráfico 5 (Tiempos Ejecución) omitido - faltan columnas.")

# 6. Efecto Nº Restarts GRASP (Tiempo TOTAL)
print("Generando Gráfico 6: Efecto Nº Restarts GRASP (Tiempo)...")
fig6, ax6 = plt.subplots(1, 2, figsize=(14, 6), sharey=True); fig6.subplots_adjust(wspace=0.1)
if not isinstance(ax6, np.ndarray): ax6 = [ax6]
grasp_cols_1p_time = [f'GRASP {i}r 1P Tiempo' for i in GRASP_ITERATIONS_LIST]
grasp_cols_2p_time = [f'GRASP {i}r 2P Tiempo' for i in GRASP_ITERATIONS_LIST]
plot_success_g6 = False
cols_exist_g6_1p = [col for col in grasp_cols_1p_time if col in df.columns]
if len(cols_exist_g6_1p) == len(grasp_cols_1p_time): df[grasp_cols_1p_time].plot(kind='bar', ax=ax6[0], rot=0); ax6[0].set_title('GRASP Tiempo Total - 1 Pista'); ax6[0].set_xlabel('Caso'); ax6[0].set_ylabel('Tiempo Total (s)'); ax6[0].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax6[0].grid(axis='y', linestyle='--'); plot_success_g6 = True
else: ax6[0].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax6[0].transAxes); ax6[0].set_title('GRASP Tiempo - 1 Pista (Datos Falt.)')
cols_exist_g6_2p = [col for col in grasp_cols_2p_time if col in df.columns]
if len(cols_exist_g6_2p) == len(grasp_cols_2p_time): df[grasp_cols_2p_time].plot(kind='bar', ax=ax6[1], rot=0); ax6[1].set_title('GRASP Tiempo Total - 2 Pistas'); ax6[1].set_xlabel('Caso'); ax6[1].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax6[1].grid(axis='y', linestyle='--'); plot_success_g6 = True
else: ax6[1].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax6[1].transAxes); ax6[1].set_title('GRASP Tiempo - 2 Pistas (Datos Falt.)')
if plot_success_g6:
    fig6.suptitle('Efecto del Número de Restarts en GRASP (Tiempo Total)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig6_path = os.path.join(OUTPUT_GRAPH_DIR, f'06_efecto_restarts_grasp_tiempo_{timestamp}.png'); plt.savefig(fig6_path); print(f"Gráfico guardado en: {fig6_path}")
else: plt.close(fig6); print("Gráfico 6 (GRASP Tiempo) omitido - faltan columnas.")


print("\n--- Generación de gráficos completada ---")