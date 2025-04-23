# graphics.py
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os
import time
import shutil

# --- Constantes y Configuración ---
# *** USAR RUTAS CORREGIDAS ***
JSON_INPUT_FILENAME = 'results/all_results_details_final.json' # <-- NOMBRE FINAL CONSISTENTE
OUTPUT_GRAPH_DIR = 'results/graficos_resultados'          # <-- RUTA CORREGIDA
INFINITO_COSTO = float('inf')
# Asegúrate que estas listas coincidan con main.py
GRASP_ITERATIONS_LIST = [10, 25, 50]
SA_INITIAL_TEMPS = [10000, 5000, 1000, 500, 100]
REPRESENTATIVE_SA_TEMP = SA_INITIAL_TEMPS[2] # Ejemplo: 1000

if os.path.exists(OUTPUT_GRAPH_DIR):
    try: shutil.rmtree(OUTPUT_GRAPH_DIR); print(f"Directorio '{OUTPUT_GRAPH_DIR}' borrado.")
    except OSError as e: print(f"Error borrando '{OUTPUT_GRAPH_DIR}': {e}")
try: os.makedirs(OUTPUT_GRAPH_DIR); print(f"Directorio '{OUTPUT_GRAPH_DIR}' creado.")
except OSError as e: print(f"Error creando '{OUTPUT_GRAPH_DIR}': {e}"); exit()

# --- Funciones Auxiliares ---
def get_stats_from_runs(run_list, key='cost'):
    if not isinstance(run_list, list): run_list = []
    values = [r[key] for r in run_list if isinstance(r, dict) and r.get('cost', INFINITO_COSTO) != INFINITO_COSTO and key in r]
    if not values: return {'min': None, 'avg': None, 'max': None, 'stdev': None, 'valid_count': 0}
    valid_count = len(values); return {'min': min(values), 'avg': np.mean(values), 'max': max(values), 'stdev': np.std(values) if valid_count > 1 else 0.0, 'valid_count': valid_count}

# --- Cargar Datos ---
try:
    with open(JSON_INPUT_FILENAME, 'r') as f: all_results = json.load(f)
    print(f"Datos cargados exitosamente desde '{JSON_INPUT_FILENAME}'.")
except FileNotFoundError: print(f"Error: '{JSON_INPUT_FILENAME}' no encontrado."); exit()
except json.JSONDecodeError as e: print(f"Error: '{JSON_INPUT_FILENAME}' no es JSON válido. {e}"); exit()
except Exception as e: print(f"Error inesperado cargando JSON: {e}"); exit()

# --- Procesar Datos para Graficar ---
plot_data = []
cases = sorted(all_results.keys())
for case_name in cases:
    if not isinstance(all_results.get(case_name), dict): print(f"Advertencia: Entrada inválida para '{case_name}', saltando caso."); continue
    case_results = all_results[case_name]
    data_entry = {'Caso': case_name}
    def get_res(key): return case_results.get(key, {})
    def get_cost(key): return get_res(key).get('cost', INFINITO_COSTO)
    def get_time(key): return get_res(key).get('time', 0.0)
    # Determinista & HC (Usando claves consistentes)
    data_entry['Greedy Det 1P Costo'] = get_cost('deterministic_1_runway'); data_entry['Greedy Det 1P Tiempo'] = get_time('deterministic_1_runway')
    data_entry['HC desde Det 1P Costo'] = get_cost('hc_from_deterministic_1_runway'); data_entry['HC desde Det 1P Tiempo'] = get_time('hc_from_deterministic_1_runway')
    data_entry['Greedy Det 2P Costo'] = get_cost('deterministic_2_runways'); data_entry['Greedy Det 2P Tiempo'] = get_time('deterministic_2_runways')
    data_entry['HC desde Det 2P Costo'] = get_cost('hc_from_deterministic_2_runways'); data_entry['HC desde Det 2P Tiempo'] = get_time('hc_from_deterministic_2_runways')
    # Estocástico (Stats)
    stoch_runs_1r = get_res('stochastic_1_runway_runs'); stoch_runs_2r = get_res('stochastic_2_runway_runs')
    stoch_1r_cost_stats = get_stats_from_runs(stoch_runs_1r, 'cost'); stoch_1r_time_stats = get_stats_from_runs(stoch_runs_1r, 'time')
    stoch_2r_cost_stats = get_stats_from_runs(stoch_runs_2r, 'cost'); stoch_2r_time_stats = get_stats_from_runs(stoch_runs_2r, 'time')
    data_entry['Stoch Min 1P Costo'] = stoch_1r_cost_stats['min']; data_entry['Stoch Avg 1P Tiempo'] = stoch_1r_time_stats['avg']
    data_entry['Stoch Min 2P Costo'] = stoch_2r_cost_stats['min']; data_entry['Stoch Avg 2P Tiempo'] = stoch_2r_time_stats['avg']
    # GRASP
    for iters in GRASP_ITERATIONS_LIST: data_entry[f'GRASP {iters}r 1P Costo'] = get_cost(f'grasp_stochastic_{iters}iters_1_runway'); data_entry[f'GRASP {iters}r 1P Tiempo'] = get_time(f'grasp_stochastic_{iters}iters_1_runway'); data_entry[f'GRASP {iters}r 2P Costo'] = get_cost(f'grasp_stochastic_{iters}iters_2_runways'); data_entry[f'GRASP {iters}r 2P Tiempo'] = get_time(f'grasp_stochastic_{iters}iters_2_runways')
    # SA
    for T_init in SA_INITIAL_TEMPS: tag = f"T{T_init}"; data_entry[f'SA (Det {tag}) 1P Costo'] = get_cost(f'sa_{tag}_from_det_1r'); data_entry[f'SA (Det {tag}) 1P Tiempo'] = get_time(f'sa_{tag}_from_det_1r'); data_entry[f'SA (Det {tag}) 2P Costo'] = get_cost(f'sa_{tag}_from_det_2r'); data_entry[f'SA (Det {tag}) 2P Tiempo'] = get_time(f'sa_{tag}_from_det_2r'); data_entry[f'SA (Stoch {tag}) 1P Costo'] = get_cost(f'sa_{tag}_from_best_stoch_1r'); data_entry[f'SA (Stoch {tag}) 1P Tiempo'] = get_time(f'sa_{tag}_from_best_stoch_1r'); data_entry[f'SA (Stoch {tag}) 2P Costo'] = get_cost(f'sa_{tag}_from_best_stoch_2r'); data_entry[f'SA (Stoch {tag}) 2P Tiempo'] = get_time(f'sa_{tag}_from_best_stoch_2r')
    plot_data.append(data_entry)

# Crear DataFrame
if not plot_data: print("Error: No se procesaron datos. Saliendo."); exit()
df = pd.DataFrame(plot_data)
df.replace(INFINITO_COSTO, np.nan, inplace=True); df.fillna(np.nan, inplace=True); df.set_index('Caso', inplace=True)

print("\nDataFrame procesado para graficar (incluye SA):")
print(f"Columnas: {df.columns.tolist()}")
print(df.to_string())

# --- Generación de Gráficos ---
timestamp = time.strftime("%Y%m%d_%H%M%S"); plt.style.use('seaborn-v0_8-darkgrid')

# --- Gráficos de COSTO ---
# 1. Comparación Costos Finales
print("\nGenerando Gráfico 1: Comparación de Costos Finales...")
fig1, ax1 = plt.subplots(figsize=(17, 9)); rep_grasp_iters = max(GRASP_ITERATIONS_LIST); rep_sa_temp = REPRESENTATIVE_SA_TEMP
# *** Nombres de Columna del DataFrame a usar ***
cols_costo_g1 = [ 'Greedy Det 1P Costo', 'HC desde Det 1P Costo', f'GRASP {rep_grasp_iters}r 1P Costo', f'SA (Det T{rep_sa_temp}) 1P Costo', f'SA (Stoch T{rep_sa_temp}) 1P Costo', 'Greedy Det 2P Costo', 'HC desde Det 2P Costo', f'GRASP {rep_grasp_iters}r 2P Costo', f'SA (Det T{rep_sa_temp}) 2P Costo', f'SA (Stoch T{rep_sa_temp}) 2P Costo' ]
rename_map_g1 = { 'Greedy Det 1P Costo': 'GDet-1P', 'HC desde Det 1P Costo': 'HC(Det)-1P', f'GRASP {rep_grasp_iters}r 1P Costo': f'GRASP{rep_grasp_iters}-1P', f'SA (Det T{rep_sa_temp}) 1P Costo': f'SA(Det T{rep_sa_temp})-1P', f'SA (Stoch T{rep_sa_temp}) 1P Costo': f'SA(Stoch T{rep_sa_temp})-1P', 'Greedy Det 2P Costo': 'GDet-2P', 'HC desde Det 2P Costo': 'HC(Det)-2P', f'GRASP {rep_grasp_iters}r 2P Costo': f'GRASP{rep_grasp_iters}-2P', f'SA (Det T{rep_sa_temp}) 2P Costo': f'SA(Det T{rep_sa_temp})-2P', f'SA (Stoch T{rep_sa_temp}) 2P Costo': f'SA(Stoch T{rep_sa_temp})-2P' }
cols_exist_g1 = [col for col in cols_costo_g1 if col in df.columns]
if cols_exist_g1: df_subset_g1 = df[cols_exist_g1].rename(columns={k:v for k,v in rename_map_g1.items() if k in cols_exist_g1}); df_subset_g1.plot(kind='bar', ax=ax1, rot=0, width=0.8); ax1.set_title('Comparación de Costos Finales por Algoritmo y Caso'); ax1.set_ylabel('Costo Total'); ax1.set_xlabel('Caso de Prueba'); ax1.grid(axis='y', linestyle='--'); ax1.tick_params(axis='x', rotation=0); ax1.legend(title='Algoritmo - Pistas', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small'); ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f')); plt.tight_layout(rect=[0, 0, 0.80, 1]); fig1_path = os.path.join(OUTPUT_GRAPH_DIR, f'01_costos_finales_con_sa_{timestamp}.png'); plt.savefig(fig1_path); print(f"Gráfico guardado en: {fig1_path}")
else: plt.close(fig1); print("Gráfico 1 (Costos Finales) omitido - faltan columnas.")

# 2. Comparación Costos 1P vs 2P
print("Generando Gráfico 2: Comparación Costos 1 Pista vs 2 Pistas...")
fig2, ax2 = plt.subplots(1, 4, figsize=(20, 5), sharey=True); fig2.subplots_adjust(wspace=0.05); plot_success_g2 = False;
if not isinstance(ax2, np.ndarray): ax2 = [ax2]
# *** Nombres de Columna del DataFrame a usar ***
metrics_g2 = ['Greedy Det', 'HC desde Det', f'GRASP {max(GRASP_ITERATIONS_LIST)}r', f'SA (Det T{REPRESENTATIVE_SA_TEMP})']
for i, metric in enumerate(metrics_g2):
    col_1p, col_2p = f'{metric} 1P Costo', f'{metric} 2P Costo' # Nombres correctos
    if col_1p in df.columns and col_2p in df.columns: df[[col_1p, col_2p]].plot(kind='bar', ax=ax2[i], rot=0, legend=False, width=0.8); ax2[i].set_title(f'{metric}'); ax2[i].set_xlabel('Caso'); ax2[i].grid(axis='y', linestyle='--'); ax2[i].tick_params(axis='x', rotation=0); plot_success_g2 = True
    else: ax2[i].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax2[i].transAxes); ax2[i].set_title(f'{metric} (Datos Falt.)'); ax2[i].set_xlabel('Caso')
if plot_success_g2: ax2[0].set_ylabel('Costo Total'); fig2.legend(['1 Pista', '2 Pistas'], loc='upper right'); fig2.suptitle('Comparación de Costos: 1 Pista vs 2 Pistas'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig2_path = os.path.join(OUTPUT_GRAPH_DIR, f'02_comparacion_pistas_costo_con_sa_{timestamp}.png'); plt.savefig(fig2_path); print(f"Gráfico guardado en: {fig2_path}")
else: plt.close(fig2); print("Gráfico 2 (Costos Pistas) omitido - faltan columnas.")

# 3. Efecto Restarts GRASP (Costo)
# ... (Código gráfico 3 sin cambios) ...
print("Generando Gráfico 3: Efecto Nº Restarts GRASP (Costo)...")
fig3, ax3 = plt.subplots(1, 2, figsize=(14, 6), sharey=True); fig3.subplots_adjust(wspace=0.1); plot_success_g3 = False; grasp_cols_1p_cost = [f'GRASP {i}r 1P Costo' for i in GRASP_ITERATIONS_LIST]; grasp_cols_2p_cost = [f'GRASP {i}r 2P Costo' for i in GRASP_ITERATIONS_LIST]
if not isinstance(ax3, np.ndarray): ax3 = [ax3]
cols_exist_g3_1p = [col for col in grasp_cols_1p_cost if col in df.columns]; cols_exist_g3_2p = [col for col in grasp_cols_2p_cost if col in df.columns]
if len(cols_exist_g3_1p) == len(grasp_cols_1p_cost): df[grasp_cols_1p_cost].plot(kind='bar', ax=ax3[0], rot=0); ax3[0].set_title('GRASP Costo - 1 Pista'); ax3[0].set_xlabel('Caso'); ax3[0].set_ylabel('Costo Total'); ax3[0].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax3[0].grid(axis='y', linestyle='--'); plot_success_g3 = True
else: ax3[0].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax3[0].transAxes); ax3[0].set_title('GRASP Costo - 1 Pista (Datos Falt.)')
if len(cols_exist_g3_2p) == len(grasp_cols_2p_cost): df[grasp_cols_2p_cost].plot(kind='bar', ax=ax3[1], rot=0); ax3[1].set_title('GRASP Costo - 2 Pistas'); ax3[1].set_xlabel('Caso'); ax3[1].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax3[1].grid(axis='y', linestyle='--'); plot_success_g3 = True
else: ax3[1].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax3[1].transAxes); ax3[1].set_title('GRASP Costo - 2 Pistas (Datos Falt.)')
if plot_success_g3: fig3.suptitle('Efecto del Número de Restarts en GRASP (Costo)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig3_path = os.path.join(OUTPUT_GRAPH_DIR, f'03_efecto_restarts_grasp_costo_{timestamp}.png'); plt.savefig(fig3_path); print(f"Gráfico guardado en: {fig3_path}")
else: plt.close(fig3); print("Gráfico 3 (GRASP Costo) omitido - faltan columnas.")

# 4. Boxplot Greedy Estocástico (Costo)
print("Generando Gráfico 4: Boxplot Greedy Estocástico (Costo)...")
fig4, ax4 = plt.subplots(1, len(cases), figsize=(5 * len(cases), 6), sharey=True); fig4.subplots_adjust(wspace=0.1); plot_success_g4 = False
if len(cases) == 1: ax4 = [ax4] # Asegurar que sea iterable si solo hay 1 caso
fig4.suptitle('Distribución Costos Greedy Estocástico (10 Runs)');
for i, case_name in enumerate(cases):
    runs_1p = all_results[case_name].get('stochastic_1_runway_runs', [])
    runs_2p = all_results[case_name].get('stochastic_2_runway_runs', [])
    costs_1p = [r['cost'] for r in runs_1p if isinstance(r, dict) and r.get('cost', INFINITO_COSTO) != INFINITO_COSTO]
    costs_2p = [r['cost'] for r in runs_2p if isinstance(r, dict) and r.get('cost', INFINITO_COSTO) != INFINITO_COSTO]
    # print(f"  Debug Boxplot {case_name}: Costs 1P = {costs_1p}") # Mantener si persiste problema
    # print(f"  Debug Boxplot {case_name}: Costs 2P = {costs_2p}") # Mantener si persiste problema
    data_to_plot = []; labels = []
    # Añadir datos SÓLO si la lista de costos no está vacía
    if costs_1p: data_to_plot.append(costs_1p); labels.append('1 Pista')
    if costs_2p: data_to_plot.append(costs_2p); labels.append('2 Pistas')
    if data_to_plot:
        plot_success_g4 = True
        try:
            # Usar tick_labels en lugar de labels si matplotlib >= 3.9
            bp = ax4[i].boxplot(data_to_plot, patch_artist=True, labels=labels)
            # O: bp = ax4[i].boxplot(data_to_plot, patch_artist=True, tick_labels=labels)
            colors = ['lightblue', 'lightgreen'];
            [patch.set_facecolor(color) for patch, color in zip(bp['boxes'], colors)]
            ax4[i].set_title(f'{case_name}'); ax4[i].grid(axis='y', linestyle='--');
        except Exception as e_bp: print(f"Error generando boxplot para {case_name}: {e_bp}"); ax4[i].text(0.5, 0.5, 'Error en Boxplot', ha='center', va='center', transform=ax4[i].transAxes); ax4[i].set_title(f'{case_name}')
    else: ax4[i].text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=ax4[i].transAxes); ax4[i].set_title(f'{case_name}')
    if i == 0 and plot_success_g4: ax4[i].set_ylabel('Costo Total')
if plot_success_g4: plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig4_path = os.path.join(OUTPUT_GRAPH_DIR, f'04_boxplot_estocastico_costo_{timestamp}.png'); plt.savefig(fig4_path); print(f"Gráfico guardado en: {fig4_path}")
else: print("No se generó Gráfico 4 (Boxplot Costo)."); plt.close(fig4)

# 5. Efecto Temp SA (Costo)
# ... (Código gráfico 5 sin cambios) ...
print("Generando Gráfico 5: Efecto Temperatura Inicial SA (Costo)...")
fig5, ax5 = plt.subplots(2, 2, figsize=(16, 10), sharey=True); fig5.subplots_adjust(hspace=0.3, wspace=0.1); plot_success_g5 = False; start_points = [('Det', 'Desde Det.'), ('Stoch', 'Desde Best Stoch')]; runways = [(1, '1 Pista'), (2, '2 Pistas')]
if not isinstance(ax5, np.ndarray): ax5 = ax5.reshape(1,1)
for row, (start_key, start_label) in enumerate(start_points):
    for col, (rw_num, rw_label) in enumerate(runways):
        ax = ax5[row, col]; sa_cost_cols = [f'SA ({start_key} T{T}) {rw_num}P Costo' for T in SA_INITIAL_TEMPS]; cols_exist_g5 = [c for c in sa_cost_cols if c in df.columns]
        if len(cols_exist_g5) == len(sa_cost_cols): df[sa_cost_cols].plot(kind='bar', ax=ax, rot=0, legend=False); ax.set_title(f'SA Costo ({start_label} - {rw_label})'); ax.set_xlabel('Caso'); ax.grid(axis='y', linestyle='--'); ax.set_xticklabels(df.index, rotation=0); ax.legend([f'T={T}' for T in SA_INITIAL_TEMPS], fontsize='small'); plot_success_g5 = True
        else: ax.text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax.transAxes); ax.set_title(f'SA Costo ({start_label} - {rw_label}) (Datos Falt.)')
        if col == 0: ax.set_ylabel('Costo Total (Mejor Encontrado)')
if plot_success_g5: fig5.suptitle('Efecto de la Temperatura Inicial en Simulated Annealing (Costo)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig5_path = os.path.join(OUTPUT_GRAPH_DIR, f'05_efecto_temp_sa_costo_{timestamp}.png'); plt.savefig(fig5_path); print(f"Gráfico guardado en: {fig5_path}")
else: plt.close(fig5); print("Gráfico 5 (SA Costo) omitido - faltan columnas.")

# --- Gráficos de TIEMPO ---

# 6. Comparación Tiempos Ejecución
print("Generando Gráfico 6: Comparación de Tiempos de Ejecución...")
fig6, ax6 = plt.subplots(figsize=(17, 9))
# *** USAR NOMBRES DE COLUMNA CONSISTENTES ***
cols_tiempo_g6 = [ 'Greedy Det 1P Tiempo', 'HC desde Det 1P Tiempo', f'GRASP {max(GRASP_ITERATIONS_LIST)}r 1P Tiempo', f'SA (Det T{REPRESENTATIVE_SA_TEMP}) 1P Tiempo', f'SA (Stoch T{REPRESENTATIVE_SA_TEMP}) 1P Tiempo', 'Greedy Det 2P Tiempo', 'HC desde Det 2P Tiempo', f'GRASP {max(GRASP_ITERATIONS_LIST)}r 2P Tiempo', f'SA (Det T{REPRESENTATIVE_SA_TEMP}) 2P Tiempo', f'SA (Stoch T{REPRESENTATIVE_SA_TEMP}) 2P Tiempo', 'Stoch Avg 1P Tiempo', 'Stoch Avg 2P Tiempo']
# *** MAPA DE RENOMBRAMIENTO CONSISTENTE ***
rename_map_g6 = {'Greedy Det 1P Tiempo': 'GDet-1P', 'HC desde Det 1P Tiempo': 'HC(Det)-1P', f'GRASP {max(GRASP_ITERATIONS_LIST)}r 1P Tiempo': f'GRASP{max(GRASP_ITERATIONS_LIST)}-1P', f'SA (Det T{REPRESENTATIVE_SA_TEMP}) 1P Tiempo': f'SA(Det T{REPRESENTATIVE_SA_TEMP})-1P', f'SA (Stoch T{REPRESENTATIVE_SA_TEMP}) 1P Tiempo': f'SA(Stoch T{REPRESENTATIVE_SA_TEMP})-1P', 'Greedy Det 2P Tiempo': 'GDet-2P', 'HC desde Det 2P Tiempo': 'HC(Det)-2P', f'GRASP {max(GRASP_ITERATIONS_LIST)}r 2P Tiempo': f'GRASP{max(GRASP_ITERATIONS_LIST)}-2P', f'SA (Det T{REPRESENTATIVE_SA_TEMP}) 2P Tiempo': f'SA(Det T{REPRESENTATIVE_SA_TEMP})-2P', f'SA (Stoch T{REPRESENTATIVE_SA_TEMP}) 2P Tiempo': f'SA(Stoch T{REPRESENTATIVE_SA_TEMP})-2P', 'Stoch Avg 1P Tiempo': 'StochAvg-1P', 'Stoch Avg 2P Tiempo': 'StochAvg-2P'}
cols_exist_g6 = [col for col in cols_tiempo_g6 if col in df.columns]
if cols_exist_g6: df_subset_g6 = df[cols_exist_g6].rename(columns={k:v for k,v in rename_map_g6.items() if k in cols_exist_g6}); df_subset_g6.plot(kind='bar', ax=ax6, rot=0, width=0.8); ax6.set_title('Comparación de Tiempos de Ejecución por Algoritmo y Caso'); ax6.set_ylabel('Tiempo (s)'); ax6.set_xlabel('Caso de Prueba'); ax6.grid(axis='y', linestyle='--'); ax6.tick_params(axis='x', rotation=0); ax6.legend(title='Algoritmo - Pistas', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small'); plt.tight_layout(rect=[0, 0, 0.80, 1]); fig6_path = os.path.join(OUTPUT_GRAPH_DIR, f'06_tiempos_ejecucion_con_sa_{timestamp}.png'); plt.savefig(fig6_path); print(f"Gráfico guardado en: {fig6_path}")
else: plt.close(fig6); print("Gráfico 6 (Tiempos Ejecución) omitido - faltan columnas.")

# 7. Efecto Restarts GRASP (Tiempo)
# ... (Código gráfico 7 sin cambios) ...
print("Generando Gráfico 7: Efecto Nº Restarts GRASP (Tiempo)...")
fig7, ax7 = plt.subplots(1, 2, figsize=(14, 6), sharey=True); fig7.subplots_adjust(wspace=0.1); plot_success_g7 = False
if not isinstance(ax7, np.ndarray): ax7 = [ax7]
grasp_cols_1p_time = [f'GRASP {i}r 1P Tiempo' for i in GRASP_ITERATIONS_LIST]; grasp_cols_2p_time = [f'GRASP {i}r 2P Tiempo' for i in GRASP_ITERATIONS_LIST]
cols_exist_g7_1p = [col for col in grasp_cols_1p_time if col in df.columns]; cols_exist_g7_2p = [col for col in grasp_cols_2p_time if col in df.columns]
if len(cols_exist_g7_1p) == len(grasp_cols_1p_time): df[grasp_cols_1p_time].plot(kind='bar', ax=ax7[0], rot=0); ax7[0].set_title('GRASP Tiempo Total - 1 Pista'); ax7[0].set_xlabel('Caso'); ax7[0].set_ylabel('Tiempo Total (s)'); ax7[0].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax7[0].grid(axis='y', linestyle='--'); plot_success_g7 = True
else: ax7[0].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax7[0].transAxes); ax7[0].set_title('GRASP Tiempo - 1 Pista (Datos Falt.)')
if len(cols_exist_g7_2p) == len(grasp_cols_2p_time): df[grasp_cols_2p_time].plot(kind='bar', ax=ax7[1], rot=0); ax7[1].set_title('GRASP Tiempo Total - 2 Pistas'); ax7[1].set_xlabel('Caso'); ax7[1].legend([f'{i} R.' for i in GRASP_ITERATIONS_LIST]); ax7[1].grid(axis='y', linestyle='--'); plot_success_g7 = True
else: ax7[1].text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax7[1].transAxes); ax7[1].set_title('GRASP Tiempo - 2 Pistas (Datos Falt.)')
if plot_success_g7: fig7.suptitle('Efecto del Número de Restarts en GRASP (Tiempo Total)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig7_path = os.path.join(OUTPUT_GRAPH_DIR, f'07_efecto_restarts_grasp_tiempo_{timestamp}.png'); plt.savefig(fig7_path); print(f"Gráfico guardado en: {fig7_path}")
else: plt.close(fig7); print("Gráfico 7 (GRASP Tiempo) omitido - faltan columnas.")

# 8. Efecto Temperatura Inicial SA (Tiempo)
# ... (Código gráfico 8 sin cambios) ...
print("Generando Gráfico 8: Efecto Temperatura Inicial SA (Tiempo)...")
fig8, ax8 = plt.subplots(2, 2, figsize=(16, 10), sharey=False); fig8.subplots_adjust(hspace=0.3, wspace=0.15); plot_success_g8 = False
start_points = [('Det', 'Desde Det.'), ('Stoch', 'Desde Best Stoch')]; runways = [(1, '1 Pista'), (2, '2 Pistas')]
if not isinstance(ax8, np.ndarray): ax8 = ax8.reshape(1,1)
for row, (start_key, start_label) in enumerate(start_points):
    for col, (rw_num, rw_label) in enumerate(runways):
        ax = ax8[row, col]; sa_time_cols = [f'SA ({start_key} T{T}) {rw_num}P Tiempo' for T in SA_INITIAL_TEMPS]; cols_exist_g8 = [c for c in sa_time_cols if c in df.columns]
        if len(cols_exist_g8) == len(sa_time_cols): df[sa_time_cols].plot(kind='bar', ax=ax, rot=0, legend=False); ax.set_title(f'SA Tiempo ({start_label} - {rw_label})'); ax.set_xlabel('Caso'); ax.grid(axis='y', linestyle='--'); ax.set_xticklabels(df.index, rotation=0); ax.legend([f'T={T}' for T in SA_INITIAL_TEMPS], fontsize='small'); plot_success_g8 = True;
        else: ax.text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax.transAxes); ax.set_title(f'SA Tiempo ({start_label} - {rw_label}) (Datos Falt.)')
        if col == 0: ax.set_ylabel('Tiempo Ejecución SA (s)')
if plot_success_g8: fig8.suptitle('Efecto de la Temperatura Inicial en Simulated Annealing (Tiempo Ejecución)'); plt.tight_layout(rect=[0, 0.03, 1, 0.95]); fig8_path = os.path.join(OUTPUT_GRAPH_DIR, f'08_efecto_temp_sa_tiempo_{timestamp}.png'); plt.savefig(fig8_path); print(f"Gráfico guardado en: {fig8_path}")
else: plt.close(fig8); print("Gráfico 8 (SA Tiempo) omitido - faltan columnas.")

print("\n--- Generación de gráficos completada ---")