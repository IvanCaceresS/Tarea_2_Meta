import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
import os
import time
import shutil
import statistics
import copy

JSON_INPUT_FILENAME = 'results/all_results_details.json'
OUTPUT_GRAPH_DIR = 'results/graficos_resultados'
INFINITO_COSTO = float('inf')

NUM_STOCHASTIC_RUNS = 10
HC_FROM_STOCH_LABELS = [10, 25, 50]
SA_INITIAL_TEMPS = [10000, 5000, 1000, 500, 100]

HC_FROM_STOCH_LABEL_REPRESENTATIVE = HC_FROM_STOCH_LABELS[0]
REPRESENTATIVE_SA_TEMP = SA_INITIAL_TEMPS[2]

if os.path.exists(OUTPUT_GRAPH_DIR):
    print(f"Directorio '{OUTPUT_GRAPH_DIR}' ya existe. Intentando borrar...")
    try:
        shutil.rmtree(OUTPUT_GRAPH_DIR)
        print(f"Directorio '{OUTPUT_GRAPH_DIR}' borrado.")
    except OSError as e:
        print(f"Error borrando '{OUTPUT_GRAPH_DIR}': {e}. Los gráficos pueden sobreescribirse o fallar.")
try:
    os.makedirs(OUTPUT_GRAPH_DIR)
    print(f"Directorio '{OUTPUT_GRAPH_DIR}' creado.")
except OSError as e:
    print(f"Error creando '{OUTPUT_GRAPH_DIR}': {e}. Saliendo.")
    exit()

def get_stats_from_list(results_list, cost_key='cost', time_key='time'):
    if not isinstance(results_list, list): 
        results_list = []
    stats = {
        'costs': [r.get(cost_key, INFINITO_COSTO) for r in results_list],
        'times': [r.get(time_key, 0) for r in results_list],
        'valid_runs': 0,
        'invalid_runs': 0,
        'min_cost': INFINITO_COSTO,
        'max_cost': INFINITO_COSTO,
        'avg_cost': INFINITO_COSTO,
        'stdev_cost': 0.0,
        'avg_time': 0.0,
        'total_time': 0.0,
        'best_result': None
    }
    valid_costs = [c for c in stats['costs'] if c is not None and c != INFINITO_COSTO]
    stats['valid_runs'] = len(valid_costs)
    stats['invalid_runs'] = len(results_list) - stats['valid_runs']

    if stats['valid_runs'] > 0:
        stats['min_cost'] = min(valid_costs)
        stats['max_cost'] = max(valid_costs)
        stats['avg_cost'] = statistics.mean(valid_costs)
        if stats['valid_runs'] > 1:
            try:
                stats['stdev_cost'] = statistics.stdev(valid_costs)
            except statistics.StatisticsError:
                 stats['stdev_cost'] = 0.0

        best_res_index = -1
        current_min = INFINITO_COSTO
        for i, res in enumerate(results_list):
            cost = res.get(cost_key, INFINITO_COSTO)
            if cost is not None and cost != INFINITO_COSTO:
                if cost < current_min:
                    current_min = cost
                    best_res_index = i
                elif cost == current_min and best_res_index == -1:
                     best_res_index = i
        if best_res_index != -1:
             stats['best_result'] = copy.deepcopy(results_list[best_res_index]) 

    valid_times = [t for t in stats['times'] if t is not None]
    if valid_times:
         stats['avg_time'] = statistics.mean(valid_times) if valid_times else 0.0
         stats['total_time'] = sum(valid_times)

    return stats

try:
    with open(JSON_INPUT_FILENAME, 'r') as f:
        all_results = json.load(f)
    print(f"Datos cargados exitosamente desde '{JSON_INPUT_FILENAME}'.")
except FileNotFoundError:
    print(f"Error: '{JSON_INPUT_FILENAME}' no encontrado."); exit()
except json.JSONDecodeError as e:
    print(f"Error: '{JSON_INPUT_FILENAME}' no es JSON válido. {e}"); exit()
except Exception as e:
    print(f"Error inesperado cargando JSON: {e}"); exit()

plot_data = []
stats_hc_stoch = {} 
stats_sa_stoch = {} 

cases = sorted(all_results.keys())
for case_name in cases:
    if not isinstance(all_results.get(case_name), dict):
        print(f"Advertencia: Entrada inválida para '{case_name}', saltando caso.")
        continue
    case_results = all_results[case_name]
    data_entry = {'Caso': case_name}
    stats_hc_stoch[case_name] = {}
    stats_sa_stoch[case_name] = {}

    def get_res(key): return case_results.get(key, {})
    def get_cost(key): return get_res(key).get('cost', INFINITO_COSTO)
    def get_time(key): return get_res(key).get('time', 0.0)
    def get_list(key): return case_results.get(key, []) 

    # 1. Determinista y HC desde Determinista
    data_entry['GDet 1P Costo'] = get_cost('deterministic_1_runway')
    data_entry['GDet 1P Tiempo'] = get_time('deterministic_1_runway')
    data_entry['HC(Det) 1P Costo'] = get_cost('hc_from_deterministic_1_runway')
    data_entry['HC(Det) 1P Tiempo'] = get_time('hc_from_deterministic_1_runway')
    data_entry['GDet 2P Costo'] = get_cost('deterministic_2_runways')
    data_entry['GDet 2P Tiempo'] = get_time('deterministic_2_runways')
    data_entry['HC(Det) 2P Costo'] = get_cost('hc_from_deterministic_2_runways')
    data_entry['HC(Det) 2P Tiempo'] = get_time('hc_from_deterministic_2_runways')

    # 2. Estocástico (Stats de las 10 runs base)
    stoch_runs_1r = get_list('stochastic_1_runway_runs')
    stoch_runs_2r = get_list('stochastic_2_runway_runs')
    stoch_1r_stats = get_stats_from_list(stoch_runs_1r)
    stoch_2r_stats = get_stats_from_list(stoch_runs_2r)
    data_entry['Stoch Min 1P Costo'] = stoch_1r_stats['min_cost'] 
    data_entry['Stoch Avg 1P Tiempo'] = stoch_1r_stats['avg_time']
    data_entry['Stoch Min 2P Costo'] = stoch_2r_stats['min_cost'] 
    data_entry['Stoch Avg 2P Tiempo'] = stoch_2r_stats['avg_time']

    # 3. HC desde Estocástico (Calcular stats y guardar avg/min para el DataFrame)
    hc_stoch_all_labels_1r = []
    hc_stoch_all_labels_2r = []
    # Usa la constante HC_FROM_STOCH_LABELS definida globalmente
    for label in HC_FROM_STOCH_LABELS:
        hc_runs_1r = get_list(f'hc_from_stochastic_{label}_1r_runs')
        hc_runs_2r = get_list(f'hc_from_stochastic_{label}_2r_runs')
        # Calcular y guardar stats completas para gráficos detallados
        stats_hc_stoch[case_name][f'{label}_1r'] = get_stats_from_list(hc_runs_1r)
        stats_hc_stoch[case_name][f'{label}_2r'] = get_stats_from_list(hc_runs_2r)
        # Acumular costos mínimos y tiempos promedio para el DataFrame (usando la label representativa)
        if label == HC_FROM_STOCH_LABEL_REPRESENTATIVE:
            data_entry[f'HC(Stoch L{label}) Min 1P Costo'] = stats_hc_stoch[case_name][f'{label}_1r']['min_cost']
            data_entry[f'HC(Stoch L{label}) Avg 1P Tiempo'] = stats_hc_stoch[case_name][f'{label}_1r']['avg_time']
            data_entry[f'HC(Stoch L{label}) Min 2P Costo'] = stats_hc_stoch[case_name][f'{label}_2r']['min_cost']
            data_entry[f'HC(Stoch L{label}) Avg 2P Tiempo'] = stats_hc_stoch[case_name][f'{label}_2r']['avg_time']

    # 4. SA desde Determinista
    for T_init in SA_INITIAL_TEMPS:
        tag = f"T{T_init}"
        data_entry[f'SA(Det {tag}) 1P Costo'] = get_cost(f'sa_{tag}_from_det_1r')
        data_entry[f'SA(Det {tag}) 1P Tiempo'] = get_time(f'sa_{tag}_from_det_1r')
        data_entry[f'SA(Det {tag}) 2P Costo'] = get_cost(f'sa_{tag}_from_det_2r')
        data_entry[f'SA(Det {tag}) 2P Tiempo'] = get_time(f'sa_{tag}_from_det_2r')

    # 5. SA desde Estocástico (Calcular stats y guardar avg/min para el DataFrame)
    stats_sa_stoch[case_name] = {T: {} for T in SA_INITIAL_TEMPS}
    for T_init in SA_INITIAL_TEMPS:
        tag = f"T{T_init}"
        sa_runs_1r = get_list(f'sa_{tag}_from_stochastic_1r_runs')
        sa_runs_2r = get_list(f'sa_{tag}_from_stochastic_2r_runs')
         # Calcular y guardar stats completas
        stats_sa_stoch[case_name][T_init]['1r'] = get_stats_from_list(sa_runs_1r)
        stats_sa_stoch[case_name][T_init]['2r'] = get_stats_from_list(sa_runs_2r)
        # Guardar min costo y avg tiempo para el DataFrame (para temp representativa)
        if T_init == REPRESENTATIVE_SA_TEMP:
            data_entry[f'SA(Stoch {tag}) Min 1P Costo'] = stats_sa_stoch[case_name][T_init]['1r']['min_cost']
            data_entry[f'SA(Stoch {tag}) Avg 1P Tiempo'] = stats_sa_stoch[case_name][T_init]['1r']['avg_time']
            data_entry[f'SA(Stoch {tag}) Min 2P Costo'] = stats_sa_stoch[case_name][T_init]['2r']['min_cost']
            data_entry[f'SA(Stoch {tag}) Avg 2P Tiempo'] = stats_sa_stoch[case_name][T_init]['2r']['avg_time']

    plot_data.append(data_entry)

# Crear DataFrame final
if not plot_data:
    print("Error: No se procesaron datos para el DataFrame. Saliendo.")
    exit()
df = pd.DataFrame(plot_data)
df.replace(INFINITO_COSTO, np.nan, inplace=True)
df.set_index('Caso', inplace=True)

print("\nDataFrame procesado para graficar (valores puntuales o estadísticas representativas):")
print(df.to_string(max_rows=10))

# --- Generación de Gráficos ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
plt.style.use('seaborn-v0_8-darkgrid')

# --- Gráficos de COSTO ---

# Gráfico 1: Comparación Costos Finales (Usando stats representativas)
print("\nGenerando Gráfico 1: Comparación de Costos Finales (Representativo)...")
fig1, ax1 = plt.subplots(figsize=(18, 10))
hc_stoch_label = f'HC(Stoch L{HC_FROM_STOCH_LABEL_REPRESENTATIVE})'
sa_stoch_label = f'SA(Stoch T{REPRESENTATIVE_SA_TEMP})'
cols_costo_g1 = [
    'GDet 1P Costo', 'HC(Det) 1P Costo', f'{hc_stoch_label} Min 1P Costo', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 1P Costo', f'{sa_stoch_label} Min 1P Costo',
    'GDet 2P Costo', 'HC(Det) 2P Costo', f'{hc_stoch_label} Min 2P Costo', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 2P Costo', f'{sa_stoch_label} Min 2P Costo'
]
rename_map_g1 = {
    'GDet 1P Costo': 'GDet-1P', 'HC(Det) 1P Costo': 'HC(Det)-1P', f'{hc_stoch_label} Min 1P Costo': f'{hc_stoch_label}-1P', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 1P Costo': f'SA(Det T{REPRESENTATIVE_SA_TEMP})-1P', f'{sa_stoch_label} Min 1P Costo': f'{sa_stoch_label}-1P',
    'GDet 2P Costo': 'GDet-2P', 'HC(Det) 2P Costo': 'HC(Det)-2P', f'{hc_stoch_label} Min 2P Costo': f'{hc_stoch_label}-2P', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 2P Costo': f'SA(Det T{REPRESENTATIVE_SA_TEMP})-2P', f'{sa_stoch_label} Min 2P Costo': f'{sa_stoch_label}-2P'
}
cols_exist_g1 = [col for col in cols_costo_g1 if col in df.columns]
if cols_exist_g1:
    df_subset_g1 = df[cols_exist_g1].rename(columns={k:v for k,v in rename_map_g1.items() if k in cols_exist_g1})
    df_subset_g1.plot(kind='bar', ax=ax1, rot=0, width=0.8)
    ax1.set_title('Comparación de Costos Finales por Algoritmo y Caso (Resultados Representativos)')
    ax1.set_ylabel('Costo Total (Valor/Mínimo)')
    ax1.set_xlabel('Caso de Prueba')
    ax1.grid(axis='y', linestyle='--')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(title='Algoritmo - Pistas', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    fig1_path = os.path.join(OUTPUT_GRAPH_DIR, f'01_costos_finales_representativos_{timestamp}.png')
    plt.savefig(fig1_path); print(f"Gráfico guardado en: {fig1_path}")
else:
    plt.close(fig1); print("Gráfico 1 (Costos Finales) omitido - faltan columnas.")

# Gráfico 2: Comparación Costos 1P vs 2P (Usando stats representativas)
print("Generando Gráfico 2: Comparación Costos 1 Pista vs 2 Pistas (Representativo)...")
fig2, ax2 = plt.subplots(1, 5, figsize=(25, 5), sharey=True)
fig2.subplots_adjust(wspace=0.05)
plot_success_g2 = False
if not isinstance(ax2, np.ndarray): ax2 = np.array([ax2])

# Definir métricas a comparar
metrics_g2 = [
    ('GDet', 'GDet'),
    ('HC(Det)', 'HC(Det)'),
    (f'HC(Stoch L{HC_FROM_STOCH_LABEL_REPRESENTATIVE}) Min', f'HC(Stoch L{HC_FROM_STOCH_LABEL_REPRESENTATIVE})'), # Usamos el Min Costo de HC(Stoch)
    (f'SA(Det T{REPRESENTATIVE_SA_TEMP})', f'SA(Det T{REPRESENTATIVE_SA_TEMP})'),
    (f'SA(Stoch T{REPRESENTATIVE_SA_TEMP}) Min', f'SA(Stoch T{REPRESENTATIVE_SA_TEMP})') # Usamos el Min Costo de SA(Stoch)
]

for i, (metric_key, metric_title) in enumerate(metrics_g2):
    current_ax = ax2.flat[i] # Usar flat para manejar 1D/2D
    col_1p, col_2p = f'{metric_key} 1P Costo', f'{metric_key} 2P Costo'
    if col_1p in df.columns and col_2p in df.columns:
        df[[col_1p, col_2p]].plot(kind='bar', ax=current_ax, rot=0, legend=False, width=0.8)
        current_ax.set_title(f'{metric_title}')
        current_ax.set_xlabel('Caso')
        current_ax.grid(axis='y', linestyle='--')
        current_ax.tick_params(axis='x', rotation=0)
        plot_success_g2 = True
    else:
        current_ax.text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=current_ax.transAxes)
        current_ax.set_title(f'{metric_title} (Datos Falt.)')
        current_ax.set_xlabel('Caso')

if plot_success_g2:
    ax2.flat[0].set_ylabel('Costo Total (Valor/Mínimo)')
    fig2.legend(['1 Pista', '2 Pistas'], loc='upper right')
    fig2.suptitle('Comparación de Costos: 1 Pista vs 2 Pistas (Resultados Representativos)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2_path = os.path.join(OUTPUT_GRAPH_DIR, f'02_comparacion_pistas_costo_representativo_{timestamp}.png')
    plt.savefig(fig2_path); print(f"Gráfico guardado en: {fig2_path}")
else:
    plt.close(fig2); print("Gráfico 2 (Costos Pistas) omitido - faltan columnas.")


# Gráfico 3: Boxplot HC desde Estocástico (Costo) - Reemplaza Gráfico GRASP
print("Generando Gráfico 3: Boxplot HC desde Estocástico (Costo)...")
fig3, ax3 = plt.subplots(1, len(cases), figsize=(5 * len(cases), 7), sharey=True)
fig3.subplots_adjust(wspace=0.1)
plot_success_g3 = False
if len(cases) == 1: ax3 = np.array([ax3])
# Usa la constante NUM_STOCHASTIC_RUNS definida globalmente
fig3.suptitle(f'Distribución Costos HC aplicado a {NUM_STOCHASTIC_RUNS} Soluciones Estocásticas (Label={HC_FROM_STOCH_LABEL_REPRESENTATIVE})')

for i, case_name in enumerate(cases):
    current_ax = ax3.flat[i]
    stats_hc_case = stats_hc_stoch.get(case_name, {})
    # Usa la constante HC_FROM_STOCH_LABEL_REPRESENTATIVE
    stats_1r = stats_hc_case.get(f'{HC_FROM_STOCH_LABEL_REPRESENTATIVE}_1r', {})
    stats_2r = stats_hc_case.get(f'{HC_FROM_STOCH_LABEL_REPRESENTATIVE}_2r', {})
    costs_1p = [c for c in stats_1r.get('costs', []) if c is not None and c != INFINITO_COSTO]
    costs_2p = [c for c in stats_2r.get('costs', []) if c is not None and c != INFINITO_COSTO]

    data_to_plot = []
    labels = []
    if costs_1p: data_to_plot.append(costs_1p); labels.append('1 Pista')
    if costs_2p: data_to_plot.append(costs_2p); labels.append('2 Pistas')

    if data_to_plot:
        plot_success_g3 = True
        try:
            bp = current_ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
            current_ax.set_title(f'{case_name}')
            current_ax.grid(axis='y', linestyle='--')
        except Exception as e_bp:
            print(f"Error generando boxplot HC para {case_name}: {e_bp}")
            current_ax.text(0.5, 0.5, 'Error en Boxplot', ha='center', va='center', transform=current_ax.transAxes)
            current_ax.set_title(f'{case_name} (Error)')
    else:
        current_ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=current_ax.transAxes)
        current_ax.set_title(f'{case_name} (Sin datos)')

    if i == 0 and plot_success_g3: current_ax.set_ylabel('Costo Total (HC desde Stoch)')

if plot_success_g3:
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3_path = os.path.join(OUTPUT_GRAPH_DIR, f'03_boxplot_hc_desde_estocastico_costo_{timestamp}.png')
    plt.savefig(fig3_path); print(f"Gráfico guardado en: {fig3_path}")
else:
    print("No se generó Gráfico 3 (Boxplot HC Costo).")
    plt.close(fig3)

# Gráfico 4: Boxplot Greedy Estocástico Base (Costo) - SIN CAMBIOS LÓGICOS
print("Generando Gráfico 4: Boxplot Greedy Estocástico Base (Costo)...")
fig4, ax4 = plt.subplots(1, len(cases), figsize=(5 * len(cases), 6), sharey=True)
fig4.subplots_adjust(wspace=0.1)
plot_success_g4 = False
if len(cases) == 1: ax4 = np.array([ax4]) 
fig4.suptitle('Distribución Costos Greedy Estocástico Base (10 Runs)')

for i, case_name in enumerate(cases):
    current_ax = ax4.flat[i]
    case_data = all_results.get(case_name, {})
    runs_1p = case_data.get('stochastic_1_runway_runs', [])
    runs_2p = case_data.get('stochastic_2_runway_runs', [])
    costs_1p = [r['cost'] for r in runs_1p if isinstance(r, dict) and r.get('cost') is not None and r['cost'] != INFINITO_COSTO]
    costs_2p = [r['cost'] for r in runs_2p if isinstance(r, dict) and r.get('cost') is not None and r['cost'] != INFINITO_COSTO]

    data_to_plot = []
    labels = []
    if costs_1p: data_to_plot.append(costs_1p); labels.append('1 Pista')
    if costs_2p: data_to_plot.append(costs_2p); labels.append('2 Pistas')

    if data_to_plot:
        plot_success_g4 = True
        try:
            bp = current_ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                 patch.set_facecolor(color)
            current_ax.set_title(f'{case_name}')
            current_ax.grid(axis='y', linestyle='--')
        except Exception as e_bp:
            print(f"Error generando boxplot Estocástico Base para {case_name}: {e_bp}")
            current_ax.text(0.5, 0.5, 'Error en Boxplot', ha='center', va='center', transform=current_ax.transAxes)
            current_ax.set_title(f'{case_name} (Error)')
    else:
        current_ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=current_ax.transAxes)
        current_ax.set_title(f'{case_name} (Sin datos)')

    if i == 0 and plot_success_g4: current_ax.set_ylabel('Costo Total (Greedy Stoch)')

if plot_success_g4:
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig4_path = os.path.join(OUTPUT_GRAPH_DIR, f'04_boxplot_estocastico_base_costo_{timestamp}.png')
    plt.savefig(fig4_path); print(f"Gráfico guardado en: {fig4_path}")
else:
    print("No se generó Gráfico 4 (Boxplot Estocástico Base).")
    plt.close(fig4)


# Gráfico 5: Efecto Temp SA (Costo - Usando Avg Cost para Stoch)
print("Generando Gráfico 5: Efecto Temperatura Inicial SA (Costo - Avg para Stoch)...")
fig5, ax5 = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
fig5.subplots_adjust(hspace=0.3, wspace=0.1)
plot_success_g5 = False
start_points = [('Det', 'Desde Det.'), ('Stoch', 'Desde Stoch (Avg)')]
runways = [(1, '1 Pista', '1r'), (2, '2 Pistas', '2r')]
if not isinstance(ax5, np.ndarray): ax5 = ax5.reshape(2,2)

# Crear DataFrame temporal para SA Costos
sa_cost_data = {'Caso': cases}
# SA desde Det
for T in SA_INITIAL_TEMPS:
    sa_cost_data[f'SA(Det T{T}) 1P Costo'] = [all_results[c].get(f'sa_T{T}_from_det_1r', {}).get('cost', np.nan) for c in cases]
    sa_cost_data[f'SA(Det T{T}) 2P Costo'] = [all_results[c].get(f'sa_T{T}_from_det_2r', {}).get('cost', np.nan) for c in cases]
# SA desde Stoch (Avg Cost)
for T in SA_INITIAL_TEMPS:
    sa_cost_data[f'SA(Stoch T{T}) 1P Costo'] = [stats_sa_stoch.get(c, {}).get(T, {}).get('1r', {}).get('avg_cost', np.nan) for c in cases]
    sa_cost_data[f'SA(Stoch T{T}) 2P Costo'] = [stats_sa_stoch.get(c, {}).get(T, {}).get('2r', {}).get('avg_cost', np.nan) for c in cases]
df_sa_cost = pd.DataFrame(sa_cost_data).set_index('Caso')
df_sa_cost.replace(INFINITO_COSTO, np.nan, inplace=True)

for row, (start_key, start_label) in enumerate(start_points):
    for col, (rw_num, rw_label, rw_suffix) in enumerate(runways):
        ax = ax5[row, col]
        sa_cost_cols = [f'SA({start_key} T{T}) {rw_num}P Costo' for T in SA_INITIAL_TEMPS]
        cols_exist_g5 = [c for c in sa_cost_cols if c in df_sa_cost.columns]

        if len(cols_exist_g5) == len(sa_cost_cols) and not df_sa_cost[cols_exist_g5].isnull().all().all():
            df_sa_cost[sa_cost_cols].plot(kind='bar', ax=ax, rot=0)
            ax.set_title(f'SA Costo ({start_label} - {rw_label})')
            ax.set_xlabel('Caso')
            ax.grid(axis='y', linestyle='--')
            ax.tick_params(axis='x', rotation=0)
            ax.legend([f'T={T}' for T in SA_INITIAL_TEMPS], fontsize='small', title="T Inicial")
            plot_success_g5 = True
        else:
            ax.text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'SA Costo ({start_label} - {rw_label}) (Datos Falt.)')

        if col == 0: ax.set_ylabel('Costo SA (Valor/Promedio)')

if plot_success_g5:
    fig5.suptitle('Efecto de la Temperatura Inicial en Simulated Annealing (Costo)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig5_path = os.path.join(OUTPUT_GRAPH_DIR, f'05_efecto_temp_sa_costo_avg_{timestamp}.png')
    plt.savefig(fig5_path); print(f"Gráfico guardado en: {fig5_path}")
else:
    plt.close(fig5); print("Gráfico 5 (SA Costo) omitido - faltan columnas.")


# --- Gráficos de TIEMPO ---

# Gráfico 6: Comparación Tiempos Ejecución (Usando avg tiempo para Stoch-based)
print("Generando Gráfico 6: Comparación de Tiempos de Ejecución (Representativo)...")
fig6, ax6 = plt.subplots(figsize=(18, 10))
# Usar las mismas etiquetas que en Gráfico 1, pero con Tiempo
cols_tiempo_g6 = [
    'GDet 1P Tiempo', 'HC(Det) 1P Tiempo', f'{hc_stoch_label} Avg 1P Tiempo', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 1P Tiempo', f'{sa_stoch_label} Avg 1P Tiempo', 'Stoch Avg 1P Tiempo',
    'GDet 2P Tiempo', 'HC(Det) 2P Tiempo', f'{hc_stoch_label} Avg 2P Tiempo', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 2P Tiempo', f'{sa_stoch_label} Avg 2P Tiempo', 'Stoch Avg 2P Tiempo'
]
rename_map_g6 = {
    'GDet 1P Tiempo': 'GDet-1P', 'HC(Det) 1P Tiempo': 'HC(Det)-1P', f'{hc_stoch_label} Avg 1P Tiempo': f'{hc_stoch_label}(Avg)-1P', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 1P Tiempo': f'SA(Det T{REPRESENTATIVE_SA_TEMP})-1P', f'{sa_stoch_label} Avg 1P Tiempo': f'{sa_stoch_label}(Avg)-1P', 'Stoch Avg 1P Tiempo': 'StochBase(Avg)-1P',
    'GDet 2P Tiempo': 'GDet-2P', 'HC(Det) 2P Tiempo': 'HC(Det)-2P', f'{hc_stoch_label} Avg 2P Tiempo': f'{hc_stoch_label}(Avg)-2P', f'SA(Det T{REPRESENTATIVE_SA_TEMP}) 2P Tiempo': f'SA(Det T{REPRESENTATIVE_SA_TEMP})-2P', f'{sa_stoch_label} Avg 2P Tiempo': f'{sa_stoch_label}(Avg)-2P', 'Stoch Avg 2P Tiempo': 'StochBase(Avg)-2P'
}
cols_exist_g6 = [col for col in cols_tiempo_g6 if col in df.columns]
if cols_exist_g6:
    df_subset_g6 = df[cols_exist_g6].rename(columns={k:v for k,v in rename_map_g6.items() if k in cols_exist_g6})
    df_subset_g6 = df_subset_g6.apply(pd.to_numeric, errors='coerce')
    df_subset_g6.plot(kind='bar', ax=ax6, rot=0, width=0.8)
    ax6.set_title('Comparación de Tiempos de Ejecución por Algoritmo y Caso (Resultados Representativos)')
    ax6.set_ylabel('Tiempo Promedio/Total (s)')
    ax6.set_xlabel('Caso de Prueba')
    ax6.grid(axis='y', linestyle='--')
    ax6.tick_params(axis='x', rotation=0)
    ax6.legend(title='Algoritmo - Pistas', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    fig6_path = os.path.join(OUTPUT_GRAPH_DIR, f'06_tiempos_ejecucion_representativos_{timestamp}.png')
    plt.savefig(fig6_path); print(f"Gráfico guardado en: {fig6_path}")
else:
    plt.close(fig6); print("Gráfico 6 (Tiempos Ejecución) omitido - faltan columnas.")

# Gráfico 7: Boxplot Tiempos HC desde Estocástico - Reemplaza Gráfico GRASP Tiempo
print("Generando Gráfico 7: Boxplot Tiempos HC desde Estocástico...")
fig7, ax7 = plt.subplots(1, len(cases), figsize=(5 * len(cases), 7), sharey=True)
fig7.subplots_adjust(wspace=0.1)
plot_success_g7 = False
if len(cases) == 1: ax7 = np.array([ax7])
fig7.suptitle(f'Distribución Tiempos HC aplicado a Soluciones Estocásticas (Label={HC_FROM_STOCH_LABEL_REPRESENTATIVE})')

for i, case_name in enumerate(cases):
    current_ax = ax7.flat[i]
    stats_hc_case = stats_hc_stoch.get(case_name, {})
    stats_1r = stats_hc_case.get(f'{HC_FROM_STOCH_LABEL_REPRESENTATIVE}_1r', {})
    stats_2r = stats_hc_case.get(f'{HC_FROM_STOCH_LABEL_REPRESENTATIVE}_2r', {})
    times_1p = [t for t in stats_1r.get('times', []) if t is not None]
    times_2p = [t for t in stats_2r.get('times', []) if t is not None]

    data_to_plot = []
    labels = []
    if times_1p: data_to_plot.append(times_1p); labels.append('1 Pista')
    if times_2p: data_to_plot.append(times_2p); labels.append('2 Pistas')

    if data_to_plot:
        plot_success_g7 = True
        try:
            bp = current_ax.boxplot(data_to_plot, patch_artist=True, labels=labels)
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
            current_ax.set_title(f'{case_name}')
            current_ax.grid(axis='y', linestyle='--')
        except Exception as e_bp:
            print(f"Error generando boxplot HC Tiempo para {case_name}: {e_bp}")
            current_ax.text(0.5, 0.5, 'Error en Boxplot', ha='center', va='center', transform=current_ax.transAxes)
            current_ax.set_title(f'{case_name} (Error)')
    else:
        current_ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=current_ax.transAxes)
        current_ax.set_title(f'{case_name} (Sin datos)')

    if i == 0 and plot_success_g7: current_ax.set_ylabel('Tiempo Ejecución HC (s)')

if plot_success_g7:
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig7_path = os.path.join(OUTPUT_GRAPH_DIR, f'07_boxplot_hc_desde_estocastico_tiempo_{timestamp}.png')
    plt.savefig(fig7_path); print(f"Gráfico guardado en: {fig7_path}")
else:
    print("No se generó Gráfico 7 (Boxplot HC Tiempo).")
    plt.close(fig7)


# Gráfico 8: Efecto Temperatura Inicial SA (Tiempo - Usando Avg Tiempo para Stoch)
print("Generando Gráfico 8: Efecto Temperatura Inicial SA (Tiempo - Avg para Stoch)...")
fig8, ax8 = plt.subplots(2, 2, figsize=(16, 10), sharey=False)
fig8.subplots_adjust(hspace=0.3, wspace=0.15)
plot_success_g8 = False
start_points = [('Det', 'Desde Det.'), ('Stoch', 'Desde Stoch (Avg)')]
runways = [(1, '1 Pista', '1r'), (2, '2 Pistas', '2r')]
if not isinstance(ax8, np.ndarray): ax8 = ax8.reshape(2,2)

# Crear DataFrame temporal para SA Tiempos
sa_time_data = {'Caso': cases}
# SA desde Det
for T in SA_INITIAL_TEMPS:
    sa_time_data[f'SA(Det T{T}) 1P Tiempo'] = [all_results[c].get(f'sa_T{T}_from_det_1r', {}).get('time', np.nan) for c in cases]
    sa_time_data[f'SA(Det T{T}) 2P Tiempo'] = [all_results[c].get(f'sa_T{T}_from_det_2r', {}).get('time', np.nan) for c in cases]
# SA desde Stoch (Avg Tiempo)
for T in SA_INITIAL_TEMPS:
     # Asegurarse que las claves existen antes de acceder
    sa_time_data[f'SA(Stoch T{T}) 1P Tiempo'] = [stats_sa_stoch.get(c, {}).get(T, {}).get('1r', {}).get('avg_time', np.nan) for c in cases]
    sa_time_data[f'SA(Stoch T{T}) 2P Tiempo'] = [stats_sa_stoch.get(c, {}).get(T, {}).get('2r', {}).get('avg_time', np.nan) for c in cases]
df_sa_time = pd.DataFrame(sa_time_data).set_index('Caso')
df_sa_time.replace(INFINITO_COSTO, np.nan, inplace=True)

for row, (start_key, start_label) in enumerate(start_points):
    for col, (rw_num, rw_label, rw_suffix) in enumerate(runways):
        ax = ax8[row, col]
        sa_time_cols = [f'SA({start_key} T{T}) {rw_num}P Tiempo' for T in SA_INITIAL_TEMPS]
        cols_exist_g8 = [c for c in sa_time_cols if c in df_sa_time.columns]

        if len(cols_exist_g8) == len(sa_time_cols) and not df_sa_time[cols_exist_g8].isnull().all().all():
            df_sa_time[sa_time_cols].apply(pd.to_numeric, errors='coerce').plot(kind='bar', ax=ax, rot=0)
            ax.set_title(f'SA Tiempo ({start_label} - {rw_label})')
            ax.set_xlabel('Caso')
            ax.grid(axis='y', linestyle='--')
            ax.tick_params(axis='x', rotation=0)
            ax.legend([f'T={T}' for T in SA_INITIAL_TEMPS], fontsize='small', title="T Inicial")
            plot_success_g8 = True
        else:
            ax.text(0.5, 0.5, 'Datos Faltantes', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'SA Tiempo ({start_label} - {rw_label}) (Datos Falt.)')

        if col == 0: ax.set_ylabel('Tiempo Ejecución SA (s) (Valor/Promedio)')

if plot_success_g8:
    fig8.suptitle('Efecto de la Temperatura Inicial en SA (Tiempo Ejecución)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig8_path = os.path.join(OUTPUT_GRAPH_DIR, f'08_efecto_temp_sa_tiempo_avg_{timestamp}.png')
    plt.savefig(fig8_path); print(f"Gráfico guardado en: {fig8_path}")
else:
    plt.close(fig8); print("Gráfico 8 (SA Tiempo) omitido - faltan columnas.")

print("\n--- Generación de gráficos completada ---")