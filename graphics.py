import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors # Para colores
import pandas as pd
import numpy as np
import os
import time
import shutil
import statistics
import copy
import sys # Para sys.exit
import traceback # Para errores más detallados

# --- Constantes y Configuración ---

CSV_INPUT_FILENAME = 'results/results_summary.csv'
OUTPUT_GRAPH_DIR = 'results/graficos'

# Verificar si el archivo CSV existe
if not os.path.exists(CSV_INPUT_FILENAME):
    print(f"ERROR: El archivo CSV '{CSV_INPUT_FILENAME}' no fue encontrado.")
    print("Asegúrate de que el archivo CSV generado por main.py")
    print("exista en la ruta especificada.")
    sys.exit(1)


INFINITO_COSTO = float('inf')

# Parámetros que coinciden con main.py
NUM_STOCHASTIC_RUNS = 10
RCL_SIZE = 3
HC_MAX_ITER = 500
GRASP_RESTARTS_LIST = [10, 25, 50]
NUM_GRASP_EXECUTIONS = 10
SA_INITIAL_TEMPS = [10000, 5000, 1000, 500, 100]
SA_T_MIN = 0.1
SA_ALPHA = 0.95
SA_ITER_PER_TEMP = 100
SA_MAX_NEIGHBOR_ATTEMPTS = 50

# Selección representativa para algunos gráficos comparativos
REPRESENTATIVE_GRASP_ITERS = GRASP_RESTARTS_LIST[1] # Usar 25 restarts como representativo
REPRESENTATIVE_SA_TEMP = SA_INITIAL_TEMPS[2] # Usar 1000 como representativo

# Crear/limpiar directorio de salida
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
    sys.exit(1)


# --- Cargar Datos desde CSV ---
print(f"Cargando datos desde: {CSV_INPUT_FILENAME}")
try:
    # Leer el CSV especificando tipos donde sea necesario para evitar problemas
    # Ampliar na_values para incluir cadenas como 'ERR_FMT' o 'ERR_CALC' si pueden aparecer
    df_raw = pd.read_csv(CSV_INPUT_FILENAME, keep_default_na=True,
                         na_values=['', '#N/A', '#N/A N/A', '#NA', '-1.#IND',
                                    '-1.#QNAN', '-NaN', '-nan', '1.#IND', '1.#QNAN',
                                    '<NA>', 'N/A', 'NA', 'NULL', 'NaN', 'None',
                                    'nan', 'null', 'INF', '-INF', 'ERR_FMT', 'ERR_CALC'])

    # Convertir columnas relevantes a numérico, tratando errores
    # Asegúrate de incluir 'Tiempo_s' aquí si no lo estaba
    cols_to_numeric = ['Costo Final', 'Tiempo_s', 'Costo Inicial', 'Mejora_%']
    for col in cols_to_numeric:
         # Extraer solo número de Tiempo_s (quitar 's')
        if col == 'Tiempo_s' and df_raw[col].dtype == 'object':
             df_raw[col] = df_raw[col].astype(str).str.replace('s', '', regex=False)
        # Extraer solo número de Mejora_% (quitar '%')
        if col == 'Mejora_%' and df_raw[col].dtype == 'object':
            df_raw[col] = df_raw[col].astype(str).str.replace('%', '', regex=False)

        # Intentar convertir a numérico
        df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

    # Reemplazar posibles infinitos (si no fueron capturados por na_values) y NaN con np.nan o un valor manejable
    df_raw.replace([np.inf, -np.inf], np.nan, inplace=True)
    # print(f"Tipos de datos después de conversión:\n{df_raw.dtypes}") # Debug

except FileNotFoundError:
    print(f"Error: '{CSV_INPUT_FILENAME}' no encontrado."); sys.exit(1)
except Exception as e:
    print(f"Error inesperado cargando o procesando CSV: {e}");
    traceback.print_exc() # Import traceback
    sys.exit(1)

print(f"Datos cargados. {len(df_raw)} filas leídas.")
# print(df_raw.head().to_string()) # Debug
# print(df_raw.info()) # Verificar tipos y NaNs

# --- Procesar Datos Agrupados para Estadísticas ---
# Calcular estadísticas para algoritmos con múltiples ejecuciones (Stoch, GRASP, SA from Stoch)

# Función para calcular estadísticas agrupadas desde el DataFrame
def calculate_grouped_stats(df_group):
    valid_costs = df_group['Costo Final'].dropna()
    valid_times = df_group['Tiempo_s'].dropna() # <--- CORRECCIÓN: Calcular stats de tiempo aquí también
    valid_count = len(valid_costs)
    stats = {
        'valid_runs': valid_count,
        'min_cost': valid_costs.min() if valid_count > 0 else np.nan,
        'max_cost': valid_costs.max() if valid_count > 0 else np.nan,
        'avg_cost': valid_costs.mean() if valid_count > 0 else np.nan,
        'stdev_cost': valid_costs.std() if valid_count > 1 else 0.0,
        'avg_time': valid_times.mean() if len(valid_times) > 0 else np.nan, # <-- CORRECCIÓN
        'min_time': valid_times.min() if len(valid_times) > 0 else np.nan, # <-- CORRECCIÓN (Opcional)
        'max_time': valid_times.max() if len(valid_times) > 0 else np.nan, # <-- CORRECCIÓN (Opcional)
        'stdev_time': valid_times.std() if len(valid_times) > 1 else 0.0, # <-- CORRECCIÓN (Opcional)
        'total_time': valid_times.sum() # <-- CORRECCIÓN
    }
    return pd.Series(stats)

# Filtrar y agrupar para obtener estadísticas
print("Calculando estadísticas agrupadas...")
try:
    # Estocástico (ignorar grupos si dan error)
    stats_stoch = df_raw[df_raw['Algoritmo'] == 'Greedy Stochastic'].groupby(['Caso', 'Pistas']).apply(calculate_grouped_stats, include_groups=False).unstack()

    # GRASP
    # Asegurarse que Parametros no sea NaN antes de agrupar
    df_grasp_filtered = df_raw[(df_raw['Algoritmo'] == 'GRASP') & df_raw['Parametros'].notna()].copy()
    if not df_grasp_filtered.empty:
        stats_grasp = df_grasp_filtered.groupby(['Caso', 'Pistas', 'Parametros']).apply(calculate_grouped_stats, include_groups=False).unstack(level=[1,2]) # Unstack por Pistas y Parametros
    else:
        stats_grasp = pd.DataFrame() # Dataframe vacío si no hay datos GRASP
        print("Advertencia: No se encontraron datos válidos para GRASP.")

    # SA desde Estocástico
    df_sa_stoch_filtered = df_raw[(df_raw['Algoritmo'] == 'SA') & df_raw['Punto Partida'].str.contains('Stochastic', na=False) & df_raw['Parametros'].notna()].copy()
    if not df_sa_stoch_filtered.empty:
        stats_sa_stoch = df_sa_stoch_filtered.groupby(['Caso', 'Pistas', 'Parametros']).apply(calculate_grouped_stats, include_groups=False).unstack(level=[1,2])
    else:
        stats_sa_stoch = pd.DataFrame() # Dataframe vacío si no hay datos SA(Stoch)
        print("Advertencia: No se encontraron datos válidos para SA(Stochastic).")

except Exception as e_stats:
    print(f"Error calculando estadísticas agrupadas: {e_stats}")
    traceback.print_exc()
    # Podrías decidir salir o continuar con DataFrames vacíos
    stats_stoch = pd.DataFrame()
    stats_grasp = pd.DataFrame()
    stats_sa_stoch = pd.DataFrame()

# Crear DataFrame principal para gráficos comparativos (usando valores únicos y stats)
unique_cases = sorted(df_raw['Caso'].unique()) # <--- CORRECCIÓN: Obtener casos únicos
df_plot = pd.DataFrame(index=unique_cases)

# Añadir datos de corridas únicas (Det, HC(Det), SA(Det))
# Usar try-except para manejar posibles errores al extraer T_init
print("Procesando datos de corridas únicas...")
for idx, row in df_raw[df_raw['Algoritmo'].isin(['Greedy Deterministic', 'HC', 'SA']) & (df_raw['Punto Partida'].isin(['N/A', 'Deterministic']))].iterrows():
    case = row['Caso']
    if case not in df_plot.index: continue # Saltar si el caso no está en nuestro índice (raro)
    algo = row['Algoritmo']
    pistas = row['Pistas']
    start = row['Punto Partida']
    params = row['Parametros']

    # Crear nombres de columna descriptivos
    col_prefix = f"GDet" if algo == 'Greedy Deterministic' else algo # Acortar nombre
    if algo == 'HC': col_prefix = f"HC(Det)"
    elif algo == 'SA':
        t_init = 'Err' # Default
        if isinstance(params, str): # Asegurar que params es string
            try:
                # Buscar 'T_init=xxx'
                parts = params.split(',')
                for part in parts:
                    if 'T_init=' in part:
                        t_init = int(part.split('=')[1])
                        break
            except Exception:
                 # print(f"Advertencia: No se pudo extraer T_init de '{params}'") # Debug opcional
                 pass # Mantener t_init = 'Err'
        col_prefix = f"SA(Det T{t_init})"

    cost_col = f"{col_prefix} {pistas}P Costo"
    time_col = f"{col_prefix} {pistas}P Tiempo"
    # No es necesario pre-crear columnas con NaN si usamos .loc
    df_plot.loc[case, cost_col] = row['Costo Final']
    df_plot.loc[case, time_col] = row['Tiempo_s']

# Añadir estadísticas de corridas múltiples (Stoch, GRASP, SA(Stoch))
# Usar .get() en los DataFrames de stats para evitar KeyErrors
print("Procesando estadísticas de corridas múltiples...")
for case in df_plot.index:
    for pistas in [1, 2]:
        # Estocástico (usar stats_stoch)
        if not stats_stoch.empty:
            df_plot.loc[case, f'Stoch Min {pistas}P Costo'] = stats_stoch.get(('min_cost', pistas), {}).get(case, np.nan)
            df_plot.loc[case, f'Stoch Avg {pistas}P Costo'] = stats_stoch.get(('avg_cost', pistas), {}).get(case, np.nan) # Añadir Avg también
            df_plot.loc[case, f'Stoch Avg {pistas}P Tiempo'] = stats_stoch.get(('avg_time', pistas), {}).get(case, np.nan)

        # GRASP (usar stats_grasp y representativo)
        if not stats_grasp.empty:
             for grasp_iter in GRASP_RESTARTS_LIST:
                 param_str = f"Restarts={grasp_iter}, RCL={RCL_SIZE}, HC_iter={HC_MAX_ITER}"
                 avg_cost = stats_grasp.get(('avg_cost', pistas, param_str), {}).get(case, np.nan)
                 avg_time = stats_grasp.get(('avg_time', pistas, param_str), {}).get(case, np.nan)
                 df_plot.loc[case, f'GRASP {grasp_iter}r Avg {pistas}P Costo'] = avg_cost
                 df_plot.loc[case, f'GRASP {grasp_iter}r Avg {pistas}P Tiempo'] = avg_time
                 if grasp_iter == REPRESENTATIVE_GRASP_ITERS:
                     min_cost = stats_grasp.get(('min_cost', pistas, param_str), {}).get(case, np.nan)
                     df_plot.loc[case, f'GRASP {grasp_iter}r Min {pistas}P Costo'] = min_cost # Guardar el representativo Min

        # SA desde Estocástico (usar stats_sa_stoch y representativo)
        if not stats_sa_stoch.empty:
            for T_init in SA_INITIAL_TEMPS:
                param_str = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
                avg_cost = stats_sa_stoch.get(('avg_cost', pistas, param_str), {}).get(case, np.nan)
                avg_time = stats_sa_stoch.get(('avg_time', pistas, param_str), {}).get(case, np.nan)
                df_plot.loc[case, f'SA(Stoch T{T_init}) Avg {pistas}P Costo'] = avg_cost
                df_plot.loc[case, f'SA(Stoch T{T_init}) Avg {pistas}P Tiempo'] = avg_time
                if T_init == REPRESENTATIVE_SA_TEMP:
                    min_cost = stats_sa_stoch.get(('min_cost', pistas, param_str), {}).get(case, np.nan)
                    df_plot.loc[case, f'SA(Stoch T{T_init}) Min {pistas}P Costo'] = min_cost # Guardar el representativo Min


print("\nDataFrame procesado para graficar (valores puntuales y estadísticas):")
print(df_plot.head().to_string())


# --- Generación de Gráficos ---
timestamp = time.strftime("%Y%m%d_%H%M%S")
plt.style.use('seaborn-v0_8-darkgrid')
case_list = df_plot.index # <--- CORRECCIÓN: Usar el índice de df_plot como la lista de casos

# --- GRÁFICOS ---

# Gráfico 0: Factibilidad por Algoritmo y Pista (NUEVO)
# (Sin cambios, parece funcionar)
print("\nGenerando Gráfico 0: Conteo de Soluciones Factibles/Infactibles...")
fig0, ax0 = plt.subplots(figsize=(15, 8))
try:
    feasibility_counts = df_raw.groupby(['Caso', 'Algoritmo', 'Pistas', 'Factible']).size().unstack(fill_value=0)
    feasibility_summary = df_raw.groupby(['Algoritmo', 'Pistas', 'Factible']).size().unstack(fill_value=0)

    if 'Factible' not in feasibility_summary.columns: feasibility_summary['Factible'] = 0
    if 'Infactible' not in feasibility_summary.columns: feasibility_summary['Infactible'] = 0
    feasibility_summary['Total'] = feasibility_summary['Factible'] + feasibility_summary['Infactible'] # Sumar explícitamente
    feasibility_summary['% Factible'] = (feasibility_summary['Factible'] / feasibility_summary['Total'].replace(0, np.nan)) * 100
    feasibility_summary['% Infactible'] = (feasibility_summary['Infactible'] / feasibility_summary['Total'].replace(0, np.nan)) * 100

    cols_to_plot_g0 = ['Factible', 'Infactible']
    feasibility_summary[cols_to_plot_g0].plot(kind='bar', stacked=True, ax=ax0, rot=45)
    ax0.set_title('Conteo de Soluciones Factibles vs. Infactibles por Algoritmo y Pistas')
    ax0.set_ylabel('Número de Ejecuciones')
    ax0.set_xlabel('Algoritmo - Pistas')
    ax0.tick_params(axis='x', rotation=45, ha='right')
    ax0.legend(title='Factibilidad')
    ax0.grid(axis='y', linestyle='--')
    plt.tight_layout()
    fig0_path = os.path.join(OUTPUT_GRAPH_DIR, f'00_factibilidad_soluciones_{timestamp}.png')
    plt.savefig(fig0_path); print(f"Gráfico guardado en: {fig0_path}")
except Exception as e:
     plt.close(fig0); print(f"Error generando Gráfico 0 (Factibilidad): {e}")


# Gráfico 1: Comparación Costos Finales (AVG para multi-run)
# (Sin cambios, parece funcionar)
print("\nGenerando Gráfico 1: Comparación de Costos Finales (Promedio/Valor)...")
fig1, ax1 = plt.subplots(figsize=(18, 10))
grasp_rep_label_avg = f'GRASP {REPRESENTATIVE_GRASP_ITERS}r Avg'
sa_det_rep_label = f'SA(Det T{REPRESENTATIVE_SA_TEMP})'
sa_stoch_rep_label_avg = f'SA(Stoch T{REPRESENTATIVE_SA_TEMP}) Avg'

cols_costo_g1 = [
    'GDet 1P Costo', 'HC(Det) 1P Costo', f'{grasp_rep_label_avg} 1P Costo', f'{sa_det_rep_label} 1P Costo', f'{sa_stoch_rep_label_avg} 1P Costo',
    'GDet 2P Costo', 'HC(Det) 2P Costo', f'{grasp_rep_label_avg} 2P Costo', f'{sa_det_rep_label} 2P Costo', f'{sa_stoch_rep_label_avg} 2P Costo'
]
rename_map_g1 = {
    'GDet 1P Costo': 'GDet-1P', 'HC(Det) 1P Costo': 'HC(Det)-1P', f'{grasp_rep_label_avg} 1P Costo': f'GRASP{REPRESENTATIVE_GRASP_ITERS}(Avg)-1P', f'{sa_det_rep_label} 1P Costo': f'{sa_det_rep_label}-1P', f'{sa_stoch_rep_label_avg} 1P Costo': f'{sa_stoch_rep_label_avg}(Avg)-1P',
    'GDet 2P Costo': 'GDet-2P', 'HC(Det) 2P Costo': 'HC(Det)-2P', f'{grasp_rep_label_avg} 2P Costo': f'GRASP{REPRESENTATIVE_GRASP_ITERS}(Avg)-2P', f'{sa_det_rep_label} 2P Costo': f'{sa_det_rep_label}-2P', f'{sa_stoch_rep_label_avg} 2P Costo': f'{sa_stoch_rep_label_avg}(Avg)-2P'
}
cols_exist_g1 = [col for col in cols_costo_g1 if col in df_plot.columns]
if cols_exist_g1:
    df_subset_g1 = df_plot[cols_exist_g1].rename(columns={k:v for k,v in rename_map_g1.items() if k in cols_exist_g1})
    df_subset_g1.plot(kind='bar', ax=ax1, rot=0, width=0.8)
    ax1.set_title('Comparación de Costos Finales por Algoritmo y Caso (Resultados Promedio/Valor)')
    ax1.set_ylabel('Costo Total (Valor/Promedio)')
    ax1.set_xlabel('Caso de Prueba')
    ax1.grid(axis='y', linestyle='--')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(title='Algoritmo - Pistas', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    plt.tight_layout(rect=[0, 0, 0.80, 1]) # Ajustar para leyenda fuera
    fig1_path = os.path.join(OUTPUT_GRAPH_DIR, f'01_costos_finales_promedio_{timestamp}.png')
    plt.savefig(fig1_path); print(f"Gráfico guardado en: {fig1_path}")
else:
    plt.close(fig1); print("Gráfico 1 (Costos Finales) omitido - faltan columnas.")


# Gráfico 2: Comparación Costos 1P vs 2P (Usando AVG para multi-run)
# (Sin cambios, parece funcionar)
print("Generando Gráfico 2: Comparación Costos 1 Pista vs 2 Pistas (Promedio/Valor)...")
fig2, ax2 = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
fig2.subplots_adjust(wspace=0.05)
plot_success_g2 = False
if not isinstance(ax2, np.ndarray): ax2 = np.array([ax2])

metrics_g2 = [
    ('GDet', 'GDet'),
    ('HC(Det)', 'HC(Det)'),
    (f'GRASP {REPRESENTATIVE_GRASP_ITERS}r Avg', f'GRASP {REPRESENTATIVE_GRASP_ITERS}r (Avg)'),
    (f'SA(Stoch T{REPRESENTATIVE_SA_TEMP}) Avg', f'SA Stoch T{REPRESENTATIVE_SA_TEMP} (Avg)')
]

for i, (metric_key, metric_title) in enumerate(metrics_g2):
    current_ax = ax2.flat[i]
    col_1p, col_2p = f'{metric_key} 1P Costo', f'{metric_key} 2P Costo'
    if col_1p in df_plot.columns and col_2p in df_plot.columns and not df_plot[[col_1p, col_2p]].isnull().all().all():
        df_plot[[col_1p, col_2p]].plot(kind='bar', ax=current_ax, rot=0, legend=False, width=0.8)
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
    ax2.flat[0].set_ylabel('Costo Total (Valor/Promedio)')
    fig2.legend(['1 Pista', '2 Pistas'], loc='upper right')
    fig2.suptitle('Comparación de Costos: 1 Pista vs 2 Pistas (Resultados Promedio/Valor)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2_path = os.path.join(OUTPUT_GRAPH_DIR, f'02_comparacion_pistas_costo_promedio_{timestamp}.png')
    plt.savefig(fig2_path); print(f"Gráfico guardado en: {fig2_path}")
else:
    plt.close(fig2); print("Gráfico 2 (Costos Pistas) omitido - faltan columnas.")


# Gráfico 3: Boxplot GRASP por Configuración de Restarts (Costo)
print("Generando Gráfico 3: Boxplot GRASP por Configuración de Restarts (Costo)...")
# <--- CORRECCIÓN: Usar case_list (índice de df_plot)
fig3, ax3 = plt.subplots(1, len(case_list), figsize=(6 * len(case_list), 7), sharey=True)
fig3.subplots_adjust(wspace=0.15)
plot_success_g3 = False
if len(case_list) == 1: ax3 = np.array([ax3]) # Asegurar que sea iterable
fig3.suptitle(f'Distribución Costos GRASP ({NUM_GRASP_EXECUTIONS} Runs por Configuración)')

colors = plt.cm.viridis(np.linspace(0, 1, len(GRASP_RESTARTS_LIST))) # Colores para configs

# <--- CORRECCIÓN: Iterar sobre case_list
for i, case_name in enumerate(case_list):
    if len(case_list) > 1: current_ax = ax3.flat[i]
    else: current_ax = ax3[0] # Acceder al único eje
    data_to_plot_1p = []
    data_to_plot_2p = []
    labels_1p = []
    labels_2p = []
    positions_1p = []
    positions_2p = []
    color_map = {}

    base_pos = 0
    for k, restart_iters in enumerate(GRASP_RESTARTS_LIST):
        param_str = f"Restarts={restart_iters}, RCL={RCL_SIZE}, HC_iter={HC_MAX_ITER}"
        # 1 Pista
        # Filtrar directamente df_raw para obtener todas las runs
        costs_1p = df_raw[(df_raw['Caso'] == case_name) &
                          (df_raw['Algoritmo'] == 'GRASP') &
                          (df_raw['Pistas'] == 1) &
                          (df_raw['Parametros'] == param_str)]['Costo Final'].dropna().tolist()
        if costs_1p:
            data_to_plot_1p.append(costs_1p)
            labels_1p.append(f'{restart_iters}r-1P')
            positions_1p.append(base_pos + 1)
            color_map[f'{restart_iters}r-1P'] = colors[k]

        # 2 Pistas
        costs_2p = df_raw[(df_raw['Caso'] == case_name) &
                          (df_raw['Algoritmo'] == 'GRASP') &
                          (df_raw['Pistas'] == 2) &
                          (df_raw['Parametros'] == param_str)]['Costo Final'].dropna().tolist()
        if costs_2p:
            data_to_plot_2p.append(costs_2p)
            labels_2p.append(f'{restart_iters}r-2P')
            positions_2p.append(base_pos + 2)
            color_map[f'{restart_iters}r-2P'] = colors[k]
        base_pos += 3 # Espacio entre grupos de restarts

    all_data = data_to_plot_1p + data_to_plot_2p
    all_labels = labels_1p + labels_2p
    all_positions = positions_1p + positions_2p

    if all_data: # Si hay datos para este caso
        plot_success_g3 = True
        try:
            # Filtrar datos vacíos que pueden causar error en boxplot
            valid_indices = [idx for idx, data in enumerate(all_data) if data]
            if not valid_indices: continue # Saltar si no hay datos válidos después de filtrar listas vacías

            filtered_data = [all_data[idx] for idx in valid_indices]
            filtered_labels = [all_labels[idx] for idx in valid_indices]
            filtered_positions = [all_positions[idx] for idx in valid_indices]

            bp = current_ax.boxplot(filtered_data, patch_artist=True, labels=filtered_labels, positions=filtered_positions, widths=0.6, showfliers=False) # Ocultar outliers si son muchos
            for j, patch in enumerate(bp['boxes']):
                patch.set_facecolor(color_map[filtered_labels[j]]) # Usar color por config

            current_ax.set_title(f'{case_name}')
            current_ax.grid(axis='y', linestyle='--')
            current_ax.tick_params(axis='x', rotation=45, labelsize=8)
            if filtered_positions:
                 current_ax.set_xticks(filtered_positions) # Asegurar que los ticks coincidan
                 current_ax.set_xticklabels(filtered_labels, rotation=45, ha='right')
                 current_ax.set_xlim(min(filtered_positions) - 1, max(filtered_positions) + 1)

        except Exception as e_bp:
            print(f"Error generando boxplot GRASP para {case_name}: {e_bp}")
            current_ax.text(0.5, 0.5, 'Error en Boxplot', ha='center', va='center', transform=current_ax.transAxes)
            current_ax.set_title(f'{case_name} (Error)')
    else:
        current_ax.text(0.5, 0.5, 'Sin datos GRASP válidos', ha='center', va='center', transform=current_ax.transAxes)
        current_ax.set_title(f'{case_name} (Sin datos)')

    if i == 0 and plot_success_g3: current_ax.set_ylabel('Costo Total (GRASP)')

if plot_success_g3:
    handles = [plt.Rectangle((0,0),1,1, color=colors[k]) for k in range(len(GRASP_RESTARTS_LIST))]
    fig3.legend(handles, [f'{r} Restarts' for r in GRASP_RESTARTS_LIST], loc='upper right', title="Config. Restarts")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig3_path = os.path.join(OUTPUT_GRAPH_DIR, f'03_boxplot_grasp_costo_{timestamp}.png')
    plt.savefig(fig3_path); print(f"Gráfico guardado en: {fig3_path}")
else:
    print("No se generó Gráfico 3 (Boxplot GRASP Costo).")
    plt.close(fig3)


# Gráfico 4: Boxplot Greedy Estocástico Base (Costo)
print("Generando Gráfico 4: Boxplot Greedy Estocástico Base (Costo)...")
# <--- CORRECCIÓN: Usar case_list
fig4, ax4 = plt.subplots(1, len(case_list), figsize=(5 * len(case_list), 6), sharey=True)
fig4.subplots_adjust(wspace=0.1)
plot_success_g4 = False
if len(case_list) == 1: ax4 = np.array([ax4]) # Asegurar iterable
fig4.suptitle(f'Distribución Costos Greedy Estocástico Base ({NUM_STOCHASTIC_RUNS} Runs)')

# <--- CORRECCIÓN: Iterar sobre case_list
for i, case_name in enumerate(case_list):
    if len(case_list) > 1: current_ax = ax4.flat[i]
    else: current_ax = ax4[0]

    # Filtrar df_raw directamente
    costs_1p = df_raw[(df_raw['Caso'] == case_name) & (df_raw['Algoritmo'] == 'Greedy Stochastic') & (df_raw['Pistas'] == 1)]['Costo Final'].dropna().tolist()
    costs_2p = df_raw[(df_raw['Caso'] == case_name) & (df_raw['Algoritmo'] == 'Greedy Stochastic') & (df_raw['Pistas'] == 2)]['Costo Final'].dropna().tolist()

    data_to_plot = []
    labels = []
    positions = [] # Para controlar espaciado
    if costs_1p: data_to_plot.append(costs_1p); labels.append('1 Pista'); positions.append(1)
    if costs_2p: data_to_plot.append(costs_2p); labels.append('2 Pistas'); positions.append(2)

    if data_to_plot:
        plot_success_g4 = True
        try:
            bp = current_ax.boxplot(data_to_plot, patch_artist=True, labels=labels, positions=positions, widths=0.5, showfliers=False)
            colors = ['lightblue', 'lightgreen']
            for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
                patch.set_facecolor(color)
            current_ax.set_title(f'{case_name}')
            current_ax.grid(axis='y', linestyle='--')
            current_ax.set_xticks(positions) # Asegurar ticks correctos
            current_ax.set_xticklabels(labels)
            current_ax.set_xlim(0, 3) # Ajustar límites x
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
# (Sin cambios lógicos, parece funcionar)
print("Generando Gráfico 5: Efecto Temperatura Inicial SA (Costo - Avg para Stoch)...")
fig5, ax5 = plt.subplots(2, 2, figsize=(16, 10), sharey=True)
fig5.subplots_adjust(hspace=0.3, wspace=0.1)
plot_success_g5 = False
start_points = [('Det', 'Deterministic'), ('Stoch', 'Stochastic')] # Clave para filtrar df_raw
runways = [(1, '1 Pista'), (2, '2 Pistas')]
if not isinstance(ax5, np.ndarray): ax5 = ax5.reshape(2,2)

# Re-crear df_sa_cost_plot por si acaso
df_sa_cost_plot = pd.DataFrame(index=case_list)
for T_init in SA_INITIAL_TEMPS:
    for pistas in [1, 2]:
        # Desde Det (valor único de df_plot)
        col_det_cost = f'SA(Det T{T_init}) {pistas}P Costo'
        if col_det_cost in df_plot.columns:
             df_sa_cost_plot[col_det_cost] = df_plot[col_det_cost]
        # Desde Stoch (promedio de stats_sa_stoch)
        if not stats_sa_stoch.empty:
            param_str_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
            avg_cost = stats_sa_stoch.get(('avg_cost', pistas, param_str_sa), pd.Series(index=case_list, dtype=float))
            df_sa_cost_plot[f'SA(Stoch T{T_init}) {pistas}P Costo'] = avg_cost.reindex(case_list) # Asegurar alineación

df_sa_cost_plot.replace(INFINITO_COSTO, np.nan, inplace=True)


for row, (start_key, start_label) in enumerate(start_points):
    for col, (rw_num, rw_label) in enumerate(runways):
        ax = ax5[row, col]
        sa_cost_cols = [f'SA({start_key} T{T}) {rw_num}P Costo' for T in SA_INITIAL_TEMPS]
        cols_exist_g5 = [c for c in sa_cost_cols if c in df_sa_cost_plot.columns]

        # <--- CORRECCIÓN: Graficar aunque falten columnas, si hay alguna
        if cols_exist_g5 and not df_sa_cost_plot[cols_exist_g5].isnull().all().all():
            df_sa_cost_plot[cols_exist_g5].plot(kind='bar', ax=ax, rot=0, width=0.8)
            ax.set_title(f'SA Costo ({start_label} - {rw_label})')
            ax.set_xlabel('Caso')
            ax.grid(axis='y', linestyle='--')
            ax.tick_params(axis='x', rotation=0)
            # Renombrar columnas para la leyenda (solo las que existen)
            legend_labels = [f'T={c.split("T")[1].split(")")[0]}' for c in cols_exist_g5]
            ax.legend(legend_labels, fontsize='small', title="T Inicial")
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


# Gráfico 6: Comparación Tiempos Ejecución (AVG/Valor)
# (Sin cambios, parece funcionar)
print("Generando Gráfico 6: Comparación de Tiempos de Ejecución (Promedio/Valor)...")
fig6, ax6 = plt.subplots(figsize=(18, 10))
cols_tiempo_g6 = [col for col in df_plot.columns if 'Tiempo' in col]
rename_map_g6 = {}
for col in cols_tiempo_g6:
    parts = col.split(' ')
    algo_label = parts[0]
    pista_label = parts[-2]
    short_label = f"{algo_label}-{pista_label}"
    if 'GRASP' in algo_label: short_label = f"GRASP{REPRESENTATIVE_GRASP_ITERS}(Avg)-{pista_label}"
    elif 'SA(Det' in algo_label: short_label = f"SA(Det T{REPRESENTATIVE_SA_TEMP})-{pista_label}"
    elif 'SA(Stoch' in algo_label: short_label = f"SA(Stoch T{REPRESENTATIVE_SA_TEMP})(Avg)-{pista_label}"
    elif 'HC(Det)' in algo_label: short_label = f"HC(Det)-{pista_label}"
    elif 'Stoch Avg' in col: short_label = f"StochBase(Avg)-{pista_label}" # Ajuste para estocástico
    rename_map_g6[col] = short_label

cols_exist_g6 = [col for col in cols_tiempo_g6 if col in df_plot.columns]
if cols_exist_g6:
    df_subset_g6 = df_plot[cols_exist_g6].rename(columns=rename_map_g6)
    df_subset_g6 = df_subset_g6.apply(pd.to_numeric, errors='coerce')
    if not df_subset_g6.isnull().all().all():
        df_subset_g6.plot(kind='bar', ax=ax6, rot=0, width=0.8)
        ax6.set_title('Comparación de Tiempos de Ejecución por Algoritmo y Caso (Resultados Promedio/Valor)')
        ax6.set_ylabel('Tiempo Promedio/Total (s)')
        ax6.set_xlabel('Caso de Prueba')
        ax6.grid(axis='y', linestyle='--')
        ax6.tick_params(axis='x', rotation=0)
        ax6.legend(title='Algoritmo - Pistas', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.80, 1]) # Ajustar para leyenda
        fig6_path = os.path.join(OUTPUT_GRAPH_DIR, f'06_tiempos_ejecucion_promedio_{timestamp}.png')
        plt.savefig(fig6_path); print(f"Gráfico guardado en: {fig6_path}")
    else:
         plt.close(fig6); print("Gráfico 6 (Tiempos Ejecución) omitido - todos los valores son NaN.")
else:
    plt.close(fig6); print("Gráfico 6 (Tiempos Ejecución) omitido - faltan columnas.")


# Gráfico 7: Boxplot Tiempos GRASP por Configuración
print("Generando Gráfico 7: Boxplot Tiempos GRASP por Configuración...")
# <--- CORRECCIÓN: Usar case_list
fig7, ax7 = plt.subplots(1, len(case_list), figsize=(6 * len(case_list), 7), sharey=True)
fig7.subplots_adjust(wspace=0.15)
plot_success_g7 = False
if len(case_list) == 1: ax7 = np.array([ax7]) # Asegurar iterable
fig7.suptitle(f'Distribución Tiempos GRASP ({NUM_GRASP_EXECUTIONS} Runs por Configuración)')

colors = plt.cm.viridis(np.linspace(0, 1, len(GRASP_RESTARTS_LIST)))

# <--- CORRECCIÓN: Iterar sobre case_list
for i, case_name in enumerate(case_list):
    if len(case_list) > 1: current_ax = ax7.flat[i]
    else: current_ax = ax7[0]
    data_to_plot_1p = []
    data_to_plot_2p = []
    labels_1p = []
    labels_2p = []
    positions_1p = []
    positions_2p = []
    color_map = {}
    base_pos = 0

    for k, restart_iters in enumerate(GRASP_RESTARTS_LIST):
        param_str = f"Restarts={restart_iters}, RCL={RCL_SIZE}, HC_iter={HC_MAX_ITER}"
        # 1 Pista
        times_1p = df_raw[(df_raw['Caso'] == case_name) & (df_raw['Algoritmo'] == 'GRASP') &
                          (df_raw['Pistas'] == 1) & (df_raw['Parametros'] == param_str)]['Tiempo_s'].dropna().tolist()
        if times_1p:
            data_to_plot_1p.append(times_1p); labels_1p.append(f'{restart_iters}r-1P')
            positions_1p.append(base_pos + 1); color_map[f'{restart_iters}r-1P'] = colors[k]
        # 2 Pistas
        times_2p = df_raw[(df_raw['Caso'] == case_name) & (df_raw['Algoritmo'] == 'GRASP') &
                          (df_raw['Pistas'] == 2) & (df_raw['Parametros'] == param_str)]['Tiempo_s'].dropna().tolist()
        if times_2p:
            data_to_plot_2p.append(times_2p); labels_2p.append(f'{restart_iters}r-2P')
            positions_2p.append(base_pos + 2); color_map[f'{restart_iters}r-2P'] = colors[k]
        base_pos += 3

    all_data = data_to_plot_1p + data_to_plot_2p
    all_labels = labels_1p + labels_2p
    all_positions = positions_1p + positions_2p

    if all_data:
        plot_success_g7 = True
        try:
            valid_indices = [idx for idx, data in enumerate(all_data) if data]
            if not valid_indices: continue

            filtered_data = [all_data[idx] for idx in valid_indices]
            filtered_labels = [all_labels[idx] for idx in valid_indices]
            filtered_positions = [all_positions[idx] for idx in valid_indices]

            bp = current_ax.boxplot(filtered_data, patch_artist=True, labels=filtered_labels, positions=filtered_positions, widths=0.6, showfliers=False)
            for j, patch in enumerate(bp['boxes']): patch.set_facecolor(color_map[filtered_labels[j]])
            current_ax.set_title(f'{case_name}')
            current_ax.grid(axis='y', linestyle='--')
            current_ax.tick_params(axis='x', rotation=45, labelsize=8)
            if filtered_positions:
                current_ax.set_xticks(filtered_positions)
                current_ax.set_xticklabels(filtered_labels, rotation=45, ha='right')
                current_ax.set_xlim(min(filtered_positions) - 1, max(filtered_positions) + 1)
        except Exception as e_bp:
            print(f"Error generando boxplot GRASP Tiempo para {case_name}: {e_bp}")
            current_ax.text(0.5, 0.5, 'Error en Boxplot', ha='center', va='center', transform=current_ax.transAxes)
            current_ax.set_title(f'{case_name} (Error)')
    else:
        current_ax.text(0.5, 0.5, 'Sin datos GRASP válidos', ha='center', va='center', transform=current_ax.transAxes)
        current_ax.set_title(f'{case_name} (Sin datos)')

    if i == 0 and plot_success_g7: current_ax.set_ylabel('Tiempo Ejecución GRASP (s)')

if plot_success_g7:
    handles = [plt.Rectangle((0,0),1,1, color=colors[k]) for k in range(len(GRASP_RESTARTS_LIST))]
    fig7.legend(handles, [f'{r} Restarts' for r in GRASP_RESTARTS_LIST], loc='upper right', title="Config. Restarts")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    fig7_path = os.path.join(OUTPUT_GRAPH_DIR, f'07_boxplot_grasp_tiempo_{timestamp}.png')
    plt.savefig(fig7_path); print(f"Gráfico guardado en: {fig7_path}")
else:
    print("No se generó Gráfico 7 (Boxplot GRASP Tiempo).")
    plt.close(fig7)


# Gráfico 8: Efecto Temperatura Inicial SA (Tiempo - Usando Avg Tiempo para Stoch)
print("Generando Gráfico 8: Efecto Temperatura Inicial SA (Tiempo - Avg para Stoch)...")
fig8, ax8 = plt.subplots(2, 2, figsize=(16, 10), sharey=False) # sharey=False para tiempos
fig8.subplots_adjust(hspace=0.3, wspace=0.15)
plot_success_g8 = False
start_points = [('Det', 'Deterministic'), ('Stoch', 'Stochastic')]
runways = [(1, '1 Pista'), (2, '2 Pistas')]
if not isinstance(ax8, np.ndarray): ax8 = ax8.reshape(2,2)

# Re-crear df_sa_time_plot
df_sa_time_plot = pd.DataFrame(index=case_list)
for T_init in SA_INITIAL_TEMPS:
    for pistas in [1, 2]:
        # Desde Det
        col_det_time = f'SA(Det T{T_init}) {pistas}P Tiempo'
        if col_det_time in df_plot.columns:
            df_sa_time_plot[col_det_time] = df_plot[col_det_time]
        # Desde Stoch (promedio)
        if not stats_sa_stoch.empty:
            param_str_sa = f"T_init={T_init}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
            avg_time = stats_sa_stoch.get(('avg_time', pistas, param_str_sa), pd.Series(index=case_list, dtype=float))
            df_sa_time_plot[f'SA(Stoch T{T_init}) {pistas}P Tiempo'] = avg_time.reindex(case_list)

df_sa_time_plot.replace(INFINITO_COSTO, np.nan, inplace=True)


for row, (start_key, start_label) in enumerate(start_points):
    for col, (rw_num, rw_label) in enumerate(runways):
        ax = ax8[row, col]
        sa_time_cols = [f'SA({start_key} T{T}) {rw_num}P Tiempo' for T in SA_INITIAL_TEMPS]
        cols_exist_g8 = [c for c in sa_time_cols if c in df_sa_time_plot.columns]

        # <--- CORRECCIÓN: Graficar si existen *algunas* columnas, no necesariamente todas
        if cols_exist_g8 and not df_sa_time_plot[cols_exist_g8].isnull().all().all():
            # Asegurarse que sean numéricos antes de plotear
            df_sa_time_plot[cols_exist_g8].apply(pd.to_numeric, errors='coerce').plot(kind='bar', ax=ax, rot=0, width=0.8)
            ax.set_title(f'SA Tiempo ({start_label} - {rw_label})')
            ax.set_xlabel('Caso')
            ax.grid(axis='y', linestyle='--')
            ax.tick_params(axis='x', rotation=0)
            # <--- CORRECCIÓN: Crear leyenda solo para columnas existentes
            legend_labels = [f'T={c.split("T")[1].split(")")[0]}' for c in cols_exist_g8]
            ax.legend(legend_labels, fontsize='small', title="T Inicial")
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
    plt.close(fig8); print("Gráfico 8 (SA Tiempo) omitido - faltan columnas o datos.")


# Gráfico 9: Mejora Porcentual vs. Punto de Partida
# (La lógica de cálculo de mejora parece compleja y puede fallar si faltan costos iniciales)
# (Se requiere la columna 'Mejora_%' ya calculada numéricamente)
print("Generando Gráfico 9: Mejora Porcentual de Metaheurísticas...")
fig9, ax9 = plt.subplots(2, 2, figsize=(15, 10), sharey=True)
fig9.subplots_adjust(hspace=0.3, wspace=0.15)
plot_success_g9 = False
if not isinstance(ax9, np.ndarray): ax9 = ax9.reshape(2,2)

# Asegurar que la columna Mejora_%_Num exista y sea numérica
if 'Mejora_%' in df_raw.columns:
     df_raw['Mejora_%_Num'] = pd.to_numeric(df_raw['Mejora_%'], errors='coerce')
else:
     df_raw['Mejora_%_Num'] = np.nan # Crearla si no existe para evitar errores posteriores


# 1. HC (desde Det y Stoch)
ax = ax9[0, 0]
try:
    hc_det_imp = df_raw[(df_raw['Algoritmo'] == 'HC') & (df_raw['Punto Partida'] == 'Deterministic')].set_index(['Caso', 'Pistas'])['Mejora_%_Num'].unstack()
    hc_stoch_imp = df_raw[(df_raw['Algoritmo'] == 'HC') & (df_raw['Punto Partida'].str.contains('Stochastic', na=False))].groupby(['Caso', 'Pistas'])['Mejora_%_Num'].mean().unstack()

    hc_imp = pd.DataFrame({
        'HC(Det)-1P': hc_det_imp.get(1),
        'HC(Stoch Avg)-1P': hc_stoch_imp.get(1),
        'HC(Det)-2P': hc_det_imp.get(2),
        'HC(Stoch Avg)-2P': hc_stoch_imp.get(2)
    }).reindex(case_list) # Asegurar orden de casos

    if not hc_imp.isnull().all().all():
        hc_imp.plot(kind='bar', ax=ax, rot=0); plot_success_g9 = True
        ax.set_title('Mejora % - Hill Climbing')
        ax.set_ylabel('Mejora (%) vs Inicial'); ax.set_xlabel('Caso')
        ax.legend(fontsize='small'); ax.grid(axis='y', linestyle='--')
    else: raise ValueError("Todos los datos de mejora HC son NaN")
except Exception as e:
     print(f"Error preparando datos para Gráfico 9.1 (HC Imp): {e}")
     ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=ax.transAxes); ax.set_title('Mejora % - Hill Climbing (Faltan Datos)')

# 2. GRASP (vs Best Stoch) - Promedio de mejora de las runs
ax = ax9[0, 1]
try:
    grasp_imp_data = {}
    df_grasp_filtered_imp = df_raw[(df_raw['Algoritmo'] == 'GRASP') & df_raw['Parametros'].notna()].copy()

    if not df_grasp_filtered_imp.empty:
         for pistas in [1,2]:
             stoch_min_costs = df_plot[f'Stoch Min {pistas}P Costo'] # Usar el min calculado antes
             # Mapear costo inicial a cada fila de GRASP
             df_grasp_filtered_imp.loc[df_grasp_filtered_imp['Pistas'] == pistas, 'Costo_Inicial_Stoch_Min'] = df_grasp_filtered_imp['Caso'].map(stoch_min_costs)

         # Calcular mejora vs Stoch Min
         df_grasp_filtered_imp['Mejora_%_vs_StochMin'] = 100 * (df_grasp_filtered_imp['Costo_Inicial_Stoch_Min'] - df_grasp_filtered_imp['Costo Final']) / df_grasp_filtered_imp['Costo_Inicial_Stoch_Min'].replace(0, np.nan)
         df_grasp_filtered_imp.replace([np.inf, -np.inf], np.nan, inplace=True) # Limpiar inf generados por división

         # Agrupar por Parametros y Pistas para el gráfico
         grasp_imp_avg = df_grasp_filtered_imp.groupby(['Caso', 'Pistas', 'Parametros'])['Mejora_%_vs_StochMin'].mean().unstack(level=[1,2])

         # Seleccionar columnas para el gráfico (todas las configs GRASP)
         cols_grasp_imp = []
         rename_map_g9_grasp = {}
         for grasp_iter in GRASP_RESTARTS_LIST:
             param_str = f"Restarts={grasp_iter}, RCL={RCL_SIZE}, HC_iter={HC_MAX_ITER}"
             for pistas in [1,2]:
                 if ('Mejora_%_vs_StochMin', pistas, param_str) in grasp_imp_avg.columns:
                     col_name = ('Mejora_%_vs_StochMin', pistas, param_str)
                     cols_grasp_imp.append(col_name)
                     rename_map_g9_grasp[col_name] = f'GRASP {grasp_iter}r Avg Imp-{pistas}P'

         if cols_grasp_imp:
             grasp_imp_plot = grasp_imp_avg[cols_grasp_imp].rename(columns=rename_map_g9_grasp).reindex(case_list)
             if not grasp_imp_plot.isnull().all().all():
                 grasp_imp_plot.plot(kind='bar', ax=ax, rot=0); plot_success_g9 = True
                 ax.set_title('Mejora % Promedio - GRASP vs Best Stoch')
                 ax.set_ylabel('Mejora Promedio (%)'); ax.set_xlabel('Caso')
                 ax.legend(fontsize='x-small', title="Config GRASP"); ax.grid(axis='y', linestyle='--') # Leyenda más pequeña
             else: raise ValueError("Todos los datos de mejora GRASP son NaN")
         else: raise ValueError("No se encontraron columnas de mejora GRASP")
    else: raise ValueError("No hay datos filtrados de GRASP para mejora")

except Exception as e:
     print(f"Error preparando datos para Gráfico 9.2 (GRASP Imp): {e}")
     ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=ax.transAxes); ax.set_title('Mejora % - GRASP (Faltan Datos)')


# 3. SA(Det) (vs Det)
ax = ax9[1, 0]
try:
    sa_det_imp = df_raw[(df_raw['Algoritmo'] == 'SA') & (df_raw['Punto Partida'] == 'Deterministic')].set_index(['Caso', 'Pistas', 'Parametros'])['Mejora_%_Num'].unstack(level=[1,2])
    sa_det_imp_rep = pd.DataFrame()
    cols_sa_det_imp = []
    rename_map_g9_sadet = {}
    param_str_rep = f"T_init={REPRESENTATIVE_SA_TEMP}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
    for pistas in [1,2]:
         col_tuple = ('Mejora_%_Num', pistas, param_str_rep)
         # Verificar si la tupla existe como columna en el MultiIndex
         if col_tuple in sa_det_imp.columns:
             col_name_new = f'SA(Det T{REPRESENTATIVE_SA_TEMP})-{pistas}P'
             sa_det_imp_rep[col_name_new] = sa_det_imp[col_tuple]
             cols_sa_det_imp.append(col_name_new)

    if not sa_det_imp_rep.empty and not sa_det_imp_rep.isnull().all().all():
        sa_det_imp_rep.reindex(case_list).plot(kind='bar', ax=ax, rot=0); plot_success_g9 = True
        ax.set_title(f'Mejora % - SA(Det T={REPRESENTATIVE_SA_TEMP}) vs Det')
        ax.set_ylabel('Mejora (%) vs Det'); ax.set_xlabel('Caso')
        ax.legend(fontsize='small'); ax.grid(axis='y', linestyle='--')
    else: raise ValueError("Todos los datos de mejora SA(Det) son NaN o faltan")
except Exception as e:
     print(f"Error preparando datos para Gráfico 9.3 (SA Det Imp): {e}")
     ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=ax.transAxes); ax.set_title(f'Mejora % - SA(Det T={REPRESENTATIVE_SA_TEMP}) (Faltan Datos)')


# 4. SA(Stoch) (vs Stoch) - Promedio de mejora
ax = ax9[1, 1]
try:
    sa_stoch_imp_avg = df_raw[(df_raw['Algoritmo'] == 'SA') & (df_raw['Punto Partida'].str.contains('Stochastic', na=False))].groupby(['Caso', 'Pistas', 'Parametros'])['Mejora_%_Num'].mean().unstack(level=[1,2])
    sa_stoch_imp_rep = pd.DataFrame()
    cols_sa_stoch_imp = []
    rename_map_g9_sastoch = {}
    param_str_rep = f"T_init={REPRESENTATIVE_SA_TEMP}, T_min={SA_T_MIN}, alpha={SA_ALPHA}, iter/T={SA_ITER_PER_TEMP}, neigh_att={SA_MAX_NEIGHBOR_ATTEMPTS}"
    for pistas in [1,2]:
        col_tuple = ('Mejora_%_Num', pistas, param_str_rep)
        if col_tuple in sa_stoch_imp_avg.columns:
             col_name_new = f'SA(Stoch T{REPRESENTATIVE_SA_TEMP} Avg Imp)-{pistas}P'
             sa_stoch_imp_rep[col_name_new] = sa_stoch_imp_avg[col_tuple]
             cols_sa_stoch_imp.append(col_name_new)

    if not sa_stoch_imp_rep.empty and not sa_stoch_imp_rep.isnull().all().all():
        sa_stoch_imp_rep.reindex(case_list).plot(kind='bar', ax=ax, rot=0); plot_success_g9 = True
        ax.set_title(f'Mejora % Promedio - SA(Stoch T={REPRESENTATIVE_SA_TEMP}) vs Stoch Ini')
        ax.set_ylabel('Mejora Promedio (%) vs Stoch'); ax.set_xlabel('Caso')
        ax.legend(fontsize='small'); ax.grid(axis='y', linestyle='--')
    else: raise ValueError("Todos los datos de mejora SA(Stoch) son NaN o faltan")
except Exception as e:
     print(f"Error preparando datos para Gráfico 9.4 (SA Stoch Imp): {e}")
     ax.text(0.5, 0.5, 'Sin datos válidos', ha='center', va='center', transform=ax.transAxes); ax.set_title(f'Mejora % - SA(Stoch T={REPRESENTATIVE_SA_TEMP}) (Faltan Datos)')


if plot_success_g9:
    fig9.suptitle('Mejora Porcentual Obtenida por Metaheurísticas vs. Soluciones Iniciales')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig9_path = os.path.join(OUTPUT_GRAPH_DIR, f'09_mejora_porcentual_{timestamp}.png')
    plt.savefig(fig9_path); print(f"Gráfico guardado en: {fig9_path}")
else:
    print("No se generó Gráfico 9 (Mejora Porcentual).")
    plt.close(fig9)


print("\n--- Generación de gráficos completada ---")
# plt.show() # Descomentar si quieres ver los gráficos interactivamente al final