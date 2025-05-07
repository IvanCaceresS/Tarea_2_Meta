import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict
import numpy as np 
# Mapeo global para el número de aviones por caso
AVIONES_MAP = {
    'case1.txt': 15,
    'case2.txt': 20,
    'case3.txt': 44,
    'case4.txt': 100
}

# Directorio para guardar los gráficos
GRAFICOS_DIR = "graficos" # Nueva carpeta para la versión corregida

# --- Definiciones de Nombres y Orden para Gráficos ---
# Nombres de display para leyendas (más cortos)
ALGO_DISPLAY_NAMES = {
    'GD': 'GD',
    'GE_Solo': 'GE_Solo',
    'GRASP_HC_Det_0Restarts': 'GRASP_Det(0R)',
    'GRASP_HC_Estoc_R10': 'GRASP_E(10R)', 
    'GRASP_HC_Estoc_R25': 'GRASP_E(25R)', 
    'GRASP_HC_Estoc_R50': 'GRASP_E(50R)', 
    'GRASP_HC_Estoc_R100': 'GRASP_E(100R)',
    'TS_Det': 'TS_Det', 
    'TS_Estoc': 'TS_Estoc' 
}

# Orden base para los algoritmos (usando las claves de ALGO_DISPLAY_NAMES)
ALGO_ORDER_BASE = [
    'GD', 'GE_Solo', 'GRASP_HC_Det_0Restarts', 
    'GRASP_HC_Estoc_R10', 'GRASP_HC_Estoc_R25', 
    'GRASP_HC_Estoc_R50', 'GRASP_HC_Estoc_R100',
    'TS_Det', 'TS_Estoc'
]

# Orden para los gráficos específicos de GRASP (usando los valores de ALGO_DISPLAY_NAMES)
GRASP_VARIANTS_ORDER_DISPLAY = [
    ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R10'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R25'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R50'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R100']
]
# Orden para los gráficos específicos de TS (usando los valores de ALGO_DISPLAY_NAMES)
TS_VARIANTS_ORDER_DISPLAY = [
    ALGO_DISPLAY_NAMES['TS_Det'],
    ALGO_DISPLAY_NAMES['TS_Estoc']
]

# Paletas de colores (se ajustarán dinámicamente en las funciones de ploteo)
PALETA_TS_VARIANTS = sns.color_palette("coolwarm", 2) 


def parse_details(details_str):
    """Parsea la cadena 'DetallesParametros' en un diccionario."""
    params = { 
        'K_RCL': None, 'IterGRASP': None, 'IterHC': None, 'Semilla': None, 
        'SolInicial': None, 'MaxIter': None, 'Ten': None, 'MaxNoImp':None, 
        'AspirationCriteria': None, 'TipoVecindad': None, 'CfgID': None
    }
    if pd.isna(details_str) or details_str == 'N/A' or not details_str:
        return params
    
    parts = details_str.split(';')
    for part in parts:
        key_value = part.split(':', 1)
        if len(key_value) == 2:
            key, value = key_value[0].strip(), key_value[1].strip()
            
            current_val = value
            try:
                num_val = float(value)
                if num_val == int(num_val):
                    current_val = int(num_val)
                else:
                    current_val = num_val
            except ValueError:
                 if key == 'AspirationCriteria': 
                     current_val = value.lower() == 'true' if value.lower() in ['true', 'false'] else value
            
            if key == 'Inicio': 
                params['SolInicial'] = current_val
            elif key in params:
                 params[key] = current_val
    return params

def get_base_algo_name(algo_name_from_csv):
    """ Extrae el nombre base del algoritmo para agrupación y mapeo a ALGO_DISPLAY_NAMES. """
    if algo_name_from_csv.startswith("GRASP_HC_Estoc_R"):
        match = re.match(r"(GRASP_HC_Estoc_R\d+)", algo_name_from_csv)
        if match: return match.group(1) 
    elif algo_name_from_csv.startswith("TS_GD_"): 
        return "TS_Det" 
    elif algo_name_from_csv.startswith("TS_GE_"): 
        return "TS_Estoc" 
    return algo_name_from_csv 

def load_and_preprocess_data(csv_filepath):
    """Carga y preprocesa los datos del CSV."""
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: El archivo '{csv_filepath}' no fue encontrado.")
        return None

    df['CostoPenalizadoSolucion'] = pd.to_numeric(df['CostoPenalizadoSolucion'], errors='coerce')
    df['TiempoComputacional_seg'] = pd.to_numeric(df['TiempoComputacional_seg'], errors='coerce')
    df['NumPistas'] = df['NumPistas'].astype(int)
    df['EsEstrictamenteFactible'] = df['EsEstrictamenteFactible'].apply(lambda x: str(x).strip().lower() == 'true')

    details_parsed_list = [parse_details(row['DetallesParametros']) for _, row in df.iterrows()]
    details_df = pd.DataFrame(details_parsed_list)
    df = pd.concat([df.reset_index(drop=True), details_df.reset_index(drop=True)], axis=1)
    
    df['AlgoBase'] = df['Algoritmo'].apply(get_base_algo_name)
    df['AlgoritmoDisplay'] = df['AlgoBase'].map(ALGO_DISPLAY_NAMES).fillna(df['AlgoBase'])
    
    def case_sort_key(case_name):
        match = re.search(r'\d+', str(case_name)) 
        return int(match.group(0)) if match else float('inf')
    
    df['NombreCaso_SortKey'] = df['NombreCaso'].apply(case_sort_key)
    df = df.sort_values(by=['NombreCaso_SortKey', 'NumPistas', 'Algoritmo']).reset_index(drop=True)
    df = df.drop(columns=['NombreCaso_SortKey'])
    return df

def get_ordered_hue_list(plot_df_alg_column_unique, algo_map_for_plot):
    base_algo_from_plot_name = {}
    for base_key, plot_name in algo_map_for_plot.items():
        base_algo_from_plot_name[plot_name] = base_key

    ordered_list = sorted(
        plot_df_alg_column_unique,
        key=lambda name: ALGO_ORDER_BASE.index(base_algo_from_plot_name.get(name))
                         if base_algo_from_plot_name.get(name) in ALGO_ORDER_BASE else float('inf')
    )
    return ordered_list

def plot_cost_comparison_main_algos(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 10)) 
    
    df_factible = df[df['EsEstrictamenteFactible']].copy()
    if df_factible.empty:
        print("Advertencia para plot_cost_comparison_main_algos: No hay soluciones factibles.")
        return

    algo_map_plot_cost = { 
        'GD': ALGO_DISPLAY_NAMES['GD'],
        'GE_Solo': ALGO_DISPLAY_NAMES['GE_Solo'] + ' (Mejor)',
        'GRASP_HC_Det_0Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
        'GRASP_HC_Estoc_R10': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R10'] + ' (Mejor)',
        'GRASP_HC_Estoc_R25': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R25'] + ' (Mejor)',
        'GRASP_HC_Estoc_R50': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R50'] + ' (Mejor)',
        'GRASP_HC_Estoc_R100': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R100'] + ' (Mejor)',
        'TS_Det': ALGO_DISPLAY_NAMES['TS_Det'] + ' (Mejor Cfg)', 
        'TS_Estoc': ALGO_DISPLAY_NAMES['TS_Estoc'] + ' (Mejor Cfg)' 
    }
    
    df_grouped_min = df_factible.groupby(['NombreCaso', 'NumPistas', 'AlgoBase'])['CostoPenalizadoSolucion'].min().reset_index()
    df_grouped_min['AlgoritmoPlotName'] = df_grouped_min['AlgoBase'].map(algo_map_plot_cost)
    df_grouped_min.dropna(subset=['AlgoritmoPlotName'], inplace=True) 

    if df_grouped_min.empty:
        print("Advertencia para plot_cost_comparison_main_algos: No hay datos para graficar después del mapeo.")
        return

    df_grouped_min['Caso_Pistas'] = df_grouped_min['NombreCaso'] + " (" + df_grouped_min['NumPistas'].astype(str) + "P)"
    
    def sort_key_caso_pistas_str(caso_pistas_str):
        nombre_caso, pistas_str = caso_pistas_str.split(" (")
        match = re.search(r'\d+', nombre_caso)
        num_caso_val = int(match.group(0)) if match else float('inf')
        num_pistas_val = int(pistas_str.replace("P)", ""))
        return (num_caso_val, num_pistas_val)
    sorted_caso_pistas = sorted(df_grouped_min['Caso_Pistas'].unique(), key=sort_key_caso_pistas_str)
    
    hue_order_for_plot = get_ordered_hue_list(df_grouped_min['AlgoritmoPlotName'].unique(), algo_map_plot_cost)
    current_palette = sns.color_palette("tab10", n_colors=len(hue_order_for_plot))

    sns.barplot(x='Caso_Pistas', y='CostoPenalizadoSolucion', hue='AlgoritmoPlotName', data=df_grouped_min, 
                order=sorted_caso_pistas, hue_order=hue_order_for_plot, palette=current_palette)
    plt.title('Comparación de Mejor Costo Penalizado Factible por Algoritmo', fontsize=16)
    plt.ylabel('Costo Penalizado Solución (Mejor Factible)', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    min_cost_abs = df_factible['CostoPenalizadoSolucion'].abs().replace(0, np.nan).min() 
    linthresh_val = min_cost_abs if pd.notna(min_cost_abs) and min_cost_abs > 0 else 10
    plt.yscale('symlog', linthresh=linthresh_val) 
    plt.legend(title='Algoritmo', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.80, 1]) 
    plt.savefig(os.path.join(output_dir, "1_comparacion_costos_algoritmos.png"), dpi=300)
    plt.close()
    print("Gráfico '1_comparacion_costos_algoritmos.png' generado.")


def plot_time_comparison_main_algos(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(18, 10))
    
    algo_map_plot_time = { 
        'GD': ALGO_DISPLAY_NAMES['GD'],
        'GE_Solo': ALGO_DISPLAY_NAMES['GE_Solo'] + ' (Prom.)',
        'GRASP_HC_Det_0Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
        'GRASP_HC_Estoc_R10': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R10'] + ' (Prom.)',
        'GRASP_HC_Estoc_R25': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R25'] + ' (Prom.)',
        'GRASP_HC_Estoc_R50': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R50'] + ' (Prom.)',
        'GRASP_HC_Estoc_R100': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_R100'] + ' (Prom.)',
        'TS_Det': ALGO_DISPLAY_NAMES['TS_Det'] + ' (Prom. Cfg)', 
        'TS_Estoc': ALGO_DISPLAY_NAMES['TS_Estoc'] + ' (Prom. Cfg)' 
    }

    df_time_avg = df.groupby(['NombreCaso', 'NumPistas', 'AlgoBase'])['TiempoComputacional_seg'].mean().reset_index()
    df_time_avg['AlgoritmoPlotName'] = df_time_avg['AlgoBase'].map(algo_map_plot_time)
    df_time_avg.dropna(subset=['AlgoritmoPlotName'], inplace=True)
    
    if df_time_avg.empty:
        print("Advertencia para plot_time_comparison_main_algos: No hay datos para graficar después del mapeo.")
        return

    df_time_avg['Caso_Pistas'] = df_time_avg['NombreCaso'] + " (" + df_time_avg['NumPistas'].astype(str) + "P)"

    def sort_key_caso_pistas_str(caso_pistas_str):
        nombre_caso, pistas_str = caso_pistas_str.split(" (")
        match = re.search(r'\d+', nombre_caso)
        num_caso_val = int(match.group(0)) if match else float('inf')
        num_pistas_val = int(pistas_str.replace("P)", ""))
        return (num_caso_val, num_pistas_val)
    sorted_caso_pistas = sorted(df_time_avg['Caso_Pistas'].unique(), key=sort_key_caso_pistas_str)
    
    hue_order_for_plot = get_ordered_hue_list(df_time_avg['AlgoritmoPlotName'].unique(), algo_map_plot_time)
    current_palette = sns.color_palette("tab10", n_colors=len(hue_order_for_plot))

    sns.barplot(x='Caso_Pistas', y='TiempoComputacional_seg', hue='AlgoritmoPlotName', data=df_time_avg, 
                order=sorted_caso_pistas, hue_order=hue_order_for_plot, palette=current_palette)
    plt.title('Comparación de Tiempo Computacional Promedio por Algoritmo', fontsize=16)
    plt.ylabel('Tiempo Computacional Promedio (s)', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale('log') 
    plt.legend(title='Algoritmo', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9, title_fontsize=11)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.savefig(os.path.join(output_dir, "2_comparacion_tiempos_algoritmos.png"), dpi=300)
    plt.close()
    print("Gráfico '2_comparacion_tiempos_algoritmos.png' generado.")


def plot_ge_solo_boxplot_costos(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    df_ge_factible = df[(df['AlgoBase'] == 'GE_Solo') & df['EsEstrictamenteFactible']].copy()
    
    if df_ge_factible.empty:
        print("Advertencia para plot_ge_solo_boxplot_costos: No hay datos GE_Solo factibles.")
        return

    df_ge_factible['Caso_Pistas'] = df_ge_factible['NombreCaso'] + " (" + df_ge_factible['NumPistas'].astype(str) + "P)"
    
    def sort_key_caso_pistas_str(caso_pistas_str):
        nombre_caso, pistas_str = caso_pistas_str.split(" (")
        match = re.search(r'\d+', nombre_caso)
        num_caso_val = int(match.group(0)) if match else float('inf')
        num_pistas_val = int(pistas_str.replace("P)", ""))
        return (num_caso_val, num_pistas_val)
    sorted_caso_pistas = sorted(df_ge_factible['Caso_Pistas'].unique(), key=sort_key_caso_pistas_str)

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='Caso_Pistas', y='CostoPenalizadoSolucion', data=df_ge_factible, 
                order=sorted_caso_pistas, whis=[5, 95], palette="coolwarm") 
    plt.title('GE_Solo: Distribución de Costos Factibles por Semilla', fontsize=16)
    plt.ylabel('Costo Penalizado Solución', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_ge_solo_boxplot_costos.png"), dpi=300)
    plt.close()
    print("Gráfico '3_ge_solo_boxplot_costos.png' generado.")


def plot_grasp_restarts_vs_cost_boxplot(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    
    grasp_algos_base_keys = [ 
        'GRASP_HC_Det_0Restarts', 'GRASP_HC_Estoc_R10', 'GRASP_HC_Estoc_R25', 
        'GRASP_HC_Estoc_R50', 'GRASP_HC_Estoc_R100'
    ]
    df_grasp_factible = df[
        df['AlgoBase'].isin(grasp_algos_base_keys) & 
        df['EsEstrictamenteFactible']
    ].copy()

    if df_grasp_factible.empty:
        print("Advertencia para plot_grasp_restarts_vs_cost_boxplot: No hay datos GRASP factibles.")
        return
    
    grasp_order_display_actual = [name for name in GRASP_VARIANTS_ORDER_DISPLAY if name in df_grasp_factible['AlgoritmoDisplay'].unique()]
    if not grasp_order_display_actual: 
         print("Advertencia para plot_grasp_restarts_vs_cost_boxplot: No se encontraron variantes GRASP válidas en AlgoritmoDisplay.")
         return
    df_grasp_factible['AlgoritmoDisplay'] = pd.Categorical(df_grasp_factible['AlgoritmoDisplay'], categories=grasp_order_display_actual, ordered=True)
    df_grasp_factible.dropna(subset=['AlgoritmoDisplay'], inplace=True) 

    sorted_nombre_caso = sorted(df_grasp_factible['NombreCaso'].unique(), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else x)
    
    current_grasp_palette = sns.color_palette("viridis", n_colors=len(grasp_order_display_actual))

    g = sns.catplot(x='NombreCaso', y='CostoPenalizadoSolucion', hue='AlgoritmoDisplay', 
                    col='NumPistas', data=df_grasp_factible, kind='box', 
                    height=5.5, aspect=1.2, legend_out=True, whis=[5,95], 
                    order=sorted_nombre_caso, hue_order=grasp_order_display_actual, palette=current_grasp_palette,
                    sharey=False) 
    
    g.set_axis_labels("Caso de Prueba", "Costo Penalizado (Factible)", fontsize=12)
    g.set_titles("Pistas: {col_name}", size=14)
    g.fig.suptitle('GRASP: Distribución de Costos por Variante y Nº de Restarts', y=1.03, fontsize=16)
    g.despine(left=True)
    if g.legend: 
      g.legend.set_title("Variante GRASP")
    
    for ax_idx, ax in enumerate(g.axes.flat):
        if not ax.has_data(): continue 
        current_num_pistas = g.col_names[ax_idx % len(g.col_names)] 
        
        subplot_data = df_grasp_factible[df_grasp_factible['NumPistas'] == current_num_pistas]

        if subplot_data.empty: continue

        min_val = subplot_data['CostoPenalizadoSolucion'].min()
        max_val = subplot_data['CostoPenalizadoSolucion'].max()
        
        if pd.notna(min_val) and pd.notna(max_val) and min_val > 0:
            if max_val / min_val > 100: 
                 min_positive = subplot_data.loc[subplot_data['CostoPenalizadoSolucion'] > 0, 'CostoPenalizadoSolucion'].min()
                 linthresh = max(1, min_positive / 10) if pd.notna(min_positive) and min_positive > 0 else 1
                 try:
                     ax.set_yscale('symlog', linthresh=linthresh)
                 except ValueError: 
                     ax.set_yscale('log')
            else: 
                 ax.set_yscale('log')
        elif pd.notna(min_val) and min_val <= 0 : 
            min_positive_abs = subplot_data.loc[subplot_data['CostoPenalizadoSolucion'] != 0, 'CostoPenalizadoSolucion'].abs().min()
            linthresh = max(1, min_positive_abs / 10) if pd.notna(min_positive_abs) and min_positive_abs > 0 else 1
            try:
                ax.set_yscale('symlog', linthresh=linthresh)
            except ValueError:
                pass 
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "4_grasp_restarts_vs_cost_boxplot.png"), dpi=300)
    plt.close()
    print("Gráfico '4_grasp_restarts_vs_cost_boxplot.png' generado.")


def plot_grasp_restarts_vs_time(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    
    grasp_algos_base_keys = [
        'GRASP_HC_Det_0Restarts', 'GRASP_HC_Estoc_R10', 'GRASP_HC_Estoc_R25', 
        'GRASP_HC_Estoc_R50', 'GRASP_HC_Estoc_R100'
    ]
    df_grasp = df[df['AlgoBase'].isin(grasp_algos_base_keys)].copy()

    if df_grasp.empty:
        print("Advertencia para plot_grasp_restarts_vs_time: No hay datos GRASP para graficar tiempos.")
        return

    df_grasp_time_avg = df_grasp.groupby(['NombreCaso', 'NumPistas', 'AlgoBase'])['TiempoComputacional_seg'].mean().reset_index()
    df_grasp_time_avg['Caso_Pistas'] = df_grasp_time_avg['NombreCaso'] + " (" + df_grasp_time_avg['NumPistas'].astype(str) + "P)"
    
    df_grasp_time_avg['VarianteGRASP'] = df_grasp_time_avg['AlgoBase'].map(ALGO_DISPLAY_NAMES)
    df_grasp_time_avg.dropna(subset=['VarianteGRASP'], inplace=True) 
    
    grasp_order_display_actual = [name for name in GRASP_VARIANTS_ORDER_DISPLAY if name in df_grasp_time_avg['VarianteGRASP'].unique()]
    if not grasp_order_display_actual: 
        print("Advertencia para plot_grasp_restarts_vs_time: No se encontraron variantes GRASP válidas en los datos agrupados después del mapeo.")
        return 
    df_grasp_time_avg['VarianteGRASP'] = pd.Categorical(df_grasp_time_avg['VarianteGRASP'], categories=grasp_order_display_actual, ordered=True)
    
    def sort_key_caso_pistas_str(caso_pistas_str):
        nombre_caso, pistas_str = caso_pistas_str.split(" (")
        match = re.search(r'\d+', nombre_caso)
        num_caso_val = int(match.group(0)) if match else float('inf')
        num_pistas_val = int(pistas_str.replace("P)", ""))
        return (num_caso_val, num_pistas_val)
    sorted_caso_pistas = sorted(df_grasp_time_avg['Caso_Pistas'].unique(), key=sort_key_caso_pistas_str)

    current_grasp_palette = sns.color_palette("viridis", n_colors=len(grasp_order_display_actual))

    plt.figure(figsize=(15, 8))
    sns.barplot(x='Caso_Pistas', y='TiempoComputacional_seg', hue='VarianteGRASP', 
                data=df_grasp_time_avg, order=sorted_caso_pistas, hue_order=grasp_order_display_actual, palette=current_grasp_palette)
    plt.title('Tiempo Computacional Promedio para Variantes de GRASP', fontsize=16)
    plt.ylabel('Tiempo Computacional Promedio (s)', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale('log')
    plt.legend(title='Variante GRASP', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, "5_grasp_restarts_vs_time.png"), dpi=300)
    plt.close()
    print("Gráfico '5_grasp_restarts_vs_time.png' generado.")

def plot_ts_tenure_vs_cost(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    df_ts_factible = df[
        df['AlgoBase'].isin(['TS_Det', 'TS_Estoc']) & 
        df['EsEstrictamenteFactible'] & 
        df['Ten'].notna() 
    ].copy()
    df_ts_factible['Ten'] = pd.to_numeric(df_ts_factible['Ten'], errors='coerce')
    df_ts_factible.dropna(subset=['Ten'], inplace=True) 

    if df_ts_factible.empty:
        print("Advertencia para plot_ts_tenure_vs_cost: No hay datos TS factibles con Tenure numérico.")
        return
        
    df_ts_agg_min = df_ts_factible.groupby(
        ['NombreCaso', 'NumPistas', 'AlgoBase', 'Ten'] 
    )['CostoPenalizadoSolucion'].min().reset_index()
    
    df_ts_agg_min['VarianteTS'] = df_ts_agg_min['AlgoBase'].map(ALGO_DISPLAY_NAMES)
    df_ts_agg_min.dropna(subset=['VarianteTS'], inplace=True)
    
    ts_variants_present = [v for v in TS_VARIANTS_ORDER_DISPLAY if v in df_ts_agg_min['VarianteTS'].unique()]
    if not ts_variants_present: 
        print("Advertencia para plot_ts_tenure_vs_cost: No se encontraron variantes TS válidas en AlgoritmoDisplay.")
        return 
    df_ts_agg_min['VarianteTS'] = pd.Categorical(df_ts_agg_min['VarianteTS'], categories=ts_variants_present, ordered=True)
    
    sorted_nombre_caso = sorted(df_ts_agg_min['NombreCaso'].unique(), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else x)

    g = sns.relplot(
        data=df_ts_agg_min,
        x="Ten", y="CostoPenalizadoSolucion",
        hue="VarianteTS", style="VarianteTS", 
        col="NombreCaso", row="NumPistas",
        kind="line", marker="o", 
        height=4, aspect=1.1, legend="full",
        palette=PALETA_TS_VARIANTS, 
        hue_order=ts_variants_present, style_order=ts_variants_present,
        col_order = sorted_nombre_caso, 
        facet_kws={'sharey': False, 'sharex': True, 'margin_titles': True} 
    )

    g.set_axis_labels("Tamaño Lista Tabú (Ten)", "Mejor Costo Penalizado (Factible)", fontsize=12)
    g.set_titles("Caso: {col_name} | Pistas: {row_name}", size=14) 
    g.fig.suptitle('Tabu Search: Efecto del Tamaño de Lista Tabú (Ten) en el Mejor Costo', y=1.03, fontsize=16)
    if g.legend: g.legend.set_title("Variante TS")
    
    for ax in g.axes.flat:
        if not ax.has_data(): continue
        
        title_text = ax.get_title()
        current_col_name_from_title = None
        current_row_name_from_title = None

        try:
            name_part_str, pistes_part_str = title_text.split(' | ', 1)
            
            if name_part_str.startswith("Caso: ") and pistes_part_str.startswith("Pistas: "):
                current_col_name_from_title = name_part_str.split(': ', 1)[1]
                current_row_name_from_title = int(pistes_part_str.split(': ', 1)[1])
            else:
                continue
        except ValueError: 
            continue
        
        if current_col_name_from_title is None or current_row_name_from_title is None:
            continue
            
        relevant_data = df_ts_agg_min[
            (df_ts_agg_min['NombreCaso'] == current_col_name_from_title) &
            (df_ts_agg_min['NumPistas'] == current_row_name_from_title)
        ]['CostoPenalizadoSolucion']

        if relevant_data.empty: continue

        if any(relevant_data <= 0):
            min_positive_abs = relevant_data[relevant_data != 0].abs().min()
            linthresh = max(1, min_positive_abs / 10) if pd.notna(min_positive_abs) and min_positive_abs > 0 else 1
            try:
                ax.set_yscale('symlog', linthresh=linthresh)
            except ValueError: 
                ax.set_yscale('log') 
        elif relevant_data.min() > 0 :
            ax.set_yscale('log')

        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
        tenure_values = sorted(df_ts_agg_min['Ten'].unique())
        ax.set_xticks(tenure_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter()) 

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "6_ts_tenure_vs_cost.png"), dpi=300)
    plt.close()
    print("Gráfico '6_ts_tenure_vs_cost.png' generado.")


def plot_ts_tenure_vs_time(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    df_ts = df[
        df['AlgoBase'].isin(['TS_Det', 'TS_Estoc']) & 
        df['Ten'].notna()
    ].copy()
    df_ts['Ten'] = pd.to_numeric(df_ts['Ten'], errors='coerce')
    df_ts.dropna(subset=['Ten'], inplace=True)

    if df_ts.empty:
        print("Advertencia para plot_ts_tenure_vs_time: No hay datos TS con Tenure numérico.")
        return

    df_ts_time_avg = df_ts.groupby(
         ['NombreCaso', 'NumPistas', 'AlgoBase', 'Ten']
         )['TiempoComputacional_seg'].mean().reset_index()
    
    df_ts_time_avg['VarianteTS'] = df_ts_time_avg['AlgoBase'].map(ALGO_DISPLAY_NAMES)
    df_ts_time_avg.dropna(subset=['VarianteTS'], inplace=True)
    
    ts_variants_present = [v for v in TS_VARIANTS_ORDER_DISPLAY if v in df_ts_time_avg['VarianteTS'].unique()]
    if not ts_variants_present: 
        print("Advertencia para plot_ts_tenure_vs_time: No se encontraron variantes TS válidas en AlgoritmoDisplay.")
        return 
    df_ts_time_avg['VarianteTS'] = pd.Categorical(df_ts_time_avg['VarianteTS'], categories=ts_variants_present, ordered=True)
    
    sorted_nombre_caso = sorted(df_ts_time_avg['NombreCaso'].unique(), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else x)

    g = sns.relplot(
        data=df_ts_time_avg,
        x="Ten", y="TiempoComputacional_seg",
        hue="VarianteTS", style="VarianteTS",
        col="NombreCaso", row="NumPistas",
        kind="line", marker="o",
        height=4, aspect=1.1, legend="full",
        palette=PALETA_TS_VARIANTS, 
        hue_order=ts_variants_present, style_order=ts_variants_present,
        col_order = sorted_nombre_caso, 
        facet_kws={'sharey': False, 'sharex': True, 'margin_titles': True}
    )

    g.set_axis_labels("Tamaño Lista Tabú (Ten)", "Tiempo Computacional Promedio (s)", fontsize=12)
    g.set_titles("Caso: {col_name} | Pistas: {row_name}", size=14)
    g.fig.suptitle('Tabu Search: Efecto del Tamaño de Lista Tabú (Ten) en el Tiempo', y=1.03, fontsize=16)
    if g.legend: g.legend.set_title("Variante TS")

    for ax in g.axes.flat:
        if not ax.has_data(): continue
        ax.set_yscale('log')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, axis='y')
        title_text = ax.get_title()
        current_col_name_from_title = None
        current_row_name_from_title = None
        try:
            name_part_str, pistes_part_str = title_text.split(' | ', 1)
            if name_part_str.startswith("Caso: ") and pistes_part_str.startswith("Pistas: "):
                current_col_name_from_title = name_part_str.split(': ', 1)[1]
                current_row_name_from_title = int(pistes_part_str.split(': ', 1)[1])
        except ValueError:
            pass

        if current_col_name_from_title and current_row_name_from_title is not None:
            ax_specific_data = df_ts_time_avg[
                (df_ts_time_avg['NombreCaso'] == current_col_name_from_title) &
                (df_ts_time_avg['NumPistas'] == current_row_name_from_title)
            ]
            tenure_values_ax = sorted(ax_specific_data['Ten'].unique())
            if tenure_values_ax:
                 ax.set_xticks(tenure_values_ax)
            else:
                 ax.set_xticks(sorted(df_ts_time_avg['Ten'].unique()))
        else:
            ax.set_xticks(sorted(df_ts_time_avg['Ten'].unique()))

        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())


    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "7_ts_tenure_vs_time.png"), dpi=300)
    plt.close()
    print("Gráfico '7_ts_tenure_vs_time.png' generado.")


def main():
    if not os.path.exists(GRAFICOS_DIR):
        os.makedirs(GRAFICOS_DIR)
        print(f"Directorio '{GRAFICOS_DIR}' creado.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_filepath = os.path.join(script_dir, 'results', 'resultado.csv')
    
    if not os.path.exists(csv_filepath):
        print(f"Error CRÍTICO: El archivo CSV '{csv_filepath}' no se encuentra.")
        print(f"Buscado en: {os.path.abspath(csv_filepath)}")
        alternative_path = os.path.join(os.getcwd(), 'results', 'resultado.csv')
        if os.path.exists(alternative_path):
            print(f"Intentando con ruta alternativa: {alternative_path}")
            csv_filepath = alternative_path
        else:
            print(f"Ruta alternativa también falló: {alternative_path}")
            print("Asegúrate de que el archivo existe en la ruta especificada y tiene datos.")
            return

    df_resultados = load_and_preprocess_data(csv_filepath)

    if df_resultados is not None and not df_resultados.empty:
        plot_cost_comparison_main_algos(df_resultados.copy(), GRAFICOS_DIR)
        plot_time_comparison_main_algos(df_resultados.copy(), GRAFICOS_DIR)
        plot_ge_solo_boxplot_costos(df_resultados.copy(), GRAFICOS_DIR)
        plot_grasp_restarts_vs_cost_boxplot(df_resultados.copy(), GRAFICOS_DIR) 
        plot_grasp_restarts_vs_time(df_resultados.copy(), GRAFICOS_DIR)
        plot_ts_tenure_vs_cost(df_resultados.copy(), GRAFICOS_DIR)
        plot_ts_tenure_vs_time(df_resultados.copy(), GRAFICOS_DIR)

        print(f"\nTodos los gráficos solicitados han sido generados en la carpeta '{GRAFICOS_DIR}'.")
    else:
        print("No se pudieron generar gráficos debido a problemas con los datos o el archivo CSV.")

if __name__ == '__main__':
    main()
