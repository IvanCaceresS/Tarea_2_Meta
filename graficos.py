import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from collections import defaultdict
import numpy as np # Para NaN si es necesario

# Mapeo global para el número de aviones por caso
AVIONES_MAP = {
    'case1.txt': 15,
    'case2.txt': 20,
    'case3.txt': 44,
    'case4.txt': 100
}

# Directorio para guardar los gráficos
GRAFICOS_DIR = "graficos_v3" # Nueva carpeta para no sobrescribir

# Paleta de colores y orden de algoritmos para consistencia
PALETA_PRINCIPALES = sns.color_palette("muted", 6) 
PALETA_GRASP = sns.color_palette("viridis", 4) # Para las variantes de GRASP
PALETA_GE_SOLO = sns.color_palette("pastel", 1)


# Nombres de display para leyendas (más cortos)
ALGO_DISPLAY_NAMES = {
    'GD': 'GD',
    'GE_Solo': 'GE_Solo',
    'GRASP_HC_Det_0Restarts': 'GRASP_Det(0R)',
    'GRASP_HC_Estoc_10Restarts': 'GRASP_E(10R)',
    'GRASP_HC_Estoc_50Restarts': 'GRASP_E(50R)',
    'GRASP_HC_Estoc_100Restarts': 'GRASP_E(100R)'
}
# Orden para los gráficos principales
ALGO_ORDER_MAIN_DISPLAY = [
    ALGO_DISPLAY_NAMES['GD'],
    ALGO_DISPLAY_NAMES['GE_Solo'] + ' (Mejor)', # Añadir sufijo para el gráfico de comparación
    ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_10Restarts'] + ' (Mejor)',
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_50Restarts'] + ' (Mejor)',
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_100Restarts'] + ' (Mejor)'
]
# Orden para los gráficos específicos de GRASP
GRASP_VARIANTS_ORDER_DISPLAY = [
    ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_10Restarts'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_50Restarts'],
    ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_100Restarts']
]


def parse_details(details_str):
    params = {'K_RCL': None, 'IterGRASP': None, 'IterHC': None, 'Semilla': None, 'Inicio': None}
    if pd.isna(details_str) or details_str == 'N/A' or not details_str:
        return params
    
    parts = details_str.split(';')
    for part in parts:
        key_value = part.split(':', 1)
        if len(key_value) == 2:
            key, value = key_value[0].strip(), key_value[1].strip()
            try:
                if key == 'K_RCL': params['K_RCL'] = int(value)
                elif key == 'IterGRASP': params['IterGRASP'] = int(value)
                elif key == 'IterHC': params['IterHC'] = int(value)
                elif key == 'Semilla': params['Semilla'] = int(value)
                elif key == 'Inicio': params['Inicio'] = value 
            except ValueError:
                if key == 'Inicio': params['Inicio'] = value
    return params

def load_and_preprocess_data(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: El archivo '{csv_filepath}' no fue encontrado.")
        return None

    df['CostoPenalizadoSolucion'] = pd.to_numeric(df['CostoPenalizadoSolucion'], errors='coerce')
    df['TiempoComputacional_seg'] = pd.to_numeric(df['TiempoComputacional_seg'], errors='coerce')
    df['NumPistas'] = df['NumPistas'].astype(int)
    df['EsEstrictamenteFactible'] = df['EsEstrictamenteFactible'].map({'True': True, 'False': False, True: True, False: False})

    details_parsed_list = [parse_details(row['DetallesParametros']) for _, row in df.iterrows()]
    details_df = pd.DataFrame(details_parsed_list)
    df = pd.concat([df.reset_index(drop=True), details_df.reset_index(drop=True)], axis=1)
    
    def case_sort_key(case_name):
        match = re.search(r'\d+', str(case_name)) 
        return int(match.group(0)) if match else float('inf')
    
    df['NombreCaso_SortKey'] = df['NombreCaso'].apply(case_sort_key)
    df = df.sort_values(by=['NombreCaso_SortKey', 'NumPistas', 'Algoritmo']).reset_index(drop=True)
    df = df.drop(columns=['NombreCaso_SortKey'])
    
    df['AlgoritmoDisplay'] = df['Algoritmo'].map(ALGO_DISPLAY_NAMES).fillna(df['Algoritmo'])

    print("Columnas después del preprocesamiento:", df.columns.tolist())
    print(f"\nValores únicos en 'Algoritmo' para gráficos: {df['Algoritmo'].unique()}")
    return df

def plot_cost_comparison_main_algos(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 9)) 
    
    plot_data = []
    df_factible = df[df['EsEstrictamenteFactible'] == True].copy()
    if df_factible.empty:
        print("Advertencia para plot_cost_comparison_main_algos: No hay soluciones factibles.")
        return

    # Mapeo de algoritmos CSV a nombres para la leyenda de este gráfico específico
    algos_plot_specific_display = {
        'GD': ALGO_DISPLAY_NAMES['GD'],
        'GE_Solo': ALGO_DISPLAY_NAMES['GE_Solo'] + ' (Mejor)',
        'GRASP_HC_Det_0Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
        'GRASP_HC_Estoc_10Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_10Restarts'] + ' (Mejor)',
        'GRASP_HC_Estoc_50Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_50Restarts'] + ' (Mejor)',
        'GRASP_HC_Estoc_100Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_100Restarts'] + ' (Mejor)'
    }

    for nombre_caso in df_factible['NombreCaso'].unique():
        for num_pistas in df_factible['NumPistas'].unique():
            for algo_csv, algo_display_name in algos_plot_specific_display.items():
                costo_algo = df_factible[
                    (df_factible['Algoritmo'] == algo_csv) &
                    (df_factible['NombreCaso'] == nombre_caso) &
                    (df_factible['NumPistas'] == num_pistas)
                ]['CostoPenalizadoSolucion'].min() 
                
                if pd.notna(costo_algo):
                    plot_data.append({'NombreCaso': nombre_caso, 'NumPistas': num_pistas, 
                                      'Algoritmo': algo_display_name, 'Costo': costo_algo})
            
    if not plot_data:
        print("Advertencia para plot_cost_comparison_main_algos: No hay datos para graficar.")
        return

    plot_df = pd.DataFrame(plot_data)
    plot_df['Caso_Pistas'] = plot_df['NombreCaso'] + " (" + plot_df['NumPistas'].astype(str) + "P)"
    
    def sort_key_caso_pistas_str(caso_pistas_str):
        nombre_caso, pistas_str = caso_pistas_str.split(" (")
        match = re.search(r'\d+', nombre_caso)
        num_caso_val = int(match.group(0)) if match else float('inf')
        num_pistas_val = int(pistas_str.replace("P)", ""))
        return (num_caso_val, num_pistas_val)

    sorted_caso_pistas = sorted(plot_df['Caso_Pistas'].unique(), key=sort_key_caso_pistas_str)
    
    plot_df['Algoritmo'] = pd.Categorical(plot_df['Algoritmo'], categories=ALGO_ORDER_MAIN_DISPLAY, ordered=True)

    sns.barplot(x='Caso_Pistas', y='Costo', hue='Algoritmo', data=plot_df, 
                order=sorted_caso_pistas, hue_order=ALGO_ORDER_MAIN_DISPLAY, palette=PALETA_PRINCIPALES)
    plt.title('Comparación de Mejor Costo Penalizado Factible por Algoritmo', fontsize=16)
    plt.ylabel('Costo Penalizado Solución (Mejor Factible)', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale('log') # Usar escala logarítmica para el costo
    plt.legend(title='Algoritmo', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.83, 1]) 
    plt.savefig(os.path.join(output_dir, "1_comparacion_costos_algoritmos.png"), dpi=300)
    plt.close()
    print("Gráfico '1_comparacion_costos_algoritmos.png' generado.")


def plot_time_comparison_main_algos(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 9))
    
    plot_data = []
    # Mapeo para este gráfico específico
    algos_map_display_ordered_time = {
        'GD': ALGO_DISPLAY_NAMES['GD'],
        'GE_Solo': ALGO_DISPLAY_NAMES['GE_Solo'] + ' (Promedio)',
        'GRASP_HC_Det_0Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
        'GRASP_HC_Estoc_10Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_10Restarts'] + ' (Promedio)',
        'GRASP_HC_Estoc_50Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_50Restarts'] + ' (Promedio)',
        'GRASP_HC_Estoc_100Restarts': ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_100Restarts'] + ' (Promedio)'
    }
    # Orden para la leyenda de este gráfico
    algo_order_time_display = [
        ALGO_DISPLAY_NAMES['GD'],
        ALGO_DISPLAY_NAMES['GE_Solo'] + ' (Promedio)',
        ALGO_DISPLAY_NAMES['GRASP_HC_Det_0Restarts'],
        ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_10Restarts'] + ' (Promedio)',
        ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_50Restarts'] + ' (Promedio)',
        ALGO_DISPLAY_NAMES['GRASP_HC_Estoc_100Restarts'] + ' (Promedio)'
    ]


    for nombre_caso in df['NombreCaso'].unique():
        for num_pistas in df['NumPistas'].unique():
            for algo_csv, algo_display_name in algos_map_display_ordered_time.items():
                tiempos_algo = df[
                    (df['Algoritmo'] == algo_csv) &
                    (df['NombreCaso'] == nombre_caso) &
                    (df['NumPistas'] == num_pistas)
                ]['TiempoComputacional_seg']
                
                if not tiempos_algo.empty:
                    tiempo_promedio = tiempos_algo.mean()
                    if pd.notna(tiempo_promedio):
                        plot_data.append({'NombreCaso': nombre_caso, 'NumPistas': num_pistas, 
                                          'Algoritmo': algo_display_name, 'Tiempo': tiempo_promedio})
    
    if not plot_data:
        print("Advertencia para plot_time_comparison_main_algos: No hay datos para graficar.")
        return

    plot_df = pd.DataFrame(plot_data)
    plot_df['Caso_Pistas'] = plot_df['NombreCaso'] + " (" + plot_df['NumPistas'].astype(str) + "P)"

    def sort_key_caso_pistas_str(caso_pistas_str):
        nombre_caso, pistas_str = caso_pistas_str.split(" (")
        match = re.search(r'\d+', nombre_caso)
        num_caso_val = int(match.group(0)) if match else float('inf')
        num_pistas_val = int(pistas_str.replace("P)", ""))
        return (num_caso_val, num_pistas_val)
    sorted_caso_pistas = sorted(plot_df['Caso_Pistas'].unique(), key=sort_key_caso_pistas_str)
    
    plot_df['Algoritmo'] = pd.Categorical(plot_df['Algoritmo'], categories=algo_order_time_display, ordered=True)

    sns.barplot(x='Caso_Pistas', y='Tiempo', hue='Algoritmo', data=plot_df, 
                order=sorted_caso_pistas, hue_order=algo_order_time_display, palette=PALETA_PRINCIPALES)
    plt.title('Comparación de Tiempo Computacional Promedio por Algoritmo', fontsize=16)
    plt.ylabel('Tiempo Computacional Promedio (s)', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale('log') 
    plt.legend(title='Algoritmo', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.80, 1])
    plt.savefig(os.path.join(output_dir, "2_comparacion_tiempos_algoritmos.png"), dpi=300)
    plt.close()
    print("Gráfico '2_comparacion_tiempos_algoritmos.png' generado.")


def plot_ge_solo_boxplot_costos(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    df_ge_factible = df[(df['Algoritmo'] == 'GE_Solo') & (df['EsEstrictamenteFactible'] == True)].copy()
    
    if df_ge_factible.empty:
        print("Advertencia para plot_ge_solo_boxplot_costos: No hay datos GE_Solo factibles para graficar.")
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
                order=sorted_caso_pistas, whis=[5, 95], palette=PALETA_GE_SOLO * len(sorted_caso_pistas)) # Repetir color
    plt.title('GE_Solo: Distribución de Costos Factibles por Semilla', fontsize=16)
    plt.ylabel('Costo Penalizado Solución', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_ge_solo_boxplot_costos.png"), dpi=300)
    plt.close()
    print("Gráfico '3_ge_solo_boxplot_costos.png' generado.")


def plot_grasp_restarts_vs_cost_boxplot(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    grasp_algos_csv_names = [
        'GRASP_HC_Det_0Restarts', 
        'GRASP_HC_Estoc_10Restarts', 
        'GRASP_HC_Estoc_50Restarts', 
        'GRASP_HC_Estoc_100Restarts'
    ]
    df_grasp_factible = df[
        df['Algoritmo'].isin(grasp_algos_csv_names) & 
        (df['EsEstrictamenteFactible'] == True)
    ].copy()

    if df_grasp_factible.empty:
        print("Advertencia para plot_grasp_restarts_vs_cost_boxplot: No hay datos GRASP factibles.")
        return

    df_grasp_factible['VarianteGRASP'] = df_grasp_factible['Algoritmo'].map(ALGO_DISPLAY_NAMES)
    df_grasp_factible['VarianteGRASP'] = pd.Categorical(df_grasp_factible['VarianteGRASP'], categories=GRASP_VARIANTS_ORDER_DISPLAY, ordered=True)

    sorted_nombre_caso = sorted(df_grasp_factible['NombreCaso'].unique(), key=lambda x: int(re.search(r'\d+', x).group(0)) if re.search(r'\d+', x) else x)

    g = sns.catplot(x='NombreCaso', y='CostoPenalizadoSolucion', hue='VarianteGRASP', 
                    col='NumPistas', data=df_grasp_factible, kind='box', 
                    height=5.5, aspect=1.2, legend_out=True, whis=[5,95], # Mostrar percentiles 5 y 95
                    order=sorted_nombre_caso, hue_order=GRASP_VARIANTS_ORDER_DISPLAY, palette=PALETA_GRASP) 
    
    g.set_axis_labels("Caso de Prueba", "Costo Penalizado (Factible)", fontsize=12)
    g.set_titles("Pistas: {col_name}", size=14)
    g.fig.suptitle('GRASP: Distribución de Costos por Variante y Nº de Restarts', y=1.03, fontsize=16)
    g.despine(left=True)
    # Ajustar la leyenda
    g.legend.set_title("Variante GRASP")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, "4_grasp_restarts_vs_cost_boxplot.png"), dpi=300)
    plt.close()
    print("Gráfico '4_grasp_restarts_vs_cost_boxplot.png' generado.")


def plot_grasp_restarts_vs_time(df, output_dir):
    if df is None or df.empty: return
    plt.style.use('seaborn-v0_8-whitegrid')
    grasp_algos_csv_names = [
        'GRASP_HC_Det_0Restarts', 
        'GRASP_HC_Estoc_10Restarts', 
        'GRASP_HC_Estoc_50Restarts', 
        'GRASP_HC_Estoc_100Restarts'
    ]
    df_grasp = df[df['Algoritmo'].isin(grasp_algos_csv_names)].copy()

    if df_grasp.empty:
        print("Advertencia para plot_grasp_restarts_vs_time: No hay datos GRASP para graficar tiempos.")
        return

    df_grasp_time_avg = df_grasp.groupby(['NombreCaso', 'NumPistas', 'Algoritmo'])['TiempoComputacional_seg'].mean().reset_index()
    df_grasp_time_avg['Caso_Pistas'] = df_grasp_time_avg['NombreCaso'] + " (" + df_grasp_time_avg['NumPistas'].astype(str) + "P)"
    df_grasp_time_avg['VarianteGRASP'] = df_grasp_time_avg['Algoritmo'].map(ALGO_DISPLAY_NAMES)
    df_grasp_time_avg['VarianteGRASP'] = pd.Categorical(df_grasp_time_avg['VarianteGRASP'], categories=GRASP_VARIANTS_ORDER_DISPLAY, ordered=True)

    def sort_key_caso_pistas_str(caso_pistas_str):
        nombre_caso, pistas_str = caso_pistas_str.split(" (")
        match = re.search(r'\d+', nombre_caso)
        num_caso_val = int(match.group(0)) if match else float('inf')
        num_pistas_val = int(pistas_str.replace("P)", ""))
        return (num_caso_val, num_pistas_val)
    sorted_caso_pistas = sorted(df_grasp_time_avg['Caso_Pistas'].unique(), key=sort_key_caso_pistas_str)

    plt.figure(figsize=(15, 8))
    sns.barplot(x='Caso_Pistas', y='TiempoComputacional_seg', hue='VarianteGRASP', 
                data=df_grasp_time_avg, order=sorted_caso_pistas, hue_order=GRASP_VARIANTS_ORDER_DISPLAY, palette=PALETA_GRASP)
    plt.title('Tiempo Computacional Promedio para Variantes de GRASP', fontsize=16)
    plt.ylabel('Tiempo Computacional Promedio (s)', fontsize=12)
    plt.xlabel('Caso de Prueba (Número de Pistas)', fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)
    plt.yscale('log')
    plt.legend(title='Variante GRASP', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=10, title_fontsize=12)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(os.path.join(output_dir, "5_grasp_restarts_vs_time.png"), dpi=300)
    plt.close()
    print("Gráfico '5_grasp_restarts_vs_time.png' generado.")


def main():
    if not os.path.exists(GRAFICOS_DIR):
        os.makedirs(GRAFICOS_DIR)
        print(f"Directorio '{GRAFICOS_DIR}' creado.")

    csv_filepath = os.path.join('.', 'results', 'resultado.csv')
    
    if not os.path.exists(csv_filepath):
        print(f"Error CRÍTICO: El archivo CSV '{csv_filepath}' no se encuentra.")
        print("Asegúrate de que el archivo existe en la ruta especificada y tiene datos.")
        return 

    df_resultados = load_and_preprocess_data(csv_filepath)

    if df_resultados is not None and not df_resultados.empty:
        plot_cost_comparison_main_algos(df_resultados.copy(), GRAFICOS_DIR)
        plot_time_comparison_main_algos(df_resultados.copy(), GRAFICOS_DIR)
        plot_ge_solo_boxplot_costos(df_resultados.copy(), GRAFICOS_DIR)
        plot_grasp_restarts_vs_cost_boxplot(df_resultados.copy(), GRAFICOS_DIR)
        plot_grasp_restarts_vs_time(df_resultados.copy(), GRAFICOS_DIR)
        print(f"\nTodos los gráficos solicitados han sido generados en la carpeta '{GRAFICOS_DIR}'.")
    else:
        print("No se pudieron generar gráficos debido a problemas con los datos o el archivo CSV.")

if __name__ == '__main__':
    main()
