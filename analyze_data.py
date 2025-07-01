import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

def cargar_y_analizar_datos(csv_path="training_data.csv"):
    """Load and analyze training data"""
    try:
        df = pd.read_csv(csv_path)
        print(f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas")
        
        print(f"\nEstadísticas básicas:")
        print(f"   Total episodios: {df['episodio_id'].nunique()}")
        print(f"   Total pasos: {len(df)}")
        print(f"   Reward promedio: {df['recompensa'].mean():.3f}")
        print(f"   Reward máximo: {df['recompensa'].max():.3f}")
        print(f"   Reward mínimo: {df['recompensa'].min():.3f}")
        
        #analisis episodio
        episodios = df.groupby('episodio_id').agg({
            'recompensa': 'sum',
            'frutas_recogidas': 'max',
            'pasos_totales': 'max',
            'terminado': 'max'
        }).reset_index()
        
        print(f"\nAnálisis por episodio:")
        print(f"   Reward promedio por episodio: {episodios['recompensa'].mean():.3f}")
        print(f"   Frutas promedio por episodio: {episodios['frutas_recogidas'].mean():.2f}")
        print(f"   Pasos promedio por episodio: {episodios['pasos_totales'].mean():.1f}")
        
        print(f"\nDistribución de acciones:")
        accion_counts = df['accion'].value_counts()
        for accion, count in accion_counts.items():
            porcentaje = (count / len(df)) * 100
            print(f"   Acción {accion}: {count} veces ({porcentaje:.1f}%)")
        
        print(f"\nAnálisis de terminación:")
        causa_counts = df['causa_terminacion'].value_counts()
        for causa, count in causa_counts.items():
            if causa:  # Skip empty strings
                print(f"   {causa}: {count} episodios")
        
        print(f"\n Progreso de aprendizaje:")
        # calcular reward
        episodios_ordenados = episodios.sort_values('episodio_id')
        episodios_ordenados['reward_acumulado'] = episodios_ordenados['recompensa'].cumsum()
        
        primeros_10 = episodios_ordenados.head(10)['recompensa'].mean()
        ultimos_10 = episodios_ordenados.tail(10)['recompensa'].mean()
        print(f"   Reward promedio primeros 10 episodios: {primeros_10:.3f}")
        print(f"   Reward promedio últimos 10 episodios: {ultimos_10:.3f}")
        print(f"   Mejora: {((ultimos_10 - primeros_10) / abs(primeros_10) * 100):.1f}%" if primeros_10 != 0 else "   Mejora: N/A")
        
        return df, episodios
        
    except FileNotFoundError:
        print(f"No se encontró el archivo {csv_path}")
        return None, None
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        return None, None

def visualizar_datos(df, episodios):
    """Create visualizations of the training data"""
    if df is None or episodios is None:
        return
        
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Análisis de Entrenamiento - Bad Ice Cream Agent', fontsize=16)
        
        # 1. Reward per episode
        axes[0, 0].plot(episodios['episodio_id'], episodios['recompensa'])
        axes[0, 0].set_title('Reward por Episodio')
        axes[0, 0].set_xlabel('Episodio')
        axes[0, 0].set_ylabel('Reward Total')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Fruits collected per episode
        axes[0, 1].plot(episodios['episodio_id'], episodios['frutas_recogidas'])
        axes[0, 1].set_title('Frutas Recogidas por Episodio')
        axes[0, 1].set_xlabel('Episodio')
        axes[0, 1].set_ylabel('Frutas Recogidas')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Action distribution
        accion_counts = df['accion'].value_counts()
        axes[1, 0].bar(accion_counts.index, accion_counts.values)
        axes[1, 0].set_title('Distribución de Acciones')
        axes[1, 0].set_xlabel('Acción (0=Up, 1=Down, 2=Left, 3=Right)')
        axes[1, 0].set_ylabel('Frecuencia')
        
        # 4. Steps per episode
        axes[1, 1].plot(episodios['episodio_id'], episodios['pasos_totales'])
        axes[1, 1].set_title('Pasos por Episodio')
        axes[1, 1].set_xlabel('Episodio')
        axes[1, 1].set_ylabel('Pasos Totales')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Gráficos guardados como 'training_analysis.png'")
        
    except Exception as e:
        print(f"Error al crear visualizaciones: {e}")

if __name__ == "__main__":
    print("🔍 Analizando datos de entrenamiento...")
    df, episodios = cargar_y_analizar_datos()
    
    if df is not None:
        visualizar_datos(df, episodios)
        print("\nAnálisis completado!")
    else:
        print("No se pudieron analizar los datos.") 