import pandas as pd
import numpy as np
import time
from datetime import datetime
import os

class DataCollector:
    def __init__(self, csv_path="training_data.csv"):
        self.csv_path = csv_path
        self.data = {
            'timestamp': [],
            'episodio_id': [],
            'paso_en_episodio': [],
            'estado_matriz': [],  # matrix 12x15 as string
            'accion': [],
            'recompensa': [],
            'posicion_jugador_fila': [],
            'posicion_jugador_col': [],
            'frutas_disponibles': [],  
            'enemigos_posiciones': [],  
            'terminado': [],
            'causa_terminacion': [],
            'frutas_recogidas': [],
            'pasos_totales': [],
            'reward_acumulado': []
        }
        self.episodio_actual = 0
        self.paso_actual = 0
        self.reward_acumulado = 0.0
        self.frutas_recogidas_episodio = 0
        
    def reset_episodio(self):
        """Reset episode counters"""
        self.episodio_actual += 1
        self.paso_actual = 0
        self.reward_acumulado = 0.0
        self.frutas_recogidas_episodio = 0
        
    def registrar_paso(self, matriz, accion, recompensa, pos_jugador, terminado, 
                      causa_terminacion="", frutas_recogidas=0):
        """Record a single step in the training"""
        # Extraer frutas
        frutas = np.argwhere(matriz == 3).tolist()  # valor 3 = frutas
        enemigos = np.argwhere(matriz == 4).tolist()  # valor 4 = enemigos
        
        # csv 
        frutas_str = str(frutas) if len(frutas) > 0 else "[]"
        enemigos_str = str(enemigos) if len(enemigos) > 0 else "[]"
        
        matriz_str = str(matriz.flatten().tolist())
        
        # Update data
        self.reward_acumulado += recompensa
        self.frutas_recogidas_episodio += frutas_recogidas
        self.paso_actual += 1
        
        # Record data
        self.data['timestamp'].append(datetime.now().isoformat())
        self.data['episodio_id'].append(self.episodio_actual)
        self.data['paso_en_episodio'].append(self.paso_actual)
        self.data['estado_matriz'].append(matriz_str)
        self.data['accion'].append(accion)
        self.data['recompensa'].append(recompensa)
        self.data['posicion_jugador_fila'].append(pos_jugador[0] if pos_jugador != (-1, -1) else -1)
        self.data['posicion_jugador_col'].append(pos_jugador[1] if pos_jugador != (-1, -1) else -1)
        self.data['frutas_disponibles'].append(frutas_str)
        self.data['enemigos_posiciones'].append(enemigos_str)
        self.data['terminado'].append(terminado)
        self.data['causa_terminacion'].append(causa_terminacion)
        self.data['frutas_recogidas'].append(self.frutas_recogidas_episodio)
        self.data['pasos_totales'].append(self.paso_actual)
        self.data['reward_acumulado'].append(self.reward_acumulado)
        
    def guardar_csv(self):
        """Save collected data to CSV"""
        df = pd.DataFrame(self.data)
        df.to_csv(self.csv_path, index=False)
        print(f"Datos guardados en {self.csv_path}")
        print(f"   Total de pasos registrados: {len(df)}")
        print(f"   Total de episodios: {df['episodio_id'].nunique()}")
        
    def obtener_estadisticas(self):
        """Get training statistics"""
        if len(self.data['episodio_id']) == 0:
            return {}
            
        df = pd.DataFrame(self.data)
        stats = {
            'total_pasos': len(df),
            'total_episodios': df['episodio_id'].nunique(),
            'reward_promedio': df['recompensa'].mean(),
            'reward_maximo': df['recompensa'].max(),
            'frutas_promedio_por_episodio': df.groupby('episodio_id')['frutas_recogidas'].max().mean(),
            'pasos_promedio_por_episodio': df.groupby('episodio_id')['pasos_totales'].max().mean(),
            'episodios_terminados_por_muerte': len(df[df['causa_terminacion'] == 'muerte']),
            'episodios_terminados_por_pasos': len(df[df['causa_terminacion'] == 'max_steps'])
        }
        return stats 