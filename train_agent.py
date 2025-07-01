from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from bad_ice_cream_env import BadIceCreamEnv
from pynput import keyboard as pynput_keyboard

# === Callback para detener el entrenamiento con tecla 'q' ===
class KeyboardStopCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.stop_flag = False
        self.listener = pynput_keyboard.Listener(on_press=self.on_press)
        self.listener.start()
    
    def on_press(self, key):
        try:
            if key.char == 'q':
                print("Entrenamiento detenido manualmente con la tecla 'q'")
                self.stop_flag = True
        except AttributeError:
            pass

    def _on_step(self) -> bool:
        return not self.stop_flag

# === Entrenamiento principal ===
def main():
    env = BadIceCreamEnv(collect_data=True, csv_path="training_data.csv")

    try:
        model = DQN.load("agente_bad_ice_cream", env=env)
        print("Modelo cargado desde archivo.")
    except FileNotFoundError:
        print("Iniciando nuevo entrenamiento.")
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1000,
            verbose=1
        )

    callback = KeyboardStopCallback()

    print("Entrenando agente... Presiona 'q' para detener.")
    model.learn(total_timesteps=500_000, reset_num_timesteps=False, callback=callback)
    
    # Guardar modelo al final, incluso si se interrumpe con 'q'
    model.save("agente_bad_ice_cream")
    print("Modelo guardado como 'agente_bad_ice_cream'.")
    
    # Guardar datos de entrenamiento
    env.guardar_datos()
    
    # Mostrar estadísticas
    stats = env.obtener_estadisticas()
    if stats:
        print("\n Estadísticas de entrenamiento:")
        print(f"   Total de pasos: {stats['total_pasos']}")
        print(f"   Total de episodios: {stats['total_episodios']}")
        print(f"   Reward promedio: {stats['reward_promedio']:.3f}")
        print(f"   Reward máximo: {stats['reward_maximo']:.3f}")
        print(f"   Frutas promedio por episodio: {stats['frutas_promedio_por_episodio']:.2f}")
        print(f"   Pasos promedio por episodio: {stats['pasos_promedio_por_episodio']:.1f}")
        print(f"   Episodios terminados por muerte: {stats['episodios_terminados_por_muerte']}")
        print(f"   Episodios terminados por pasos: {stats['episodios_terminados_por_pasos']}")

if __name__ == "__main__":
    main()
