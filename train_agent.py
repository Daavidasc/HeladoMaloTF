from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from bad_ice_cream_env import BadIceCreamEnv
import keyboard

# === Callback para detener el entrenamiento con tecla 'q' ===
class KeyboardStopCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        if keyboard.is_pressed('q'):
            print("🛑 Entrenamiento detenido manualmente con la tecla 'q'")
            return False  # ← Detiene el entrenamiento
        return True

# === Entrenamiento principal ===
def main():
    env = BadIceCreamEnv()

    try:
        model = DQN.load("agente_bad_ice_cream", env=env)
        print("📦 Modelo cargado desde archivo.")
    except FileNotFoundError:
        print("🆕 Iniciando nuevo entrenamiento.")
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

    print("🚀 Entrenando agente... Presiona 'q' para detener.")
    model.learn(total_timesteps=500_000, reset_num_timesteps=False, callback=callback)
    
    # Guardar modelo al final, incluso si se interrumpe con 'q'
    model.save("agente_bad_ice_cream")
    print("✅ Modelo guardado como 'agente_bad_ice_cream'.")

if __name__ == "__main__":
    main()
