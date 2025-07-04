from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from bad_ice_cream_env import BadIceCreamEnv
import keyboard

# === Callback para detener entrenamiento con tecla 'q' ===
class KeyboardStopCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        if keyboard.is_pressed('q'):
            print("🛑 Entrenamiento detenido con tecla 'q'")
            return False
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
            learning_rate=2.5e-4,            # 🔼 Ligeramente más alto para aprender más rápido
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,                  # 🔼 Más grande para actualizar con más datos por paso
            tau=0.01,                       # 🔼 Más alto que 0.005 para actualizaciones más rápidas, pero aún estables
            gamma=0.95,                     # 🔽 Menor prioridad al largo plazo, favorece recompensas inmediatas (como frutas)
            train_freq=2,                  # 🔼 Aprende cada 2 pasos, más frecuente
            target_update_interval=250,     # 🔼 Actualiza la red objetivo más seguido
            exploration_fraction=0.2,       # 🔼 Explora más al inicio
            exploration_final_eps=0.05,     # 🔽 Mantiene algo de exploración al final
            verbose=1,
            tensorboard_log="./logs/"
        )

    callback = KeyboardStopCallback()

    print("🚀 Entrenando agente... Presiona 'q' para detener.")
    model.learn(total_timesteps=2_000_000, reset_num_timesteps=False, callback=callback)

    model.save("agente_bad_ice_cream")
    print("✅ Modelo guardado como 'agente_bad_ice_cream'.")

if __name__ == "__main__":
    main()
