from stable_baselines3 import DQN
from bad_ice_cream_env import BadIceCreamEnv
import time

def main():
    env = BadIceCreamEnv()
    model = DQN.load("agente_bad_ice_cream", env=env)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        print(f"🎮 Acción: {action} | Recompensa: {reward} | Terminado: {done}")
        time.sleep(0.2)
        if done:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
