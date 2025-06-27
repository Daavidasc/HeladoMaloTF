from stable_baselines3 import DQN
from bad_ice_cream_env import BadIceCreamEnv

def main():
    env = BadIceCreamEnv()
    
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

    model.learn(total_timesteps=200_000)
    model.save("agente_bad_ice_cream")
    print("✅ Agente entrenado y guardado como 'agente_bad_ice_cream'")

if __name__ == "__main__":
    main()
