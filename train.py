import os
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from environment import PricingStrategySim
from agent import MarginAwareAgent

os.makedirs("./models", exist_ok=True)

# Create synthetic product data
products = pd.DataFrame(
    {
        "base_demand": [100, 150, 200],
        "price_sensitivity": [-1.8, -2.2, -1.5],
        "price": [5.99, 3.99, 7.99],
        "cost": [2.50, 1.75, 3.20],
        "initial_inventory": [1000, 1500, 2000],
    }
)

# Build environment
env = DummyVecEnv([lambda: PricingStrategySim(products)])

# Configure the RL model
model = MarginAwareAgent(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    policy_kwargs={"net_arch": [64, 64]},
    verbose=1,
    products=products,
)

# train with evaluation
model.learn(
    total_timesteps=10000,  # Reduced for faster training
    callback=EvalCallback(
        env,
        best_model_save_path="./models",
        eval_freq=1000,
        deterministic=True,
    ),
)

# Save the final model
model.save("models/trained_pricing_agent")
print("Training completed and model saved!")
