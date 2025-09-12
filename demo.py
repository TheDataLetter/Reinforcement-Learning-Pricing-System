# Demo scirpt for the Reinforcement Learning Pricing System
# This shows how the environment works without needing to train a model

import pandas as pd
import numpy as np
from environment import PricingStrategySim

def main():
    print("Reinforcement Learning Pricing Demo")
    print("=======================================")
    print("This demo shows how the pricing environemnt works with a single product")
    print()

    # Create sample product data (coffee product from teh article)
    products = pd.DataFrame({
        'base_demand': [100],       # Daily units sold at regular price
        'price_sensitivity': [-1.8],    # How demand changes with price
        'price': [5.99],            # Regular shelf price
        'cost': [2.50],             # Wholesale cost
        'initial_inventory': [1000]  # Starting inventory
    })

    # Create the pricing environment
    print("Creating pricing environment with coffee product...")
    env = PricingStrategySim(products)

    # Show intial state
    obs, _ = env.reset()
    print(f"📦 Initial inventory: {obs[0]:.0f} units")
    print(f"💰 Competitor price: ${obs[1]:.2f}")
    print(f"📈 Demand factor: {obs[2]:.2f}x normal")

    # Test a sample action (20% discount with medium display priority)
    action = np.array([0.8, 0.5])   # 20% off, medium display priority

    print(f"🎯 Testing action: {action[0]:.1f} price multiplier ({ (1-action[0])*100:.0f}% discount)")
    print(f"                : {action[1]:.1f} display priority")

    # Take the action
    obs, reward, done, _, info = env.step(action)

    # Show results
    print()
    print("📊 Results:")
    print(f"✅ Reward: ${reward:.2f}")
    print(f"📦 Remaining inventory: {obs[0]:.0f} units")
    print(f"💰 New competitor price: ${obs[1]:.2f}")
    print(f"📈 New demand factor: {obs[2]:.2f}x normal")
    print()
    print("Demo completed! Run train.py to train the full model.")

if __name__ == "__main__":
    main()