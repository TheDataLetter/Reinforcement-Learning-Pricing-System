import numpy as np
import pandas as pd
from environment import PricingStrategySim
from stable_baselines3 import PPO

def assess_policy(model_path, products, n_episodes=10): # Reduced episodes for faster testing
    # Load the trained model 
    from agent import MarginAwareAgent
    model = MarginAwareAgent.load(model_path)
    model.products = products

    env = PricingStrategySim(products)
    results = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        total_profit = 0 
        done = False
        step_count = 0

        while not done:
            action, _ = model.predict(obs)
            print(f"Episode {episode+1}, Step {step_count}")
            print(f"Inventory: {obs[0]}")
            print(f"Action: {action}")
            # Instead of strict equality checks, use a tolerance for floating point comparisons
            assert 0.7 - 1e-5 <= action[0] <= 1.3 + 1e-5, f"Invalid discount: {action[0]}"

            # Verify 15% margin floor with tolerance
            cost_price = obs[3]
            shelf_price = products.iloc[0]['price']
            assert (action[0] * shelf_price) >= (1.15 * cost_price) - 1e-5, "Margin breach"

            # No promos if stock < 100
            if obs[0] < 100:
                print(f"WARNING: Inventory below 100 but promotion is {action[1]}")

                assert action[1] < 1e-5, f"Promo active during stockout risk: {action[1]}"
            
            obs, reward, done, _, _ = env.step(action)
            step_count += 1
            total_profit += reward
        
        print(f"Episode {episode+1} completed with profit: {total_profit}")
        
        results.append(total_profit)

    print(f"Average Reward: {np.mean(results):.2f} ± {np.std(results):.2f}")
    return results

if __name__ == "__main__":
    products = pd.DataFrame({
        'base_demand': [100, 150, 200],
        'price_sensitivity': [-1.8, -2.2, -1.5],
        'price': [5.99, 3.99, 7.99],
        'cost': [2.50, 1.75, 3.20],
        'initial_inventory': [1000, 1500, 2000] # Add this line
    })

    # You would need to train a model first or provide a path to a pre-trained model
    print("Note: Run train.py first to create a model, then update this path")
    results = assess_policy("models/trained_pricing_agent", products, n_episodes=10)