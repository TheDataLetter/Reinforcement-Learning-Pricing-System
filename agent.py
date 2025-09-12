from stable_baselines3 import PPO
import numpy as np

class MarginAwareAgent(PPO):
    def __init__(self, *args, products=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.products = products  # Store products for later use

    def predict(self, obs, state=None, episode_start=None, deterministic=True):
        action, state = super().predict(obs, state, episode_start, deterministic)

        print(f"Agent received - obs: {obs}, action: {action}")

        # Handle both single and batch observations
        if len(obs.shape) == 1:
            # Single observation
            inventory = obs[0]
            unit_cost = obs[3]
            shelf_price = self.products.iloc[0]['price']
            min_sale_price = 1.15 * unit_cost
            action[0] = max(action[0], min_sale_price / shelf_price)
            action[0] = np.clip(action[0], 0.7, 1.3)

            print (f"Single observation - Inventory: {inventory}, Action before: {action}")

            action[0] = max(action[0], min_sale_price / shelf_price)
            action[0] = np.clip(action[0], 0.7, 1.3)

            if inventory < 100:
                print("Blocking promotion! Inventory: {inventory}")
                action[1] = 0.0
        else:
            # Batch observation
            inventory = obs[0, 0]
            unit_cost = obs[0, 3]
            shelf_price = self.products.iloc[0]['price']
            min_sale_price = 1.15 * unit_cost

            print(f"Batch obs - Inventory: {inventory}, Current action: {action}")

            action[0, 0] = max(action[0, 0], min_sale_price / shelf_price)
            action[0, 0] = np.clip(action[0, 0], 0.7, 1.3) 

            
            if inventory < 100:
                print(f"Blocking promotion! Inventory: {inventory}")
                action[0, 1] = 0.0
                

        print(f"Agent returning action: {action}")
        return action, state