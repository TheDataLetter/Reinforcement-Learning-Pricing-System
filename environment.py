# Custom Gym environment for CPG pricing simulation
# This implements the exact business logic described in the article

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PricingStrategySim(gym.Env):
    # Simulates product pricing with inventory constraints

    def __init__(self, products_df=None):
        super().__init__()

        # Use provided products or create default ones
        if products_df is not None:
            self.products = products_df
        else:
            # Default product data if none provided
            self.products = pd.DataFrame({
                'base_demand': [100],
                'price_sensitivity': [-1.8],
                'price': [5.99],
                'cost': [2.50],
                'initial_inventory': [1000]
            })

        self._setup_action_space()
        self.reset()
    
    def _setup_action_space(self):
        # Define what actions the AI can take
        # Action space: [discount_multiplier, display_priority]
        # As described in the article: discount from 0.7 (30% off) to 1.3 (30% premium)
        # Display priority from 0 (no display) to 1 (prime placement)
        self.action_space = spaces.Box(
            low=np.array([0.7, 0], dtype=np.float32),
            high=np.array([1.3, 1], dtype=np.float32), 
            dtype=np.float32
        )

        # Observation space: [inventory, competitor_price, demand_factor]
        # Matches the articles state observation description
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0.8, 0], dtype=np.float32),        # Min values [inventory, comp_price, demand_factor, cost]
            high=np.array([10000, 20, 1.2, 20], dtype=np.float32),  # Max values
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        # Reset the environment to initial state
        super().reset(seed=seed)

        # Initialize environment state (focus on first product)
        self.sku_idx = 0
        self.inventory = self.products.iloc[self.sku_idx]['initial_inventory']

        # Competitor starts with a 10% premium to our regular price
        self.comp_price = self.products.iloc[self.sku_idx]['price'] * 1.1

        # Demand starts at normal level
        self.demand_factor = 1.0

        return self._get_obs(), {}
    
    def _get_obs(self):
        # Get current observation (state) of the environment
        return np.array([
            self.inventory,
            self.comp_price, 
            self.demand_factor,
            self.products.iloc[self.sku_idx]['cost']
        ],  dtype=np.float32)
    
    def step(self, action):
        # Execute one time step in the environment
        # This implements the exact business logic from the article

        # Clip actions to valid ranges (safety first!)
        raw_discount = np.clip(action[0], 0.7, 1.3) # Price multiplier
        display = np.clip(action[1], 0, 1)          # Display priority
                                 
        # Get current product details
        sku = self.products.iloc[self.sku_idx]

        # 1. Calculate price sensitivity effect (as described in article)
        # Formula: discount ** price_sensitivity
        price_response_factor = raw_discount ** sku['price_sensitivity']
        demand = sku['base_demand'] * price_response_factor

        # 2. Apply display impact (20% maximum lift as per article)
        demand *= (1 + display * 0.2)

        # 3. Apply demand randomness (seasonality ±20%)
        demand *= self.demand_factor

        # 4. Calculate actual sales (can't exceed inventory)
        sales = min(demand, self.inventory)

        # 5. Reward calculation (the heart of the RL system)
        gross_margin = sales * (sku['price'] * raw_discount - sku['cost'])
        stockout_penalty = max(0, demand - sales) * 0.5     # Stockout penalty
        reward = gross_margin - stockout_penalty

        # 6. Update inventory
        self.inventory -= sales

        # 7. Simulate competitor reaction (±5% price change)
        self.comp_price *= np.random.uniform(0.95, 1.05)

        # 8. Random demand fluctuation (as described in article)
        self.demand_factor = np.random.uniform(0.8, 1.2)

        # Check if episode is done (out of stock)
        done = self.inventory <= 0
        truncated = False

        return self._get_obs(), reward, done, truncated, {}