import pytest
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from environment import PricingStrategySim
from agent import MarginAwareAgent


@pytest.fixture
def sample_products():
    import pandas as pd

    return pd.DataFrame(
        {
            "base_demand": [100],
            "price_sensitivity": [-1.8],
            "price": [5.99],
            "cost": [2.50],
            "initial_inventory": [1000],
        }
    )


def test_predict_with_constraints(sample_products):
    # Create a proper environment
    env = DummyVecEnv([lambda: PricingStrategySim(sample_products)])

    # Create agent with proper environment
    agent = MarginAwareAgent(
        policy="MlpPolicy", env=env, products=sample_products, verbose=0
    )

    # Test with low inventory
    obs = np.array([50.0, 6.59, 1.0, 2.5])  # inventory < 100
    _ = np.array([0.8, 0.7])

    # Agent should block promotions when inventory is low
    modified_action, _ = agent.predict(obs, deterministic=True)
    assert modified_action[1] == 0.0


if __name__ == "__main__":
    pytest.main()
