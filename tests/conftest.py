import pytest
import numpy as np
from environment import PricingStrategySim

@pytest.fixture
def sample_environment():
    # Fixture to create a sample environment for testing
    env = PricingStrategySim()
    obs, _ = env.reset()
    return env, obs

@pytest.fixture
def sample_products():
    # Fixture to provide sample product data
    return {
        'base_demand': [100, 150, 200],
        'price_sensitivity': [-1.8, -2.2, -1.5],
        'price': [5.99, 3.99, 7.99],
        'cost': [2.50, 1.75, 3.20],
        'initial_inventory': [1000, 1500, 2000]
    }