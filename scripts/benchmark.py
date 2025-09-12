#!/usr/bin/env python3
# Benchmarking script for the reinforcement learning pricing system.
# This script measure performance metrics and compares different model versions.

import sys
import os
# Add the parent directory to the path so we can import validate
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse 
import time
import numpy as np
import pandas as pd
from validate import assess_policy

def run_benchmark(model_path, products, n_episodes=100):
    # Run comprehensive benchmarking on a trained model
    print(f"Starting benchmark for {model_path}")
    start_time = time.time()

    results = assess_policy(model_path, products, n_episodes)

    end_time = time.time()
    duration = end_time - start_time

    print(f"Benchmark completed in {duration:.2f} seconds")
    print(f"Average reward: {np.mean(results):.2f} ± {np.std(results):.2f}")
    print(f"Min reward: {np.min(results):.2f}")
    print(f"Max reward: {np.max(results):.2f}")

    return {
        'mean_reward': np.mean(results),
        'std_reward': np.std(results),
        'min_reward': np.min(results),
        'max_reward': np.max(results),
        'duration':duration
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark RL pricing model')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to run')
    
    args = parser.parse_args()

    # Sample products data
    products = pd.DataFrame({
        'base_demand': [100, 150, 200],
        'price_sensitivity': [-1.8, -2.2, -1.5],
        'price': [5.99, 3.99, 7.99],
        'cost': [2.50, 1.75, 3.20],
        'initial_inventory': [1000, 1500, 2000]
    })

    run_benchmark(args.model_path, products, args.episodes)