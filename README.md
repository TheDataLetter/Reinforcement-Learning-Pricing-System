# Reinforcement Learning for CPG Pricing Optimization

This repository contains a production-ready reinforcement learning system for optimizing promotional strategies in the CPG (Consumer Packaged Goods) industry. The system uses Proximal Policy Optimization (PPO) with custom business constraints to maximize profitability while maintaining inventory health.

## Features

- Custom Gym environment for CPG pricing simulation
- Margin-aware PPO agent with business rule enforcement
- Real-time inventory management and stockout prevention
- Dynamic pricing with competitor response simulation
- Comprehensive validation and testing suite

## Quick Start

1. Install dependencies:

```bash
pip install -r requirements.txt

2. Run a demo:

python demo.py

3. Train the model:

python train.py

4. Validate the trained model:

python validate.py

# Project Structure

text
reinforcement-learning-pricing/
├── config/                 # Configuration files
├── models/                 # Trained model artifacts
├── scripts/                # Deployment and benchmarking scripts
├── tests/                  # Unit and integration tests
├── agent.py               # Custom MarginAwareAgent implementation
├── environment.py         # Pricing simulation environment
├── train.py              # Model training script
├── validate.py           # Model validation script
└── demo.py              # Demonstration script

# Configuration

	•	config/products.json: Product catalog with base demand, costs, and price sensitivity
	•	config/hyperparameters.yaml: Training parameters and model configuration

# Results

After training, the system typically achieves:

	•	Average profit: $1,600-$1,700 per episode
	•	100% compliance with business constraints
	•	Effective inventory management with zero stockouts

# Contributing

Please read our contributing guidelines before submitting pull requests.

# License

This project is licensed under the MIT License - see LICENSE file for details.

