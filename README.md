# Nibbler: Model-Free RL with Dynamically Constructed GVF Features

A JAX/Equinox implementation of the Nibbler architecture from the paper **"Towards model-free RL algorithms that scale well with unstructured data"** by Joseph Modayil and Zaheer Abbas.

📄 **Paper**: [arXiv:2311.02215](https://arxiv.org/abs/2311.02215)

## Overview

Nibbler is a reinforcement learning architecture designed to scale efficiently with high-dimensional, unstructured observations. It addresses the challenge that most function approximation methods rely on externally provisioned knowledge about input structure (e.g., convolutional networks for images, graph neural networks for graphs).

The key insight is that Nibbler dynamically constructs **General Value Function (GVF)** questions to discover and exploit predictive structure directly from the experience stream. It:

1. Learns a set of GVF networks in parallel, each predicting a base observation feature as a cumulant
2. Uses the hidden representations from these GVFs as additional features for the main action-value function
3. Dynamically selects which base features to use as cumulants and inputs based on their learned utility (measured by weight magnitude in auxiliary linear predictors)

This enables sample complexity that scales linearly with observation size, even when nonlinear features are required for control.

## Installation

### Requirements

- Python 3.9+
- CUDA-compatible GPU recommended (but CPU works)

### Setup

```bash
# Clone the repository
git clone https://github.com/ejmejm/nibbler
cd nibbler

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

For GPU support, ensure you have the appropriate JAX version installed:

```bash
# For CUDA 12
pip install --upgrade "jax[cuda12]"

# For CUDA 11
pip install --upgrade "jax[cuda11_pip]"
```

## Usage

### Training Nibbler

```bash
# Basic training with defaults (4 GVFs, 2 coupled environments)
python nibbler.py

# Training with more GVFs for larger problems
python nibbler.py --n_gvfs 8 --num_envs 4 --num_steps 5000000
```

### Training Q-Learning Baseline

```bash
# Basic DQN baseline
python q_learning_baseline.py

# With a deeper network
python q_learning_baseline.py --hidden_dims 256 256

# Linear baseline (no hidden layers)
python q_learning_baseline.py --hidden_dims

# Full example
python q_learning_baseline.py \
    --hidden_dims 256 \
    --num_envs 2 \
    --num_steps 1000000 \
    --learning_rate 0.001 \
    --epsilon 0.1 \
    --gamma 0.99 \
    --seed 42
```

### Experiment Tracking

Both scripts log metrics to MLflow. View results with:

```bash
mlflow ui
```

Then open `http://localhost:5000` in your browser.

## Nibbler-Specific Parameters

These parameters control the GVF-based feature construction unique to the Nibbler architecture:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n_gvfs` | 4 | Number of GVF networks to learn in parallel. Each GVF predicts a base observation feature. More GVFs provide richer features but increase computation. Scale with problem complexity. |
| `--inputs_per_gvf` | 82 | Number of base observation features used as input to each GVF network. Should be less than or equal to the observation dimension. Controls the receptive field of each GVF. |
| `--hidden_dim_per_gvf` | 256 | Hidden layer dimension for each GVF network. The main value function receives `n_gvfs × hidden_dim_per_gvf` additional features from the GVF representations. |
| `--tau_inputs` | 0.0 | Utility threshold for swapping GVF input features. When > 0, enables incremental top-k selection: a currently-selected input feature is replaced by an unselected one only if the unselected feature's utility exceeds the selected one's by this margin. Set to 0 to disable dynamic input selection. |
| `--tau_cumulants` | 0.0 | Utility threshold for swapping GVF cumulant features. Similar to `--tau_inputs` but for selecting which base features to predict. Utility is measured by weight magnitude in the linear reward predictor. Set to 0 to disable dynamic cumulant selection. |
| `--reset_output_weights` | False | Whether to reset the main value function's output weights (for features corresponding to a GVF) when that GVF's cumulant is changed. When enabled, provides a cleaner slate but may slow convergence. |

### Scaling Guidelines

The paper suggests scaling resources with problem size:
- **Number of GVFs**: Scale approximately with the number of coupled environments
- **Learning rate**: The step size is automatically scaled by `1/sqrt(n_gvfs)` via `--step_size_scaling_factor`
- **Inputs per GVF**: Should cover the relevant base features; setting equal to observation dim ensures all features are accessible

## Project Structure

```
nibbler/
├── nibbler.py              # Main Nibbler implementation
├── q_learning_baseline.py  # DQN baseline for comparison
├── networks.py             # Neural network modules (MLP, QVNetwork)
├── catch_env.py            # Coupled Catch environment
├── utils.py                # Utility functions
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Catch Environment

The Coupled Catch environment is a combinatorial RL testbed with an exponentially large state space where multiple independent "catch" games run in parallel. A ball falls down a grid, and the agent controls a paddle to catch it—rewards are delayed and coupled across boards.

To visualize the environment with random actions:

```bash
python catch_env.py
```

This generates `catch_animation.gif` showing the ball, paddle, and special status bits (hot, reset, catch, miss, plus, minus) over 100 steps.
