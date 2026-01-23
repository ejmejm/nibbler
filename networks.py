from typing import Optional, List

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jax import random


class DeepQNetwork(eqx.Module):
    """
    Deep Q-learning agent with configurable hidden layers.
    If hidden_dims is empty, the network is linear: obs_dim -> num_actions
    """
    layers: List[nn.Linear]
    
    def __init__(
        self,
        num_actions: int,
        obs_dim: int,
        hidden_dims: Optional[List[int]] = None,
        key: random.PRNGKey = None,
    ):
        """
        Initialize the deep Q-learning agent.
        
        Args:
            num_actions: Number of actions
            obs_dim: Observation dimension
            hidden_dims: List of hidden layer dimensions. If None or empty, network is linear.
            key: Random key for initialization
        """
        if hidden_dims is None:
            hidden_dims = []
        
        self.layers = []
        
        # Build layer dimensions
        if len(hidden_dims) == 0:
            # Linear network: obs_dim -> num_actions
            dims = [obs_dim, num_actions]
        else:
            # Deep network: obs_dim -> hidden_dims -> num_actions
            dims = [obs_dim] + hidden_dims + [num_actions]
        
        # Split keys for each layer
        if len(dims) == 2:
            keys = [key]
        else:
            keys = random.split(key, len(dims) - 1)
        
        # Create layers
        for i in range(len(dims) - 1):
            layer = nn.Linear(
                dims[i],
                dims[i + 1],
                use_bias = False,
                key = keys[i],
            )
            self.layers.append(layer)
    
    def q_values(self, observation: jax.Array) -> jax.Array:
        """
        Compute Q-values for all actions given an observation.
        
        Args:
            observation: Observation vector of shape (obs_dim,)
            
        Returns:
            Q-values for all actions, shape (num_actions,)
        """
        x = observation
        # Forward through all layers except the last
        for layer in self.layers[:-1]:
            x = layer(x)
            x = jax.nn.relu(x)
        # Last layer (no activation)
        x = self.layers[-1](x)
        return x
    
    def select_action(
        self,
        observation: jax.Array,
        epsilon: float,
        key: random.PRNGKey,
    ) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            observation: Current observation
            epsilon: Exploration probability
            key: Random key
            
        Returns:
            Selected action index
        """
        q_vals = self.q_values(observation)
        num_actions = q_vals.shape[0]
        
        # Epsilon-greedy: random with prob epsilon, greedy otherwise
        key1, key2 = random.split(key)
        explore = random.uniform(key1) < epsilon
        
        # Random action
        random_action = random.randint(key2, (), 0, num_actions)
        
        # Greedy action
        greedy_action = jnp.argmax(q_vals)
        
        return jnp.where(explore, random_action, greedy_action)