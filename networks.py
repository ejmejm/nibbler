from typing import Any, List, Optional, Tuple

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float


class MLP(eqx.Module):
    """Multi-layer perceptron with configurable hidden layers and activation function."""
    layers: List[nn.Linear]
    activation: Any = eqx.field(static=True)
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: Any = jax.nn.relu,
        use_bias: bool = False,
        key: random.PRNGKey = None,
    ):
        if hidden_dims is None:
            hidden_dims = []
        
        self.activation = activation
        self.layers = []
        
        # Build layer dimensions
        if len(hidden_dims) == 0:
            # Linear network: input_dim -> output_dim
            dims = [input_dim, output_dim]
        else:
            # Deep network: input_dim -> hidden_dims -> output_dim
            dims = [input_dim] + hidden_dims + [output_dim]
        
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
                use_bias = use_bias,
                key = keys[i],
            )
            self.layers.append(layer)
    
    def __call__(self, x: Float[Array, 'input_dim']) -> Float[Array, 'output_dim']:
        # Forward through all layers except the last
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x


class DeepQNetwork(eqx.Module):
    """Deep Q-learning agent with configurable hidden layers."""
    mlp: MLP
    
    def __init__(
        self,
        num_actions: int,
        obs_dim: int,
        hidden_dims: Optional[List[int]] = None,
        key: random.PRNGKey = None,
    ):
        """Initialize the deep Q-learning agent."""
        self.mlp = MLP(
            input_dim = obs_dim,
            output_dim = num_actions,
            hidden_dims = hidden_dims,
            activation = jax.nn.relu,
            use_bias = False,
            key = key,
        )
    
    def q_values(self, observation: Float[Array, 'obs_dim']) -> Float[Array, 'num_actions']:
        """Compute Q-values for all actions given an observation."""
        return self.mlp(observation)
    
    def select_action(
        self,
        observation: Float[Array, 'obs_dim'],
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
    
    def compute_grads_and_loss(
        self,
        obs: Float[Array, 'obs_dim'],
        action: int,
        reward: float,
        next_obs: Float[Array, 'obs_dim'],
        gamma: float,
    ) -> Tuple[Any, float]:
        """Compute gradients for Q-learning update.
        
        Args:
            agent: Current agent
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            gamma: Discount factor
            
        Returns:
            Tuple of (gradients, TD error squared as loss)
        """
        def loss_fn(network: DeepQNetwork) -> jax.Array:
            # Current Q-value for the action taken
            q_vals = network.q_values(obs)
            q_current = q_vals[action]
            
            # Target Q-value (using stop_gradient to prevent gradients from flowing through target)
            next_q_vals = jax.lax.stop_gradient(network.q_values(next_obs))
            q_target = reward + gamma * jnp.max(next_q_vals)
            
            # TD error
            td_error = q_target - q_current
            
            # Loss is squared TD error
            return td_error ** 2
        
        # Compute gradients using automatic differentiation
        loss, gradients = eqx.filter_value_and_grad(loss_fn)(self)
        
        return gradients, loss


class QVNetwork(eqx.Module):
    """QV network with shared and separate hidden layers."""
    shared_mlp: Optional[MLP]
    q_separate_mlp: MLP
    v_separate_mlp: MLP
    
    def __init__(
        self,
        n_actions: int,
        input_dim: int,
        shared_hidden_dims: Optional[List[int]] = None,
        separate_hidden_dims: Optional[List[int]] = None,
        activation: Any = jax.nn.relu,
        use_bias: bool = False,
        key: random.PRNGKey = None,
    ):
        if shared_hidden_dims is None:
            shared_hidden_dims = []
        if separate_hidden_dims is None:
            separate_hidden_dims = []
        
        # Determine shared output dimension
        if len(shared_hidden_dims) == 0:
            shared_output_dim = input_dim
            # No shared layers needed - will use identity in _compute_shared_representation
            self.shared_mlp = None
        else:
            shared_output_dim = shared_hidden_dims[-1]
            # Split keys for the three MLPs
            key, shared_key, q_key, v_key = random.split(key, 4)
            # Create shared MLP
            self.shared_mlp = MLP(
                input_dim = input_dim,
                output_dim = shared_output_dim,
                hidden_dims = shared_hidden_dims,
                activation = activation,
                use_bias = use_bias,
                key = shared_key,
            )
        
        # Split keys for separate MLPs (if shared_mlp was created, keys already split)
        if self.shared_mlp is None:
            key, q_key, v_key = random.split(key, 3)
        
        # Create Q separate MLP
        self.q_separate_mlp = MLP(
            input_dim = shared_output_dim,
            output_dim = n_actions,
            hidden_dims = separate_hidden_dims,
            activation = activation,
            use_bias = use_bias,
            key = q_key,
        )
        
        # Create V separate MLP
        self.v_separate_mlp = MLP(
            input_dim = shared_output_dim,
            output_dim = 1,
            hidden_dims = separate_hidden_dims,
            activation = activation,
            use_bias = use_bias,
            key = v_key,
        )
    
    def compute_shared_representation(self, observation: Float[Array, 'input_dim']) -> Float[Array, 'shared_dim']:
        """Compute shared representation from observation."""
        if self.shared_mlp is None:
            return observation
        return self.shared_mlp(observation)
    
    def action_values(self, observation: Float[Array, 'input_dim']) -> Float[Array, 'n_actions']:
        shared_repr = self.compute_shared_representation(observation)
        return self.q_separate_mlp(shared_repr)
    
    def state_value(self, observation: Float[Array, 'input_dim']) -> Float[Array, '1']:
        shared_repr = self.compute_shared_representation(observation)
        return self.v_separate_mlp(shared_repr)
    
    def action_and_state_values(self, observation: Float[Array, 'input_dim']) -> Tuple[Float[Array, 'n_actions'], Float[Array, '1']]:
        shared_repr = self.compute_shared_representation(observation)
        q_values = self.q_separate_mlp(shared_repr)
        state_val = self.v_separate_mlp(shared_repr)
        return q_values, state_val

    def select_action(
        self,
        observation: Float[Array, 'obs_dim'],
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
        q_vals = self.action_values(observation)
        num_actions = q_vals.shape[0]
        
        # Epsilon-greedy: random with prob epsilon, greedy otherwise
        key1, key2 = random.split(key)
        explore = random.uniform(key1) < epsilon
        
        # Random action
        random_action = random.randint(key2, (), 0, num_actions)
        
        # Greedy action
        greedy_action = jnp.argmax(q_vals)
        
        return jnp.where(explore, random_action, greedy_action)
    
    def compute_grads_and_loss(
        self,
        obs: Float[Array, 'obs_dim'],
        action: int,
        reward: float,
        next_obs: Float[Array, 'obs_dim'],
        gamma: float,
    ) -> Tuple[Any, float]:
        """Compute gradients for Q-learning update.
        
        Args:
            agent: Current agent
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            gamma: Discount factor
            
        Returns:
            Tuple of (gradients, TD error squared as loss)
        """
        def loss_fn(network: QVNetwork) -> jax.Array:
            action_vals, state_val = network.action_and_state_values(obs)
            
            # State value TD loss
            state_val_target = jnp.max(action_vals, keepdims=True)
            state_val_loss = (jax.lax.stop_gradient(state_val_target) - state_val) ** 2
            
            # Action value TD loss
            next_state_val = network.state_value(next_obs)
            action_val_target = reward + gamma * next_state_val
            action_val_loss = (jax.lax.stop_gradient(action_val_target) - action_vals[action]) ** 2
            
            total_loss = jnp.sum(state_val_loss + action_val_loss)
            
            return total_loss
        
        # Compute gradients using automatic differentiation
        loss, gradients = eqx.filter_value_and_grad(loss_fn)(self)
        
        return gradients, loss