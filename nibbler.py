import argparse
from functools import partial
from typing import Optional, Tuple, List, Any

import equinox as eqx
import equinox.nn as nn
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Float, Int
import mlflow
import numpy as np
import optax
from tqdm import tqdm

from catch_env import (
    CatchEnvironmentState, 
    MultiCatchEnvironment,
)
from networks import MLP, QVNetwork
from utils import configure_jax_config, tree_replace


UNROLL_STEPS = 4


class Nibbler(eqx.Module):
    total_feature_count: int = eqx.field(static=True)
    n_gvfs: int = eqx.field(static=True)
    hidden_dim_per_gvf: int = eqx.field(static=True)
    
    output_layer: QVNetwork
    gvf_networks: QVNetwork # vampped for n_gvfs
    reward_predictor: MLP
    gvf_input_feature_idxs: Int[Array, 'n_gvfs inputs_per_gvf']
    gvf_cumulant_feature_idxs: Int[Array, 'n_gvfs']
    rng: random.PRNGKey
    
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        hidden_dim_per_gvf: int,
        inputs_per_gvf: int,
        n_gvfs: int,
        *,
        key: random.PRNGKey,
    ):
        """
        Initialize the deep Q-learning agent.
        
        Args:
            n_actions: Number of actions
            obs_dim: Observation dimension
            hidden_dim_per_gvf: Number of hidden units per GVF
            inputs_per_gvf: Number of inputs per GVF
            n_gvfs: Number of GVFs
            key: Random key for initialization
        """
        self.rng, output_key, reward_predictor_key, gvf_key = random.split(key, 4)
        
        self.n_gvfs = n_gvfs
        self.hidden_dim_per_gvf = hidden_dim_per_gvf
        self.total_feature_count = obs_dim + hidden_dim_per_gvf * n_gvfs
        
        # Output layer: QVNetwork with linear action value and value functions
        self.output_layer = QVNetwork(
            n_actions = n_actions,
            input_dim = self.total_feature_count,
            shared_hidden_dims = [],
            separate_hidden_dims = [],
            activation = jax.nn.relu,
            use_bias = False,
            key = output_key,
        )
        
        # Linear predictor of the reward
        self.reward_predictor = MLP(
            input_dim = obs_dim,
            output_dim = 1,
            use_bias = False,
            key = reward_predictor_key,
        )
        
        # GVF QV networks: shared input layer (obs_dim -> hidden_dim_per_gvf) + linear Q/V
        gvf_keys = random.split(gvf_key, n_gvfs)
        self.gvf_networks = jax.vmap(
            partial(
                QVNetwork,
                n_actions = n_actions,
                input_dim = inputs_per_gvf,
                shared_hidden_dims = [hidden_dim_per_gvf],
                separate_hidden_dims = [],
                activation = jax.nn.relu,
                use_bias = False,
            ),
            in_axes=0,
        )(key=gvf_keys)
        
        self.gvf_cumulant_feature_idxs = jax.random.choice(
            self.rng, obs_dim, (n_gvfs,), replace=False)
        self.gvf_input_feature_idxs = jax.vmap(
            lambda key: jax.random.choice(key, obs_dim, (inputs_per_gvf,), replace=False)
        )(jax.random.split(self.rng, n_gvfs))
        
    def get_gvf_inputs(self, observation: Float[Array, 'obs_dim']) -> Float[Array, 'n_gvfs inputs_per_gvf']:
        return observation.at[self.gvf_input_feature_idxs].get(mode='promise_in_bounds')
    
    def get_gvf_cumulants(self, observation: Float[Array, 'obs_dim']) -> Float[Array, 'n_gvfs']:
        return observation.at[self.gvf_cumulant_feature_idxs].get(mode='promise_in_bounds')
    
    def compute_gvf_features(self, observation: Float[Array, 'obs_dim']) -> Float[Array, 'n_gvfs hidden_dim_per_gvf']:
        gvf_inputs = self.get_gvf_inputs(observation)
        batch_compute_features_fn = jax.vmap(
            lambda model, inputs: model.compute_shared_representation(inputs),
            in_axes = 0,
        )
        gvf_features = batch_compute_features_fn(self.gvf_networks, gvf_inputs)
        return gvf_features
    
    def compute_all_features(self, observation: Float[Array, 'obs_dim']) -> Float[Array, 'total_feature_count']:
        gvf_features = self.compute_gvf_features(observation)  # Shape: (n_gvfs, hidden_dim_per_gvf)
        gvf_features_flat = gvf_features.flatten()  # Shape: (n_gvfs * hidden_dim_per_gvf,)
        return jnp.concatenate([observation, gvf_features_flat], axis=0)
    
    def q_values(self, observation: Float[Array, 'obs_dim']) -> Float[Array, 'n_actions']:
        """
        Compute Q-values for all actions given an observation.
        
        Args:
            observation: Observation vector of shape (obs_dim,)
            
        Returns:
            Q-values for all actions, shape (num_actions,)
        """
        features = self.compute_all_features(observation)
        q_values = self.output_layer.action_values(features)
        return q_values
    
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
        ### First compute gradients for the output layer ###
        
        all_features = self.compute_all_features(obs)
        next_all_features = self.compute_all_features(next_obs)
        output_layer_grads, output_layer_loss = self.output_layer.compute_grads_and_loss(
            obs = all_features,
            action = action,
            reward = reward,
            next_obs = next_all_features,
            gamma = gamma,
        )
        
        ### Then compute gradients for the reward predictor ###
        
        def reward_loss_fn(reward_predictor: MLP) -> float:
            reward_prediction = reward_predictor(obs)
            reward_loss = jnp.sum((reward - reward_prediction) ** 2)
            return reward_loss
        
        reward_loss, reward_grads = jax.value_and_grad(reward_loss_fn)(self.reward_predictor)
        
        ### Then compute gradients for the GVF networks ###
        
        gvf_inputs = self.get_gvf_inputs(obs)
        next_gvf_inputs = self.get_gvf_inputs(next_obs)
        gvf_cumulants = self.get_gvf_cumulants(obs)
        
        batch_gvf_grad_fn = jax.vmap(
            lambda model, *args: model.compute_grads_and_loss(*args),
            in_axes = (0, 0, None, 0, 0, None),
        )
        gvf_grads, gvf_losses = batch_gvf_grad_fn(
            self.gvf_networks, gvf_inputs, action,
            gvf_cumulants, next_gvf_inputs, gamma,
        )
        
        ### Combine losses and gradients ###
        
        total_loss = output_layer_loss + jnp.sum(gvf_losses)
        gradients = eqx.filter(self, lambda x: jnp.issubdtype(x.dtype, jnp.floating))
        # Zero out in case there are any parameters we missed in the filter that should be frozen
        gradients = jax.tree.map(lambda x: jnp.zeros_like(x), gradients)
        gradients = tree_replace(
            gradients,
            output_layer = output_layer_grads,
            reward_predictor = reward_grads,
            gvf_networks = gvf_grads,
        )
        
        losses = {
            'output': output_layer_loss,
            'reward': reward_loss,
            'gvfs': gvf_losses,
            'total': total_loss,
        }
        
        return gradients, losses


def create_optimizer(
    learning_rate: float,
    momentum: float = 0.9,
) -> optax.GradientTransformation:
    """SGD optimizer with EMA-style momentum."""
    return optax.chain(
        optax.trace(decay=momentum, nesterov=False),
        optax.scale(1.0 - momentum),
        optax.scale(-learning_rate),
    )


def apply_gradients(
    agent: Nibbler,
    gradients: Any,
    optimizer_state: optax.OptState,
    optimizer: optax.GradientTransformation,
) -> Tuple[Nibbler, optax.OptState]:
    """
    Apply gradients to the agent using the optimizer.
    
    Args:
        agent: Current agent
        gradients: Gradients to apply (PyTree matching agent structure)
        optimizer_state: Current optimizer state
        optimizer: Optax optimizer
        
    Returns:
        Tuple of (updated agent, updated optimizer state)
    """
    # Compute updates from gradients
    updates, new_optimizer_state = optimizer.update(
        gradients,
        optimizer_state,
    )
    
    # Apply updates to agent
    updated_agent = eqx.apply_updates(agent, updates)
    
    return updated_agent, new_optimizer_state


class TrainState(eqx.Module):
    """Training state for deep Q-learning."""
    # Static
    gamma: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    log_interval: int = eqx.field(static=True)
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    
    # Non-static
    agent: Nibbler
    env_state: CatchEnvironmentState
    rng: random.PRNGKey
    optimizer_state: optax.OptState
    step: jax.Array = jnp.array(0)


def create_train_state(
    env_state: CatchEnvironmentState,
    agent: Nibbler,
    optimizer: optax.GradientTransformation,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    seed: Optional[int] = None,
    log_interval: int = 100,
) -> TrainState:
    """
    Create and initialize training state.
    
    Args:
        env_state: Initial environment state
        agent: Nibbler agent model
        optimizer: Optax optimizer to use for training
        gamma: Discount factor
        epsilon: Exploration probability (constant)
        seed: Random seed
        log_interval: Interval for logging metrics to MLFlow
        
    Returns:
        Initialized TrainState
    """
    # Initialize random key
    if seed is None:
        seed = np.random.randint(0, 1000000000)
    key = random.PRNGKey(seed)
    
    # Initialize optimizer state for output_layer only
    trainable_params = eqx.filter(agent, lambda x: jnp.issubdtype(x.dtype, jnp.floating))
    optimizer_state = optimizer.init(trainable_params)
    
    # Reset environment
    env_state, _ = MultiCatchEnvironment.reset(env_state)
    
    return TrainState(
        gamma = gamma,
        epsilon = epsilon,
        log_interval = log_interval,
        optimizer = optimizer,
        agent = agent,
        env_state = env_state,
        rng = key,
        optimizer_state = optimizer_state,
    )


def train_step(train_state: TrainState) -> Tuple[TrainState, dict]:
    """
    Perform one training step.
    
    Returns:
        Tuple of (updated train_state, metrics dict)
    """
    # Get current observation
    obs = MultiCatchEnvironment._get_observation(train_state.env_state)
    
    # Select action
    key, action_key = random.split(train_state.rng)
    action = train_state.agent.select_action(
        observation = obs,
        epsilon = train_state.epsilon,
        key = action_key,
    )
    
    # Take step
    new_env_state, next_obs, reward, info = MultiCatchEnvironment.step(
        train_state.env_state, 
        action
    )
    
    # Compute gradients
    gradients, losses = train_state.agent.compute_grads_and_loss(
        obs = obs,
        action = action,
        reward = reward,
        next_obs = next_obs,
        gamma = train_state.gamma,
    )
    
    # Apply gradients to output_layer only
    updated_agent, new_optimizer_state = apply_gradients(
        agent = train_state.agent,
        gradients = gradients,
        optimizer_state = train_state.optimizer_state,
        optimizer = train_state.optimizer,
    )
    
    # Update train state
    new_train_state = tree_replace(
        train_state,
        agent = updated_agent,
        env_state = new_env_state,
        step = train_state.step + 1,
        rng = key,
        optimizer_state = new_optimizer_state,
    )
    
    # Compute metrics
    metrics = {
        'reward': reward,
        'losses': losses,
    }
    
    return new_train_state, metrics


def train_deep_q(
    train_state: TrainState,
    num_steps: int = 100000,
) -> TrainState:
    """
    Train a deep Q-learning agent on the Catch environment.
    
    Args:
        train_state: Initial training state
        num_steps: Number of training steps
        
    Returns:
        Trained train state
    """
    # Calculate number of scan iterations
    log_interval = train_state.log_interval
    num_scans = num_steps // log_interval
    
    @jax.jit
    def multi_step_train(train_state: TrainState) -> Tuple[TrainState, dict]:
        train_state, metrics = jax.lax.scan(
            lambda state, _: train_step(state),
            train_state,
            length = log_interval,
            unroll = UNROLL_STEPS,
        )
        return train_state, metrics
    
    # Progress bar
    pbar = tqdm(total=num_steps, desc='Training')
    
    # Run full batch iterations
    for _ in range(num_scans):
        train_state, metrics = multi_step_train(train_state)
        
        # Average metrics
        avg_reward = metrics['reward'].mean()
        avg_output_loss = metrics['losses']['output'].mean()
        avg_reward_loss = metrics['losses']['reward'].mean()
        avg_gvfs_loss = metrics['losses']['gvfs'].mean()
        avg_total_loss = metrics['losses']['total'].mean()
        
        # Update progress bar
        pbar.update(log_interval)
        pbar.set_postfix({
            'avg_reward': f'{avg_reward:.2f}',
            'avg_loss': f'{avg_output_loss:.4f}',
        })
        
        # Log to MLFlow
        mlflow.log_metrics({
            'avg_reward': avg_reward,
            'avg_output_loss': avg_output_loss,
            'avg_reward_loss': avg_reward_loss,
            'avg_gvfs_loss': avg_gvfs_loss,
            'avg_total_loss': avg_total_loss,
        }, step=train_state.step.item())
    
    pbar.close()
    
    return train_state


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train deep Q-learning agent')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration probability (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: None)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate scaling factor, which is multiplied by sqrt(2)/sqrt(n_gvfs) (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='Momentum coefficient for SGD (default: 0.99)')
    parser.add_argument('--hidden_dims', type=int, nargs='*', default=[256],
                        help='Hidden layer dimensions (default: None, linear network)')
    parser.add_argument('--log_interval', type=int, default=10_000,
                        help='Logging interval in steps (default: 10000)')
    parser.add_argument('--num_steps', type=int, default=1_000_000,
                        help='Total number of training steps (default: 1000000)')
    parser.add_argument('--num_envs', type=int, default=2,
                        help='Total number of environments (default: 2)')
    
    ### Nibbler-specific arguments ###
    
    parser.add_argument('--n_gvfs', type=int, default=10,
                        help='Number of GVFs (default: 10)')
    parser.add_argument('--inputs_per_gvf', type=int, default=82,
                        help='Number of inputs per GVFs (default: 82)')
    parser.add_argument('--hidden_dim_per_gvf', type=int, default=256,
                        help='Number of hidden units per GVFs (default: 256)')
    
    args = parser.parse_args()
    args.learning_rate *= np.sqrt(2) / np.sqrt(args.n_gvfs)
    configure_jax_config()
    
    key = (
        jax.random.PRNGKey(args.seed) if args.seed is not None
        else jax.random.PRNGKey(np.random.randint(0, 1_000_000_000))
    )
    seeds = jax.random.randint(key, (args.num_envs + 1,), 0, 1_000_000_000)
    env_seeds, train_state_seed = seeds[:args.num_envs], seeds[args.num_envs]
    
    # Create optimizer
    optimizer = create_optimizer(
        learning_rate = args.learning_rate,
        momentum = args.momentum,
    )
    
    # Create environment
    env_state = jax.vmap(
        partial(CatchEnvironmentState,
            rows = 10,
            cols = 5,
            hot_prob = min(2.0 / args.num_envs, 1.0),
            reset_prob = 1.0, # 0.2,
            paddle_noise = 0.0, # 0.2,
            reward_delivery_prob = 1.0, # 0.2,
        ),
        in_axes = 0,
    )(seed=env_seeds)
    
    # Get environment dimensions
    obs_dim = MultiCatchEnvironment.observation_space_size(env_state)
    num_actions = MultiCatchEnvironment.action_space_size(env_state)
    
    # Create agent
    key, agent_key = random.split(key)
    agent = Nibbler(
        n_actions = num_actions,
        obs_dim = obs_dim,
        hidden_dim_per_gvf = args.hidden_dim_per_gvf,
        inputs_per_gvf = args.inputs_per_gvf,
        n_gvfs = args.n_gvfs,
        key = agent_key,
    )
    
    # Create training state
    train_state = create_train_state(
        env_state = env_state,
        agent = agent,
        optimizer = optimizer,
        gamma = args.gamma,
        epsilon = args.epsilon,
        seed = train_state_seed,
        log_interval = args.log_interval,
    )
    
    # Start MLFlow run and log hyperparameters
    mlflow.start_run()
    args_dict = vars(args)
    mlflow.log_params(args_dict)
    
    # Train agent
    train_state = train_deep_q(
        train_state = train_state,
        num_steps = args.num_steps,
    )
    
    mlflow.end_run()
    print('Training complete!')


if __name__ == '__main__':
    main()
