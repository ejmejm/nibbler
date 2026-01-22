from typing import Optional, Tuple, Any
import argparse
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import equinox.nn as nn
import numpy as np
from tqdm import tqdm
import mlflow
import optax
from catch_env import CatchEnvironment, CatchEnvironmentState
from utils import tree_replace


UNROLL_STEPS = 4


class LinearQAgent(eqx.Module):
    """
    Linear Q-learning agent using linear function approximation.
    Q(s, a) = W[a, :]^T * s
    """
    layer: nn.Linear
    
    def __init__(
        self,
        num_actions: int,
        obs_dim: int,
        key: random.PRNGKey,
    ):
        """Initialize the linear Q-learning agent."""
        # Create a linear layer: obs_dim -> num_actions, no bias
        self.layer = nn.Linear(obs_dim, num_actions, use_bias=False, key=key)
    
    def q_values(self, observation: jax.Array) -> jax.Array:
        """
        Compute Q-values for all actions given an observation.
        
        Args:
            observation: Observation vector of shape (obs_dim,)
            
        Returns:
            Q-values for all actions, shape (num_actions,)
        """
        return self.layer(observation)
    
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


def create_optimizer(
    learning_rate: float,
    momentum: float = 0.9,
) -> optax.GradientTransformation:
    return optax.sgd(learning_rate=learning_rate, momentum=momentum)


def compute_gradients(
    agent: LinearQAgent,
    obs: jax.Array,
    action: int,
    reward: float,
    next_obs: jax.Array,
    gamma: float,
) -> Tuple[Any, float]:
    """
    Compute gradients for Q-learning update.
    
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
    def loss_fn(agent: LinearQAgent) -> jax.Array:
        # Current Q-value for the action taken
        q_vals = agent.q_values(obs)
        q_current = q_vals[action]
        
        # Target Q-value (using stop_gradient to prevent gradients from flowing through target)
        next_q_vals = jax.lax.stop_gradient(agent.q_values(next_obs))
        q_target = reward + gamma * jnp.max(next_q_vals)
        
        # TD error
        td_error = q_target - q_current
        
        # Loss is squared TD error
        return td_error ** 2
    
    # Compute gradients using automatic differentiation
    loss, gradients = eqx.filter_value_and_grad(loss_fn)(agent)
    
    return gradients, loss


class TrainState(eqx.Module):
    """Training state for linear Q-learning."""
    # Static
    gamma: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    log_interval: int = eqx.field(static=True)
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    
    # Non-static
    agent: LinearQAgent
    env_state: CatchEnvironmentState
    rng: random.PRNGKey
    optimizer_state: optax.OptState
    step: jax.Array = jnp.array(0)


def create_train_state(
    env_state: CatchEnvironmentState,
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
    
    # Get environment dimensions
    obs_dim = CatchEnvironment.observation_space_size(env_state)
    num_actions = CatchEnvironment.action_space_size(env_state)
    
    # Initialize agent
    key, agent_key = random.split(key)
    agent = LinearQAgent(num_actions, obs_dim, agent_key)
    
    # Initialize optimizer state
    # Compute a dummy gradient to get the structure
    def dummy_loss(agent):
        return 0.0
    _, dummy_grads = eqx.filter_value_and_grad(dummy_loss)(agent)
    optimizer_state = optimizer.init(dummy_grads)
    
    # Reset environment
    key, reset_key = random.split(key)
    env_state, _ = CatchEnvironment.reset(env_state, reset_key)
    
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
    obs = CatchEnvironment._get_observation(train_state.env_state)
    
    # Select action
    key, action_key = random.split(train_state.rng)
    action = train_state.agent.select_action(
        observation = obs,
        epsilon = train_state.epsilon,
        key = action_key,
    )
    
    # Take step
    new_env_state, next_obs, reward, info = CatchEnvironment.step(
        train_state.env_state,
        action,
    )
    
    # Compute gradients
    gradients, loss = compute_gradients(
        agent = train_state.agent,
        obs = obs,
        action = action,
        reward = reward,
        next_obs = next_obs,
        gamma = train_state.gamma,
    )
    
    # Apply gradients using optax directly
    updates, new_optimizer_state = train_state.optimizer.update(
        gradients,
        train_state.optimizer_state,
    )
    updated_agent = eqx.apply_updates(train_state.agent, updates)
    
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
        'loss': loss,
    }
    
    return new_train_state, metrics


def train_linear_q(
    train_state: TrainState,
    num_steps: int = 100000,
) -> TrainState:
    """
    Train a linear Q-learning agent on the Catch environment.
    
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
        avg_loss = metrics['loss'].mean()
        
        # Update progress bar
        pbar.update(log_interval)
        pbar.set_postfix({
            'avg_reward': f'{avg_reward:.2f}',
            'avg_loss': f'{avg_loss:.4f}',
        })
        
        # Log to MLFlow
        mlflow.log_metrics({
            'avg_reward': avg_reward,
            'avg_loss': avg_loss,
        }, step=train_state.step.item())
    
    pbar.close()
    
    return train_state


def run_experiment():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train linear Q-learning agent')
    parser.add_argument('--epsilon', type=float, default=0.1,
                        help='Exploration probability (default: 0.1)')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor (default: 0.99)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.0,
                        help='Momentum coefficient for SGD (default: 0.99)')
    parser.add_argument('--log_interval', type=int, default=10_000,
                        help='Logging interval in steps (default: 10000)')
    parser.add_argument('--num_steps', type=int, default=1_000_000,
                        help='Total number of training steps (default: 1000000)')
    
    args = parser.parse_args()
    
    # Create optimizer
    optimizer = create_optimizer(
        learning_rate = args.learning_rate,
        momentum = args.momentum,
    )
    
    # Create environment
    env_state = CatchEnvironmentState(
        rows = 10,
        cols = 5,
        hot_prob = 1.0,
        reset_prob = 1.0,
        seed = args.seed,
    )
    
    # Create training state
    train_state = create_train_state(
        env_state = env_state,
        optimizer = optimizer,
        gamma = args.gamma,
        epsilon = args.epsilon,
        seed = args.seed,
        log_interval = args.log_interval,
    )
    
    # Start MLFlow run and log hyperparameters
    mlflow.start_run()
    mlflow.log_params({
        'learning_rate': args.learning_rate,
        'momentum': args.momentum,
        'gamma': train_state.gamma,
        'epsilon': train_state.epsilon,
        'num_steps': args.num_steps,
        'log_interval': train_state.log_interval,
        'obs_dim': CatchEnvironment.observation_space_size(train_state.env_state),
        'num_actions': CatchEnvironment.action_space_size(train_state.env_state),
        'env_rows': train_state.env_state.rows,
        'env_cols': train_state.env_state.cols,
        'env_hot_prob': train_state.env_state.hot_prob,
        'env_reset_prob': train_state.env_state.reset_prob,
    })
    
    # Train agent
    train_state = train_linear_q(
        train_state = train_state,
        num_steps = args.num_steps,
    )
    
    mlflow.end_run()
    print('Training complete!')


if __name__ == '__main__':
    run_experiment()
