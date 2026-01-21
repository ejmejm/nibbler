from typing import Optional, Tuple
import argparse
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import numpy as np
from tqdm import tqdm
import mlflow
from catch_env import CatchEnvironment, CatchEnvironmentState
from utils import tree_replace


class LinearQAgent(eqx.Module):
    """
    Linear Q-learning agent using linear function approximation.
    Q(s, a) = W[a, :]^T * s
    """
    weights: jax.Array  # Shape: (num_actions, obs_dim)
    
    def __init__(
        self,
        num_actions: int,
        obs_dim: int,
        key: random.PRNGKey,
        init_scale: float = 0.01,
    ):
        """Initialize the linear Q-learning agent."""
        # Initialize weights with small random values
        self.weights = random.normal(key, (num_actions, obs_dim)) * init_scale
    
    def q_values(self, observation: jax.Array) -> jax.Array:
        """
        Compute Q-values for all actions given an observation.
        
        Args:
            observation: Observation vector of shape (obs_dim,)
            
        Returns:
            Q-values for all actions, shape (num_actions,)
        """
        return jnp.dot(self.weights, observation)
    
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
        
        # Epsilon-greedy: random with prob epsilon, greedy otherwise
        key1, key2 = random.split(key)
        explore = random.uniform(key1) < epsilon
        
        # Random action
        random_action = random.randint(key2, (), 0, self.weights.shape[0])
        
        # Greedy action
        greedy_action = jnp.argmax(q_vals)
        
        return jnp.where(explore, random_action, greedy_action).item()
    
    def update(
        self,
        obs: jax.Array,
        action: int,
        reward: float,
        next_obs: jax.Array,
        gamma: float,
        learning_rate: float,
    ) -> Tuple['LinearQAgent', float]:
        """
        Update Q-values using Q-learning update rule.
        
        Q(s, a) ← Q(s, a) + α[r + γ max_a' Q(s', a') - Q(s, a)]
        
        Args:
            obs: Current observation
            action: Action taken
            reward: Reward received
            next_obs: Next observation
            gamma: Discount factor
            learning_rate: Learning rate (alpha)
            
        Returns:
            Tuple of (updated agent, TD error squared as loss)
        """
        # Current Q-value
        q_current = jnp.dot(self.weights[action], obs)
        
        # Target Q-value
        next_q_vals = self.q_values(next_obs)
        q_target = reward + gamma * jnp.max(next_q_vals)
        
        # TD error
        td_error = q_target - q_current
        
        # Update weights for the action taken
        # ∇_w Q(s, a) = s, so update is: w[a] ← w[a] + α * td_error * s
        weight_update = learning_rate * td_error * obs
        
        # Update the weights
        new_weights = self.weights.at[action].add(weight_update)
        
        # Return updated agent and squared TD error as loss
        loss = float(td_error ** 2)
        updated_agent = tree_replace(self, weights=new_weights)
        
        return updated_agent, loss


class TrainState(eqx.Module):
    """Training state for linear Q-learning."""
    # Static
    learning_rate: float = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    log_interval: int = eqx.field(static=True)
    
    # Non-static
    agent: LinearQAgent
    env_state: CatchEnvironmentState
    step: jax.Array
    cumulative_reward: jax.Array
    cumulative_loss: jax.Array
    rng: random.PRNGKey


def create_train_state(
    env_state: CatchEnvironmentState,
    learning_rate: float = 0.01,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    init_scale: float = 0.01,
    seed: Optional[int] = None,
    log_interval: int = 100,
) -> TrainState:
    """
    Create and initialize training state.
    
    Args:
        env_state: Initial environment state
        learning_rate: Learning rate for Q-learning
        gamma: Discount factor
        epsilon: Exploration probability (constant)
        init_scale: Initial weight scale
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
    agent = LinearQAgent(num_actions, obs_dim, agent_key, init_scale=init_scale)
    
    # Reset environment
    key, reset_key = random.split(key)
    env_state, _ = CatchEnvironment.reset(env_state, reset_key)
    
    return TrainState(
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon,
        log_interval=log_interval,
        agent=agent,
        env_state=env_state,
        step=jnp.array(0),
        cumulative_reward=jnp.array(0.0),
        cumulative_loss=jnp.array(0.0),
        rng=key,
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
        obs,
        train_state.epsilon,
        action_key,
    )
    
    # Take step
    key, step_key = random.split(key)
    new_env_state, next_obs, reward, info = CatchEnvironment.step(
        train_state.env_state,
        action,
    )
    
    # Update agent
    agent, loss = train_state.agent.update(
        obs=obs,
        action=action,
        reward=reward.item(),
        next_obs=next_obs,
        gamma=train_state.gamma,
        learning_rate=train_state.learning_rate,
    )
    
    # Update train state
    new_train_state = tree_replace(
        train_state,
        agent=agent,
        env_state=new_env_state,
        step=train_state.step + 1,
        cumulative_reward=train_state.cumulative_reward + reward,
        cumulative_loss=train_state.cumulative_loss + jnp.array(loss),
        rng=key,
    )
    
    # Compute metrics
    metrics = {
        'reward': float(reward.item()),
        'loss': loss,
        'epsilon': train_state.epsilon,
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
    # Start MLFlow run
    mlflow.start_run()
    
    # Log hyperparameters
    mlflow.log_params({
        'learning_rate': train_state.learning_rate,
        'gamma': train_state.gamma,
        'epsilon': train_state.epsilon,
        'num_steps': num_steps,
        'log_interval': train_state.log_interval,
        'obs_dim': CatchEnvironment.observation_space_size(train_state.env_state),
        'num_actions': CatchEnvironment.action_space_size(train_state.env_state),
        'env_rows': train_state.env_state.rows,
        'env_cols': train_state.env_state.cols,
        'env_hot_prob': train_state.env_state.hot_prob,
        'env_reset_prob': train_state.env_state.reset_prob,
    })
    
    # Track metrics for averaging
    reward_window = []
    loss_window = []
    
    # Progress bar
    pbar = tqdm(range(num_steps), desc='Training')
    
    for _ in pbar:
        train_state, metrics = train_step(train_state)
        
        reward_window.append(metrics['reward'])
        loss_window.append(metrics['loss'])
        
        # Keep window size manageable
        if len(reward_window) > train_state.log_interval:
            reward_window.pop(0)
            loss_window.pop(0)
        
        # Update progress bar
        avg_reward = np.mean(reward_window) if reward_window else 0.0
        avg_loss = np.mean(loss_window) if loss_window else 0.0
        reward_rate = train_state.cumulative_reward.item() / max(train_state.step.item(), 1)
        
        pbar.set_postfix({
            'reward': f'{metrics["reward"]:.2f}',
            'avg_reward': f'{avg_reward:.2f}',
            'reward_rate': f'{reward_rate:.4f}',
            'loss': f'{avg_loss:.4f}',
            'epsilon': f'{metrics["epsilon"]:.3f}',
        })
        
        # Log to MLFlow
        if train_state.step.item() % train_state.log_interval == 0:
            mlflow.log_metrics({
                'reward': metrics['reward'],
                'avg_reward': avg_reward,
                'reward_rate': reward_rate,
                'loss': avg_loss,
                'epsilon': metrics['epsilon'],
                'cumulative_reward': train_state.cumulative_reward.item(),
                'cumulative_loss': train_state.cumulative_loss.item(),
            }, step=train_state.step.item())
    
    # Log final metrics
    final_reward_rate = train_state.cumulative_reward.item() / max(train_state.step.item(), 1)
    final_avg_loss = train_state.cumulative_loss.item() / max(train_state.step.item(), 1)
    mlflow.log_metrics({
        'final_reward_rate': final_reward_rate,
        'final_avg_loss': final_avg_loss,
        'final_cumulative_reward': train_state.cumulative_reward.item(),
    })
    
    mlflow.end_run()
    
    return train_state


def main():
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
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Logging interval in steps (default: 100)')
    parser.add_argument('--num_steps', type=int, default=100000,
                        help='Total number of training steps (default: 100000)')
    
    args = parser.parse_args()
    
    # Create environment
    env_state = CatchEnvironmentState(
        rows=10,
        cols=5,
        hot_prob=1.0,
        reset_prob=1.0,
        seed=args.seed,
    )
    
    # Create training state
    train_state = create_train_state(
        env_state=env_state,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon=args.epsilon,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    
    # Train agent
    train_state = train_linear_q(
        train_state=train_state,
        num_steps=args.num_steps,
    )
    
    print('Training complete!')


if __name__ == '__main__':
    main()
