from typing import Optional, Tuple, Dict, Any, List
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import numpy as np


class CatchEnvironmentState(eqx.Module):
    """
    State of the Catch environment containing all parameters (both static and dynamic).
    """
    # Static parameters (configuration)
    rows: int = eqx.field(static=True)
    cols: int = eqx.field(static=True)
    hot_prob: float
    reset_prob: float
    reward_indicator_duration_min: int
    reward_indicator_duration_max: int
    paddle_noise: float
    
    # Dynamic parameters (state)
    rng: random.PRNGKey
    ball_row: jax.Array  # Current row of the ball
    ball_col: jax.Array  # Current column of the ball
    paddle_col: jax.Array  # Current column of the paddle
    in_reset: jax.Array  # Whether the ball is in reset state
    is_hot: jax.Array  # Whether the board is hot
    catch_bit: jax.Array  # Whether the ball was just caught
    miss_bit: jax.Array  # Whether the ball was just missed
    plus_bit: jax.Array  # Whether a positive reward is forthcoming
    minus_bit: jax.Array  # Whether a negative reward is forthcoming
    reward_countdown: jax.Array  # Countdown for reward after plus/minus bit activation
    
    def __init__(
        self,
        rows: int = 10,
        cols: int = 5,
        hot_prob: float = 0.3,
        reset_prob: float = 0.2,
        reward_indicator_duration_min: int = 1,
        reward_indicator_duration_max: int = 3,
        paddle_noise: float = 0.0,
        seed: Optional[int] = None,
    ):
        """Initialize the Catch environment state with the given parameters."""
        super().__init__()
        
        # Store static configuration
        self.rows = rows
        self.cols = cols
        self.hot_prob = hot_prob
        self.reset_prob = reset_prob
        self.reward_indicator_duration_min = reward_indicator_duration_min
        self.reward_indicator_duration_max = reward_indicator_duration_max
        self.paddle_noise = paddle_noise
        
        # Set up RNG
        if seed is None:
            seed = np.random.randint(0, 1000000000)
        self.rng = random.PRNGKey(seed)
        
        # Start with ball in reset state
        self.ball_row = jnp.array(-1)  # -1 represents reset state
        self.ball_col = jnp.array(0)
        
        # Start with paddle in middle position
        self.paddle_col = jnp.array(cols // 2)
        
        # Initialize all bits as inactive
        self.in_reset = jnp.array(True)
        self.is_hot = jnp.array(False)
        self.catch_bit = jnp.array(False)
        self.miss_bit = jnp.array(False)
        self.plus_bit = jnp.array(False)
        self.minus_bit = jnp.array(False)
        self.reward_countdown = jnp.array(0)


class CatchEnvironment(eqx.Module):
    """
    An implementation of the Catch environment with additional features like hot bit,
    reset bit, catch bit, miss bit, plus bit, and minus bit.
    
    This class contains only static methods. All parameters and state are stored in
    CatchEnvironmentState.
    """
    
    @staticmethod
    def _get_observation(state: CatchEnvironmentState) -> jax.Array:
        """
        Construct the observation vector based on the current state.
        The observation is a 1D array of 50 bits for the board, plus 6 bits for
        hot, reset, catch, miss, plus, and minus bits.
        """
        # Initialize empty board (10x5 = 50 elements)
        board = jnp.zeros((state.rows, state.cols), dtype=jnp.int32)
        
        # Place ball on board if it's not in reset
        valid_ball_pos = ~state.in_reset & (state.ball_row >= 0) & (state.ball_row < state.rows)
        board = board.at[state.ball_row, state.ball_col].set(jnp.where(valid_ball_pos, 1, 0))
        
        # Place paddle on board (always on bottom row)
        board = board.at[state.rows - 1, state.paddle_col].set(1)
        
        # Flatten board and append special bits
        flat_board = board.flatten()
        
        # Append special bits: [hot, reset, catch, miss, plus, minus]
        special_bits = jnp.array([
            state.is_hot, 
            state.in_reset, 
            state.catch_bit, 
            state.miss_bit, 
            state.plus_bit, 
            state.minus_bit
        ], dtype=jnp.int32)
        
        # Concatenate board and special bits
        observation = jnp.concatenate([flat_board, special_bits])
        
        return observation
    
    @staticmethod
    def step(state: CatchEnvironmentState, action: int) -> Tuple[CatchEnvironmentState, jax.Array, jax.Array, Dict]:
        """
        Take a step in the environment based on the provided action.
        
        Args:
            state: The current environment state
            action: 0 (left), 1 (stay), or 2 (right)
            
        Returns:
            Tuple of (new_state, observation, reward, info)
        """
        # Apply paddle noise: with probability paddle_noise, replace action with random action
        key, subkey_noise = random.split(state.rng)
        subkey_noise, subkey_action = random.split(subkey_noise)
        noise_sample = random.uniform(subkey_noise)
        should_apply_noise = noise_sample < state.paddle_noise
        random_action = random.randint(subkey_action, (), 0, 3)
        effective_action = jnp.where(should_apply_noise, random_action, action)
        
        # Update paddle position based on effective action
        # 0: left, 1: stay, 2: right
        paddle_col = jnp.clip(
            state.paddle_col + jnp.array([-1, 0, 1])[effective_action],
            0,
            state.cols - 1,
        )
        
        # Initialize reward to zero
        reward = jnp.array(0.0)
        
        # Get current state
        ball_row = state.ball_row
        ball_col = state.ball_col
        in_reset = state.in_reset
        is_hot = state.is_hot
        catch_bit = state.catch_bit
        miss_bit = state.miss_bit
        plus_bit = state.plus_bit
        minus_bit = state.minus_bit
        reward_countdown = state.reward_countdown
        
        # === HANDLE BIT FLOW: catch/miss -> plus/minus -> reset ===
        # Flow logic:
        # 1. If catch/miss is active, deactivate it and activate plus/minus (if hot) or reset (if not hot)
        # 2. If plus/minus is active and countdown reaches 0, deactivate it and activate reset
        # 3. Reset can only be active when catch/miss and plus/minus are both inactive
        
        # Check if catch/miss was active (from previous step)
        catch_was_active = catch_bit
        miss_was_active = miss_bit
        
        # Deactivate catch/miss bits (they were active for one step)
        catch_bit = jnp.array(False)
        miss_bit = jnp.array(False)
        
        # When catch/miss was active, transition to next state
        # If hot is True, activate plus/minus (and set countdown)
        # If hot is False, activate reset immediately
        should_activate_plus = catch_was_active & is_hot
        should_activate_minus = miss_was_active & is_hot
        should_activate_reset_from_catch_miss = (catch_was_active | miss_was_active) & ~is_hot
        
        # Generate random duration for plus/minus if needed
        key, subkey4 = random.split(key)
        duration = random.randint(
            subkey4, 
            (), 
            state.reward_indicator_duration_min,
            state.reward_indicator_duration_max + 1
        )
        
        # Activate plus/minus if catch/miss happened and hot is True
        plus_bit = jnp.where(
            should_activate_plus,
            True,
            plus_bit,
        )
        minus_bit = jnp.where(
            should_activate_minus,
            True,
            minus_bit,
        )
        reward_countdown = jnp.where(
            should_activate_plus | should_activate_minus,
            duration + 1, # +1 because it will be decremented this step
            reward_countdown,
        )
        
        # === HANDLE REWARD COUNTDOWN AND REWARD DELIVERY ===
        plus_minus_just_finished = reward_countdown == 1
        reward_countdown = jnp.where(
            reward_countdown > 0,
            reward_countdown - 1,
            reward_countdown,
        )
        
        # If reward countdown reached 0, issue reward
        reward = jnp.where(
            plus_minus_just_finished & plus_bit,
            1.0,
            reward,
        )
        reward = jnp.where(
            plus_minus_just_finished & minus_bit,
            -1.0,
            reward,
        )
        
        # When reward is delivered, deactivate plus/minus bits and turn off hot bit
        plus_bit = jnp.where(plus_minus_just_finished, False, plus_bit)
        minus_bit = jnp.where(plus_minus_just_finished, False, minus_bit)
        is_hot = jnp.where(plus_minus_just_finished, False, is_hot)
        
        # Activate reset if plus/minus just finished or if catch/miss happened without hot
        should_activate_reset = should_activate_reset_from_catch_miss | plus_minus_just_finished
        
        # == HANDLE BALL HITTING BOTTOM ROW ==
        # Check if ball is at bottom (before updating in_reset)
        ball_at_bottom = (ball_row == state.rows - 1) & ~in_reset
        
        # Determine if the ball is caught or missed
        is_caught = ball_at_bottom & (ball_col == paddle_col)
        is_missed = ball_at_bottom & (ball_col != paddle_col)
        
        # When ball hits bottom, set catch/miss bits (will be active next step)
        # Also move ball to reset position but don't set in_reset yet
        catch_bit = jnp.where(is_caught, True, catch_bit)
        miss_bit = jnp.where(is_missed, True, miss_bit)
        ball_row = jnp.where(ball_at_bottom, jnp.array(-1), ball_row)
        
        # == HANDLE BALL IN RESET STATE ==
        # Ball can only enter if in_reset was True in the PREVIOUS step
        # (not if it's being set to True in this step)
        key, subkey1 = random.split(key)
        will_enter_board = random.uniform(subkey1) < state.reset_prob
        
        # When in reset, ball may enter the board with probability reset_prob
        # Ball can only enter when in_reset was True in the previous step (no other dependencies)
        ball_enters = state.in_reset & will_enter_board & (ball_row == -1)
        
        # Set in_reset based on the flow logic (for next step)
        in_reset = jnp.where(should_activate_reset, True, in_reset)
        ball_row = jnp.where(should_activate_reset, jnp.array(-1), ball_row)
        
        # When ball enters, determine if board becomes hot (only set when entering)
        key, subkey2 = random.split(key)
        becomes_hot = random.uniform(subkey2) < state.hot_prob
        is_hot = jnp.where(ball_enters, becomes_hot, is_hot)
        
        # Place ball in top row at random column when entering
        key, subkey3 = random.split(key)
        new_col = random.randint(subkey3, (), 0, state.cols)
        ball_row = jnp.where(ball_enters, jnp.array(0), ball_row)
        ball_col = jnp.where(ball_enters, new_col, ball_col)
        in_reset = jnp.where(ball_enters, False, in_reset)
        
        # Advance ball if not in reset and not at bottom and not entering
        # Ball must be at a valid position (not -1) to advance
        normal_fall = ~in_reset & ~ball_at_bottom & ~ball_enters & (ball_row >= 0) & (ball_row < state.rows - 1)
        ball_row = jnp.where(normal_fall, ball_row + 1, ball_row)
        
        # Construct new state
        new_state = eqx.tree_at(
            lambda t: (
                t.rng, t.ball_row, t.ball_col, t.paddle_col, t.in_reset, 
                t.is_hot, t.catch_bit, t.miss_bit, t.plus_bit, t.minus_bit,
                t.reward_countdown
            ),
            state,
            (
                key, ball_row, ball_col, paddle_col, in_reset, 
                is_hot, catch_bit, miss_bit, plus_bit, minus_bit,
                reward_countdown
            )
        )
        
        # Get observation
        observation = CatchEnvironment._get_observation(new_state)
        
        # Info dictionary
        info = {
            "ball_row": ball_row,
            "ball_col": ball_col,
            "paddle_col": paddle_col,
            "in_reset": in_reset,
            "is_hot": is_hot,
            "catch_bit": catch_bit,
            "miss_bit": miss_bit,
            "plus_bit": plus_bit,
            "minus_bit": minus_bit,
            "reward_countdown": reward_countdown,
        }
        
        return new_state, jax.lax.stop_gradient(observation), jax.lax.stop_gradient(reward), info
    
    @staticmethod
    def reset(state: CatchEnvironmentState, key: Optional[random.PRNGKey] = None) -> Tuple[CatchEnvironmentState, jax.Array]:
        """Reset the environment to an initial state."""
        if key is None:
            key, subkey = random.split(state.rng)
        else:
            key, subkey = random.split(key)
        
        # Create a fresh state with the same static parameters
        new_state = CatchEnvironmentState(
            rows=state.rows,
            cols=state.cols,
            hot_prob=state.hot_prob,
            reset_prob=state.reset_prob,
            reward_indicator_duration_min=state.reward_indicator_duration_min,
            reward_indicator_duration_max=state.reward_indicator_duration_max,
            paddle_noise=state.paddle_noise,
            seed=random.randint(subkey, (), 0, 1000000000).item()
        )
        
        # Get observation
        observation = CatchEnvironment._get_observation(new_state)
        
        return new_state, observation
    
    @staticmethod
    def observation_space_size(state: CatchEnvironmentState) -> int:
        """Return the size of the observation space."""
        return state.rows * state.cols + 6  # board + 6 special bits
    
    @staticmethod
    def action_space_size(state: CatchEnvironmentState) -> int:
        """Return the size of the action space."""
        return 3  # left, stay, right


# Sanity check
def main():
    """Run a simple sanity check on the environment."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import imageio
    from io import BytesIO

    # Create environment state
    state = CatchEnvironmentState(seed=42, hot_prob=0.5, reset_prob=0.3)

    # Reset the environment
    state, obs = CatchEnvironment.reset(state)

    # Run a few steps
    total_reward = 0
    n_steps = 100

    # Store frames for GIF
    frames = []

    # Create a function to create a frame image
    def create_frame(obs, state, step_num):
        """Create a frame image for the current state."""
        board = obs[:state.rows * state.cols].reshape(state.rows, state.cols)
        special_bits = obs[state.rows * state.cols:]

        fig, ax = plt.subplots(figsize=(5, 10))
        ax.imshow(board, cmap='Blues')

        # Add text for special bits
        special_bit_names = ['Hot', 'Reset', 'Catch', 'Miss', 'Plus', 'Minus']
        special_bit_status = ['ON' if bit else 'OFF' for bit in special_bits]

        for i, (name, status) in enumerate(zip(special_bit_names, special_bit_status)):
            ax.text(
                -1.5, state.rows + i * 0.5,
                f"{name}: {status}",
                fontsize=10
            )

        ax.set_title(f"Step: {step_num}")
        plt.tight_layout()

        # Convert figure to image array for GIF
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
        buf.seek(0)
        frame = imageio.imread(buf)
        buf.close()
        plt.close(fig)

        return frame

    # Execute random actions
    for step in range(n_steps):
        # Choose a random action
        action = np.random.randint(0, 3)

        # Take a step
        state, obs, reward, info = CatchEnvironment.step(state, action)
        total_reward += reward.item()

        # Create frame for GIF (all steps)
        frame = create_frame(obs, state, step)
        frames.append(frame)

        # Print info for significant events
        if info["catch_bit"] or info["miss_bit"] or abs(reward) > 0:
            print(f"Step {step}:")
            print(f"  Action: {['left', 'stay', 'right'][action]}")
            print(f"  Reward: {reward.item()}")
            print(f"  Hot: {info['is_hot'].item()}")
            print(f"  Catch: {info['catch_bit'].item()}")
            print(f"  Miss: {info['miss_bit'].item()}")
            print(f"  Plus: {info['plus_bit'].item()}")
            print(f"  Minus: {info['minus_bit'].item()}")
            print(f"  Reward countdown: {info['reward_countdown'].item()}")
            print()

    # Create GIF from all frames
    print(f"Creating GIF from {len(frames)} frames...")
    imageio.mimsave('catch_animation.gif', frames, fps=2)
    print(f"GIF saved as 'catch_animation.gif'")

    print(f"Total reward after {n_steps} steps: {total_reward}")
    
    ##### JIT TESTING #####
    
    print("Testing JIT + scan compilation...")
    
    import time
    from functools import partial

    # Define a jitted step function that processes (state, action) -> (state, outputs)
    @jax.jit
    def jitted_step_fn(state, action):
        return CatchEnvironment.step(state, action)

    def run_steps_with_scan(state, n_steps=1000):
        """Run n_steps using jax.lax.scan and always 'stay' action."""
        actions = jnp.ones(n_steps, dtype=jnp.int32)  # Always stay (action=1)
        def scan_fn(state, action):
            state, _, _, _ = jitted_step_fn(state, action)
            return state, None
        final_state, _ = jax.lax.scan(scan_fn, state, actions)
        return final_state

    # Create a fresh state
    state = CatchEnvironmentState(seed=123)

    # Time normal Python loop (un-jitted, no scan)
    def vanilla_run_steps(state, n_steps=1000):
        for _ in range(n_steps):
            state, _, _, _ = CatchEnvironment.step(state, 1)
        return state

    start = time.time()
    state_vanilla = vanilla_run_steps(state, n_steps=100)
    normal_time = time.time() - start

    # Time using JIT+scan
    jitted_scan_fn = jax.jit(run_steps_with_scan, static_argnums=1)
    # Warmup
    _ = jitted_scan_fn(state, 100)

    start = time.time()
    state_jit = jitted_scan_fn(state, 100)
    jax.block_until_ready(state_jit)
    jitted_time = time.time() - start

    print(f"Normal execution time: {normal_time:.4f}s")
    print(f"Jitted+scan execution time: {jitted_time:.4f}s")
    print(f"Speedup: {normal_time / jitted_time:.2f}x")


if __name__ == "__main__":
    main()