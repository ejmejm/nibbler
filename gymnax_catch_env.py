from typing import Optional, Tuple, Dict, Any, List
import jax
import jax.numpy as jnp
from jax import random
from flax import struct
import numpy as np

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvState(environment.EnvState):
    """Environment state containing all dynamic variables."""
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
    time: int  # Time step counter


@struct.dataclass
class EnvParams(environment.EnvParams):
    """Environment parameters containing all static configuration."""
    rows: int = 10
    cols: int = 5
    hot_prob: float = 0.3
    reset_prob: float = 0.2
    reward_indicator_duration_min: int = 1
    reward_indicator_duration_max: int = 3


class Catch(environment.Environment[EnvState, EnvParams]):
    """
    An implementation of the Catch environment with additional features like hot bit,
    reset bit, catch bit, miss bit, plus bit, and minus bit.
    """

    def __init__(self):
        super().__init__()
        self.obs_shape = (56,)  # 10*5 + 6 = 56 (default rows*cols + 6 special bits)
    
    @property
    def default_params(self) -> EnvParams:
        """Default environment parameters."""
        return EnvParams()

    def get_obs(self, state: EnvState, params: EnvParams = None, key: jax.Array = None) -> jax.Array:
        """
        Construct the observation vector based on the current state.
        The observation is a 1D array of rows*cols bits for the board, plus 6 bits for
        hot, reset, catch, miss, plus, and minus bits.
        """
        if params is None:
            params = self.default_params
        
        # Initialize empty board
        board = jnp.zeros((params.rows, params.cols), dtype=jnp.int32)
        
        # Place ball on board if it's not in reset
        valid_ball_pos = ~state.in_reset & (state.ball_row >= 0) & (state.ball_row < params.rows)
        board = board.at[state.ball_row, state.ball_col].set(jnp.where(valid_ball_pos, 1, 0))
        
        # Place paddle on board (always on bottom row)
        board = board.at[params.rows - 1, state.paddle_col].set(1)
        
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
    
    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        """
        Take a step in the environment based on the provided action.
        
        Args:
            key: Random key for JAX operations
            state: Current environment state
            action: 0 (left), 1 (stay), or 2 (right)
            params: Environment parameters
            
        Returns:
            Tuple of (observation, new_state, reward, done, info)
        """
        # Convert action to int if needed
        action = jnp.asarray(action, dtype=jnp.int32)
        
        # Update paddle position based on action
        # 0: left, 1: stay, 2: right
        paddle_col = jnp.clip(
            state.paddle_col + jnp.array([-1, 0, 1])[action],
            0,
            params.cols - 1,
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
            params.reward_indicator_duration_min,
            params.reward_indicator_duration_max + 1
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
        # Check the current plus_bit/minus_bit values (before deactivation)
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
        ball_at_bottom = (ball_row == params.rows - 1) & ~in_reset
        
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
        will_enter_board = random.uniform(subkey1) < params.reset_prob
        
        # When in reset, ball may enter the board with probability reset_prob
        # Ball can only enter when in_reset was True in the previous step (no other dependencies)
        ball_enters = state.in_reset & will_enter_board & (ball_row == -1)
        
        # Set in_reset based on the flow logic (for next step)
        in_reset = jnp.where(should_activate_reset, True, in_reset)
        ball_row = jnp.where(should_activate_reset, jnp.array(-1), ball_row)
        
        # When ball enters, determine if board becomes hot (only set when entering)
        key, subkey2 = random.split(key)
        becomes_hot = random.uniform(subkey2) < params.hot_prob
        is_hot = jnp.where(ball_enters, becomes_hot, is_hot)
        
        # Place ball in top row at random column when entering
        key, subkey3 = random.split(key)
        new_col = random.randint(subkey3, (), 0, params.cols)
        ball_row = jnp.where(ball_enters, jnp.array(0), ball_row)
        ball_col = jnp.where(ball_enters, new_col, ball_col)
        in_reset = jnp.where(ball_enters, False, in_reset)
        
        # Advance ball if not in reset and not at bottom and not entering
        # Ball must be at a valid position (not -1) to advance
        normal_fall = ~in_reset & ~ball_at_bottom & ~ball_enters & (ball_row >= 0) & (ball_row < params.rows - 1)
        ball_row = jnp.where(normal_fall, ball_row + 1, ball_row)
        
        # Construct new state
        new_state = EnvState(
            ball_row=ball_row,
            ball_col=ball_col,
            paddle_col=paddle_col,
            in_reset=in_reset,
            is_hot=is_hot,
            catch_bit=catch_bit,
            miss_bit=miss_bit,
            plus_bit=plus_bit,
            minus_bit=minus_bit,
            reward_countdown=reward_countdown,
            time=state.time + 1,
        )
        
        # Check if terminal
        done = self.is_terminal(new_state, params)
        
        # Get observation
        observation = self.get_obs(new_state, params)
        
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
            "discount": self.discount(new_state, params),
        }
        
        return (
            jax.lax.stop_gradient(observation),
            jax.lax.stop_gradient(new_state),
            jnp.array(reward),
            done,
            info,
        )
    
    def reset_env(
        self, key: jax.Array, params: EnvParams
    ) -> tuple[jax.Array, EnvState]:
        """Reset the environment to an initial state."""
        # Start with ball in reset state
        ball_row = jnp.array(-1)  # -1 represents reset state
        ball_col = jnp.array(0)
        
        # Start with paddle in middle position
        paddle_col = jnp.array(params.cols // 2)
        
        # Initialize all bits as inactive
        in_reset = jnp.array(True)
        is_hot = jnp.array(False)
        catch_bit = jnp.array(False)
        miss_bit = jnp.array(False)
        plus_bit = jnp.array(False)
        minus_bit = jnp.array(False)
        reward_countdown = jnp.array(0)
        
        state = EnvState(
            ball_row=ball_row,
            ball_col=ball_col,
            paddle_col=paddle_col,
            in_reset=in_reset,
            is_hot=is_hot,
            catch_bit=catch_bit,
            miss_bit=miss_bit,
            plus_bit=plus_bit,
            minus_bit=minus_bit,
            reward_countdown=reward_countdown,
            time=0,
        )
        
        observation = self.get_obs(state, params)
        return observation, state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        """Check whether state is terminal."""
        # Catch environment doesn't have a natural terminal condition
        # It runs indefinitely, so we always return False
        return jnp.array(False)

    @property
    def name(self) -> str:
        """Environment name."""
        return "Catch-v1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        obs_size = params.rows * params.cols + 6
        return spaces.Box(0, 1, (obs_size,), dtype=jnp.int32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """State space of the environment."""
        return spaces.Dict(
            {
                "ball_row": spaces.Box(-1, params.rows, (), jnp.int32),
                "ball_col": spaces.Box(0, params.cols, (), jnp.int32),
                "paddle_col": spaces.Box(0, params.cols, (), jnp.int32),
                "in_reset": spaces.Discrete(2),
                "is_hot": spaces.Discrete(2),
                "catch_bit": spaces.Discrete(2),
                "miss_bit": spaces.Discrete(2),
                "plus_bit": spaces.Discrete(2),
                "minus_bit": spaces.Discrete(2),
                "reward_countdown": spaces.Box(0, params.reward_indicator_duration_max + 1, (), jnp.int32),
                "time": spaces.Discrete(jnp.iinfo(jnp.int32).max),
            }
        )


# Sanity check
def main():
    """Run a simple sanity check on the environment."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend for headless environments
    import matplotlib.pyplot as plt
    import imageio
    from io import BytesIO

    # # Create environment
    # env = CatchEnvironment(seed=42, hot_prob=1.0, reset_prob=1.0)

    # # Reset the environment
    # env, obs = env.reset()

    # # Run a few steps
    # total_reward = 0
    # n_steps = 100

    # # Store frames for GIF
    # frames = []

    # # Create a function to create a frame image
    # def create_frame(obs, env, step_num):
    #     """Create a frame image for the current state."""
    #     board = obs[:env.rows * env.cols].reshape(env.rows, env.cols)
    #     special_bits = obs[env.rows * env.cols:]

    #     fig, ax = plt.subplots(figsize=(5, 10))
    #     ax.imshow(board, cmap='Blues')

    #     # Add text for special bits
    #     special_bit_names = ['Hot', 'Reset', 'Catch', 'Miss', 'Plus', 'Minus']
    #     special_bit_status = ['ON' if bit else 'OFF' for bit in special_bits]

    #     for i, (name, status) in enumerate(zip(special_bit_names, special_bit_status)):
    #         ax.text(
    #             -1.5, env.rows + i * 0.5,
    #             f"{name}: {status}",
    #             fontsize=10
    #         )

    #     ax.set_title(f"Step: {step_num}")
    #     plt.tight_layout()

    #     # Convert figure to image array for GIF
    #     buf = BytesIO()
    #     fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    #     buf.seek(0)
    #     frame = imageio.imread(buf)
    #     buf.close()
    #     plt.close(fig)

    #     return frame

    # # Execute random actions
    # for step in range(n_steps):
    #     # Choose a random action
    #     action = np.random.randint(0, 3)

    #     # Take a step
    #     env, obs, reward, info = env.step(action)
    #     total_reward += reward.item()

    #     # Create frame for GIF (all steps)
    #     frame = create_frame(obs, env, step)
    #     frames.append(frame)

    #     # Print info for significant events
    #     if info["catch_bit"] or info["miss_bit"] or abs(reward) > 0:
    #         print(f"Step {step}:")
    #         print(f"  Action: {['left', 'stay', 'right'][action]}")
    #         print(f"  Reward: {reward.item()}")
    #         print(f"  Hot: {info['is_hot'].item()}")
    #         print(f"  Catch: {info['catch_bit'].item()}")
    #         print(f"  Miss: {info['miss_bit'].item()}")
    #         print(f"  Plus: {info['plus_bit'].item()}")
    #         print(f"  Minus: {info['minus_bit'].item()}")
    #         print(f"  Reward countdown: {info['reward_countdown'].item()}")
    #         print()

    # # Create GIF from all frames
    # print(f"Creating GIF from {len(frames)} frames...")
    # imageio.mimsave('catch_animation.gif', frames, fps=2)
    # print(f"GIF saved as 'catch_animation.gif'")

    # print(f"Total reward after {n_steps} steps: {total_reward}")
    
    # JIT the step function and use jax.lax.scan
    print("Testing JIT + scan compilation...")
    
    import time
    from functools import partial

    # Create environment and get default params
    env = Catch()
    params = env.default_params
    
    # Create a key for random operations
    key = random.PRNGKey(123)
    
    # Reset to get initial state
    key, reset_key = random.split(key)
    obs, state = env.reset_env(reset_key, params)

    # Define a jitted step function that processes (state, action) -> (state, outputs)
    def step_fn(state, action_key):
        action, key = action_key
        obs, state, reward, done, info = env.step_env(key, state, action, params)
        return state, (obs, reward, done, info)

    def run_steps_with_scan(state, key, n_steps=1000):
        """Run n_steps using jax.lax.scan and always 'stay' action."""
        actions = jnp.ones(n_steps, dtype=jnp.int32)  # Always stay (action=1)
        keys = random.split(key, n_steps)
        action_keys = (actions, keys)
        final_state, _ = jax.lax.scan(step_fn, state, action_keys)
        return final_state

    # Time normal Python loop (un-jitted, no scan)
    def vanilla_run_steps(state, key, n_steps=1000):
        for _ in range(n_steps):
            key, step_key = random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, 1, params)
        return state

    start = time.time()
    state_vanilla = vanilla_run_steps(state, key, n_steps=1000)
    normal_time = time.time() - start

    # Time using JIT+scan
    jitted_scan_fn = jax.jit(run_steps_with_scan, static_argnums=2)
    # Warmup
    key, warmup_key = random.split(key)
    warmup_result = jitted_scan_fn(state, warmup_key, 100)
    jax.block_until_ready(warmup_result)

    start = time.time()
    key, test_key = random.split(key)
    state_jit = jitted_scan_fn(state, test_key, 1000)
    jax.block_until_ready(state_jit)
    jitted_time = time.time() - start

    print(f"Normal execution time: {normal_time:.4f}s")
    print(f"Jitted+scan execution time: {jitted_time:.4f}s")
    print(f"Speedup: {normal_time / jitted_time:.2f}x")


if __name__ == "__main__":
    main()