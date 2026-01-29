import argparse
from functools import partial
from typing import Optional, Tuple, Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Int, PyTree
import mlflow
import numpy as np
import optax
from tqdm import tqdm

from catch_env import (
    CatchEnvironmentState, 
    MultiCatchEnvironment,
)
from networks import MLP, QVNetwork
from utils import configure_jax_config, is_float_array, tree_replace


UNROLL_STEPS = 4


class Nibbler(eqx.Module):
    total_feature_count: int = eqx.field(static=True)
    n_gvfs: int = eqx.field(static=True)
    inputs_per_gvf: int = eqx.field(static=True)
    hidden_dim_per_gvf: int = eqx.field(static=True)
    obs_dim: int = eqx.field(static=True)
    n_actions: int = eqx.field(static=True)
    input_replace_threshold: float = eqx.field(static=True)
    cumulant_replace_threshold: float = eqx.field(static=True)
    
    output_layer: QVNetwork
    gvf_networks: QVNetwork # vampped for n_gvfs
    reward_predictor: MLP
    linear_gvf_predictors: MLP # vampped for n_gvfs
    gvf_input_feature_idxs: Int[Array, 'n_gvfs inputs_per_gvf']
    gvf_cumulant_feature_idxs: Int[Array, 'n_gvfs']
    rng: random.PRNGKey
    
    def __init__(
        self,
        n_actions: int,
        obs_dim: int,
        hidden_dim_per_gvf: int,
        inputs_per_gvf: int, # g
        n_gvfs: int, # h
        input_replace_threshold: float = 0.0, # tau_K
        cumulant_replace_threshold: float = 0.0, # tau_I
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
            input_replace_threshold: Utility difference threshold for replacing GVF inputs
            cumulant_replace_threshold: Utility difference threshold for replacing GVF cumulants
            key: Random key for initialization
        """
        self.rng, output_key, reward_predictor_key, gvf_key, linear_gvf_value_key = random.split(key, 5)
        
        self.obs_dim = obs_dim # m
        self.n_actions = n_actions
        self.inputs_per_gvf = inputs_per_gvf
        self.n_gvfs = n_gvfs
        self.hidden_dim_per_gvf = hidden_dim_per_gvf
        self.total_feature_count = obs_dim + hidden_dim_per_gvf * n_gvfs
        self.input_replace_threshold = input_replace_threshold
        self.cumulant_replace_threshold = cumulant_replace_threshold
        
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
        
        # Linear predictors of the GVF value functions, used to determining utility of base inputs for each GVF
        linear_gvf_predictor_keys = random.split(linear_gvf_value_key, n_gvfs)
        self.linear_gvf_predictors = jax.vmap(
            partial(
                MLP,
                input_dim = obs_dim,
                output_dim = 1,
                use_bias = False,
            ),
            in_axes = 0,
        )(key=linear_gvf_predictor_keys)
        
        # GVF QV networks: shared input layer (obs_dim -> hidden_dim_per_gvf) + linear Q/V
        self.gvf_networks = self._make_gvf_networks(gvf_key)
        
        self.gvf_cumulant_feature_idxs = jax.random.choice(
            self.rng, obs_dim, (n_gvfs,), replace=False)
        self.gvf_input_feature_idxs = jax.vmap(
            lambda key: jax.random.choice(key, obs_dim, (inputs_per_gvf,), replace=False)
        )(jax.random.split(self.rng, n_gvfs))
    
    def _make_gvf_networks(self, key: random.PRNGKey) -> QVNetwork:
        keys = random.split(key, self.n_gvfs)
        gvf_networks = jax.vmap(
            partial(
                QVNetwork,
                n_actions = self.n_actions,
                input_dim = self.inputs_per_gvf,
                shared_hidden_dims = [self.hidden_dim_per_gvf],
                separate_hidden_dims = [],
                activation = jax.nn.relu,
                use_bias = False,
            ),
            in_axes = 0,
        )(key=keys)
        return gvf_networks
    
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
    
    def _get_updated_gvf_cumulant_feature_idxs_and_reset_mask(self) -> Tuple[Int[Array, 'n_gvfs'], PyTree]:
        """Computes the new cumulant feature idxs and a mask for any features that need to be reset."""
        feature_utilities = jnp.abs(self.reward_predictor.layers[0].weight.squeeze(axis=0))
        new_cumulant_feature_idxs, selection_mask, is_changed, changed_gvf_idx = incremental_top_k(
            selected_indices = self.gvf_cumulant_feature_idxs,
            feature_utilities = feature_utilities,
            num_features = self.obs_dim,
            tau = self.cumulant_replace_threshold,
        )
        
        # The paper mentions resetting w^pos_u, but not w^pos_v or w^pos_q,a. It probably is resetting all three though (and output weights), right?
        # I'm going to assume that this is a bug in the paper and that all of the GVF network's weights and momentum states should be reset.
        # If I'm going to do that then I should also consider resetting the linear GVF value predictors weights when I reset the GVF networks.
        # Though maybe they actually dont reset the other weights because they are linear and should be able to recover.
        # TODO: I should also try it just as the paper writes it when I'm done with the implementation.
        
        # Start with a mask of zeros for the full model
        full_zeros_mask: Nibbler = jax.tree.map(
            lambda x: jnp.zeros_like(x, dtype=jnp.bool_),
            eqx.filter(self, is_float_array),
        )
        
        # Then fill out the GVF weight reset mask
        def make_gvf_weight_reset_mask(zeros: Bool[Array, 'n_gvfs ...']) -> Bool[Array, 'n_gvfs ...']:
            mask = zeros.at[changed_gvf_idx].set(is_changed)
            return mask
        
        gvf_weight_reset_mask = jax.tree.map(make_gvf_weight_reset_mask, full_zeros_mask.gvf_networks)
        
        # Then fill out the output layer weight reset mask
        start_idx = self.obs_dim + changed_gvf_idx * self.hidden_dim_per_gvf
        end_idx = start_idx + self.hidden_dim_per_gvf * is_changed
        # Use dynamic indexing with a boolean mask since start_idx/end_idx are traced
        def make_output_weight_reset_mask(zeros: Bool[Array, 'n_actions n_features']) -> Bool[Array, 'n_actions n_features']:
            n_features = zeros.shape[1]
            feature_indices = jnp.arange(n_features)
            feature_mask = (feature_indices >= start_idx) & (feature_indices < end_idx)
            return jnp.where(feature_mask[None, :], is_changed, zeros)
        
        # Map setting the mask over the state value and action value weights
        output_weight_reset_mask = jax.tree.map(
            make_output_weight_reset_mask,
            full_zeros_mask.output_layer,
        )
        
        full_weight_mask = tree_replace(
            full_zeros_mask,
            gvf_networks = gvf_weight_reset_mask,
            output_layer = output_weight_reset_mask,
        )
        
        return new_cumulant_feature_idxs, full_weight_mask
    
    
    def _get_updated_gvf_input_feature_idxs_and_reset_mask(
        self,
    ) -> Tuple[Int[Array, 'n_gvfs inputs_per_gvf'], Bool[Array, 'n_gvfs hidden_dim_per_gvf inputs_per_gvf']]:
        """Computes the new GVF input feature idxs and a mask for weights in the GVF network input layers that need to be reset."""
        
        @partial(jax.vmap, in_axes=0)
        def update_gvf_input_feature_idxs(
            input_indices: Int[Array, 'inputs_per_gvf'],
            utilities: Float[Array, 'obs_dim']
        ) -> Int[Array, 'inputs_per_gvf']:
            new_input_feature_idxs, selection_mask, is_changed, changed_input_idx = incremental_top_k(
                selected_indices = input_indices,
                feature_utilities = utilities,
                num_features = self.obs_dim,
                tau = self.input_replace_threshold,
            )
            
            # Mask assumes the shape of all GVF networks is the same
            weight_reset_mask = jnp.zeros((self.hidden_dim_per_gvf, self.inputs_per_gvf), dtype=jnp.bool_)
            weight_reset_mask = weight_reset_mask.at[:, changed_input_idx].set(is_changed)
            
            return new_input_feature_idxs, weight_reset_mask
            
        base_feature_utilities = jnp.abs(self.linear_gvf_predictors.layers[0].weight.squeeze(axis=1))
        
        new_gvf_input_feature_idxs, weight_reset_mask = update_gvf_input_feature_idxs(
            self.gvf_input_feature_idxs, base_feature_utilities)
        
        return new_gvf_input_feature_idxs, weight_reset_mask
        
    
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
        
        @eqx.filter_value_and_grad
        def reward_loss_and_grad_fn(reward_predictor: MLP) -> float:
            reward_prediction = reward_predictor(obs)
            reward_loss = jnp.sum((reward - reward_prediction) ** 2)
            return reward_loss
        
        reward_loss, reward_grads = reward_loss_and_grad_fn(self.reward_predictor)
        
        ### Then compute the gradients for the linear GVF value predictors ###
        
        @partial(jax.vmap, in_axes=0)
        @eqx.filter_value_and_grad
        def linear_gvf_predictors_loss_and_grad_fn(linear_gvf_predictor: MLP, cumulant: Float[Array, '']) -> float:
            curr_gvf_value = linear_gvf_predictor(obs)
            next_gvf_value = linear_gvf_predictor(next_obs)
            target_value = jax.lax.stop_gradient(cumulant + gamma * next_gvf_value)
            loss = jnp.sum((target_value - curr_gvf_value) ** 2)
            return loss
        
        gvf_cumulants = self.get_gvf_cumulants(obs)
        linear_gvf_predictor_losses, linear_gvf_predictor_grads = linear_gvf_predictors_loss_and_grad_fn(
            self.linear_gvf_predictors, gvf_cumulants)
        
        
        ### Then compute gradients for the GVF networks ###
        
        gvf_inputs = self.get_gvf_inputs(obs)
        next_gvf_inputs = self.get_gvf_inputs(next_obs)
        
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
        gradients = eqx.filter(self, is_float_array)
        # Zero out in case there are any parameters we missed in the filter that should be frozen
        gradients = jax.tree.map(lambda x: jnp.zeros_like(x), gradients)
        gradients = tree_replace(
            gradients,
            output_layer = output_layer_grads,
            reward_predictor = reward_grads,
            gvf_networks = gvf_grads,
            linear_gvf_predictors = linear_gvf_predictor_grads,
        )
        
        losses = {
            'output': output_layer_loss,
            'reward': reward_loss,
            'gvfs': gvf_losses,
            'linear_gvf_predictors': linear_gvf_predictor_losses,
            'total': total_loss,
        }
        
        return gradients, losses
    
    def with_update(
        self,
        obs: Float[Array, 'obs_dim'],
        action: int,
        reward: float,
        next_obs: Float[Array, 'obs_dim'],
        gamma: float,
        optimizer: optax.GradientTransformation,
        optimizer_state: optax.OptState,
    ) -> Tuple['Nibbler', optax.OptState]:
        # Compute GVF input and cumulant changes
        new_cumulant_feature_idxs, cumulant_weight_reset_masks = self._get_updated_gvf_cumulant_feature_idxs_and_reset_mask()
        new_gvf_input_feature_idxs, input_weight_reset_mask = self._get_updated_gvf_input_feature_idxs_and_reset_mask()
        
        # The cumulant masks are for the whole network, and the input masks are only for the GVF network input layers,
        # so use the cumulant mask as a base to combine the two masks.
        gvf_input_layer_weight_reset_mask = (
            cumulant_weight_reset_masks.gvf_networks.shared_mlp.layers[0].weight |
            input_weight_reset_mask
        )
        weight_reset_masks = eqx.tree_at(
            lambda x: x.gvf_networks.shared_mlp.layers[0].weight,
            cumulant_weight_reset_masks,
            gvf_input_layer_weight_reset_mask,
        )
        
        # Reset GVF weights according to the changes
        new_rng, gvf_key = random.split(self.rng)
        reset_gvf_networks = self._make_gvf_networks(gvf_key)
        
        updated_gvf_networks = masked_weight_replace(
            tree = self.gvf_networks,
            mask_tree = weight_reset_masks.gvf_networks,
            replace_tree = reset_gvf_networks,
        )
        updated_output_layer = masked_weight_replace(
            tree = self.output_layer,
            mask_tree = weight_reset_masks.output_layer,
        )
        
        updated_agent: Nibbler = tree_replace(
            self,
            gvf_cumulant_feature_idxs = new_cumulant_feature_idxs,
            gvf_input_feature_idxs = new_gvf_input_feature_idxs,
            gvf_networks = updated_gvf_networks,
            output_layer = updated_output_layer,
            rng = new_rng,
        )
        
        # Reset momentum states according to the changes
        new_optimizer_state = reset_sgd_momentum_optim_states(
            weight_reset_masks, optimizer_state)
        
        # Compute gradients
        gradients, losses = updated_agent.compute_grads_and_loss(
            obs = obs,
            action = action,
            reward = reward,
            next_obs = next_obs,
            gamma = gamma,
        )
        
        # Apply gradients to the model weights
        updates, new_optimizer_state = optimizer.update(gradients, new_optimizer_state)
        updated_agent = eqx.apply_updates(updated_agent, updates)
        
        return updated_agent, new_optimizer_state, losses


def masked_weight_replace(tree: PyTree, mask_tree: PyTree, replace_tree: PyTree | None = None) -> PyTree:
    """
    Replace elements of `tree` with corresponding values from `replace_tree`
    wherever `mask_tree` is True. If `replace_tree` is None, use zeros_like(tree).
    
    Args:
        tree: PyTree of parameters.
        mask_tree: PyTree of boolean masks (same structure as tree), True means "replace".
        replace_tree: PyTree of replacement values or None (default is zeros_like(tree)).
        
    Returns:
        PyTree with replaced values where mask_tree is True.
    """
    if replace_tree is None:
        return jax.tree.map(
            lambda mask, w: jnp.where(mask, jnp.zeros_like(w), w),
            mask_tree,
            tree,
        )
    else:
        return jax.tree.map(
            lambda mask, r, w: jnp.where(mask, r, w),
            mask_tree,
            replace_tree,
            tree,
        )


def incremental_top_k(
    selected_indices: Int[Array, 'k'], # A_L
    feature_utilities: Float[Array, 'num_features'], # U
    num_features: int, # m
    tau: float,
) -> Tuple[Int[Array, 'k'], Bool[Array, 'num_features'], jnp.bool_, Int[Array, 'k']]:
    assert feature_utilities.shape[0] == num_features, (
        f"Feature utilities must have length num_features ({num_features}), "
        f"but has length {feature_utilities.shape[0]}"
    )
    assert 0 < selected_indices.shape[0] < num_features, (
        f"Selected indices must have length (exclusive) between 0 and num_features ({num_features}), "
        f"but has length {selected_indices.shape[0]}"
    )
    
    # Make a binary vector where 1 indicates features that are selected
    selection_mask = jnp.zeros(num_features, dtype=jnp.bool_)
    selection_mask = selection_mask.at[selected_indices].set(True) # A_M
    
    # Get the min utility feature of all selected features
    # The pattern: `x + (mask * inf)` sets the values of x with a 1 in the mask to inf
    selected_utilities = feature_utilities + (~selection_mask * jnp.inf)
    low_idx = jnp.argmin(selected_utilities)
    
    # Get the max utility feature of all unselected features
    unselected_utilities = feature_utilities + (selection_mask * -jnp.inf)
    high_idx = jnp.argmax(unselected_utilities)
    
    changed = feature_utilities[low_idx] + tau < feature_utilities[high_idx]
    
    selection_mask = selection_mask.at[low_idx].set(~changed)
    selection_mask = selection_mask.at[high_idx].set(changed)
    # Beware, this will fail silently if `selected_indices` does not have `low_idx`
    low_pos_in_selected_indices = jnp.argmax(selected_indices == low_idx)
    pos = jnp.where(changed, low_pos_in_selected_indices, -1)
    selected_indices = jnp.where(
        changed,
        selected_indices.at[low_pos_in_selected_indices].set(high_idx),
        selected_indices,
    )
    
    return selected_indices, selection_mask, changed, pos


def reset_sgd_momentum_optim_states(
    state_reset_masks: PyTree, # Boolean mask with the structure of Nibbler, where 1 is reset
    optimizer_state: Tuple[optax.TraceState, None, None],
) -> optax.OptState:
    """Resets the momentum states of the optimizer based on the weight reset masks."""
    assert isinstance(optimizer_state, Tuple) and len(optimizer_state) == 3, \
        f"An SGD with momentum optimizer state is expected to have three components, but got {len(optimizer_state)}."
    assert isinstance(optimizer_state[0], optax.TraceState), \
        f"The first component of an SGD with momentum optimizer state is expected to be a TraceState, but got {type(optimizer_state[0])}."
    assert isinstance(optimizer_state[1], optax.EmptyState) and isinstance(optimizer_state[2], optax.EmptyState), (
        f"The second and third components of an SGD with momentum optimizer state are expected to be EmptyState, "
        f"but got {optimizer_state[1]} and {optimizer_state[2]}."
    )
    
    new_trace_state = jax.tree.map(
        lambda reset_mask, curr_state: jnp.where(
            reset_mask, jnp.zeros_like(curr_state), curr_state),
        state_reset_masks,
        optimizer_state[0].trace,
    )
    
    new_optimizer_state = eqx.tree_at(
        lambda x: x[0].trace,
        optimizer_state,
        new_trace_state,
    )
    
    return new_optimizer_state


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
    trainable_params = eqx.filter(agent, is_float_array)
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
    
    updated_agent, new_optimizer_state, losses = train_state.agent.with_update(
        obs = obs,
        action = action,
        reward = reward,
        next_obs = next_obs,
        gamma = train_state.gamma,
        optimizer = train_state.optimizer,
        optimizer_state = train_state.optimizer_state,
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


def train_model(
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
        
        
        ### For debugging changing GVF cumulant and input indices ###
        
        # cumulant_feature_idxs = train_state.agent.gvf_cumulant_feature_idxs
        # utilities = jnp.abs(train_state.agent.reward_predictor.layers[0].weight.squeeze(axis=0))
        # # Compute ranks: highest utility = rank 1, lowest = rank N
        # utility_sort_indices = jnp.argsort(-utilities)  # descending order
        # # Create a mapping from feature index to rank
        # rank_map = {int(idx): int(rank) + 1 for rank, idx in enumerate(utility_sort_indices.tolist())}
        # feature_idxs_list = cumulant_feature_idxs.tolist()
        # print("GVF cumulant feature indices:", feature_idxs_list)
        # print("Utility ranks of those indices:", [rank_map.get(int(idx), None) for idx in feature_idxs_list])
        # print("GVF_0 input feature indices:", train_state.agent.gvf_input_feature_idxs[0])
        
        
        # Average metrics
        avg_reward = metrics['reward'].mean()
        avg_output_loss = metrics['losses']['output'].mean()
        avg_reward_loss = metrics['losses']['reward'].mean()
        avg_gvfs_loss = metrics['losses']['gvfs'].mean()
        avg_linear_gvf_predictor_loss = metrics['losses']['linear_gvf_predictors'].mean()
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
            'avg_linear_gvf_predictor_loss': avg_linear_gvf_predictor_loss,
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
    parser.add_argument('--step_size_scaling_factor', type=float, default=0.00141,
                        help='Step size scaling factor, which is divided by sqrt(n_gvfs) (default: 0.001 * sqrt(2))')
    parser.add_argument('--momentum', type=float, default=0.99,
                        help='Momentum coefficient for SGD (default: 0.99)')
    parser.add_argument('--hidden_dims', type=int, nargs='*', default=[256],
                        help='Hidden layer dimensions (default: None, linear network)')
    parser.add_argument('--log_interval', type=int, default=10_000,
                        help='Logging interval in steps (default: 10000)')
    parser.add_argument('--num_steps', type=int, default=10_000_000,
                        help='Total number of training steps (default: 1e6)')
    parser.add_argument('--num_envs', type=int, default=2,
                        help='Total number of environments (default: 2)')
    
    ### Nibbler-specific arguments ###
    
    parser.add_argument('--n_gvfs', type=int, default=4,
                        help='Number of GVFs (default: 10)')
    parser.add_argument('--inputs_per_gvf', type=int, default=82,
                        help='Number of inputs per GVFs (default: 82)')
    parser.add_argument('--hidden_dim_per_gvf', type=int, default=256,
                        help='Number of hidden units per GVFs (default: 256)')
    parser.add_argument('--tau_inputs', type=float, default=0.0,
                        help='Utility difference threshold for replacing GVF inputs (default: 0.0)')
    parser.add_argument('--tau_cumulants', type=float, default=0.0,
                        help='Utility difference threshold for replacing GVF cumulants (default: 0.0)')
    
    args = parser.parse_args()
    main_step_size = args.step_size_scaling_factor / np.sqrt(args.n_gvfs)
    configure_jax_config()
    
    key = (
        jax.random.PRNGKey(args.seed) if args.seed is not None
        else jax.random.PRNGKey(np.random.randint(0, 1_000_000_000))
    )
    seeds = jax.random.randint(key, (args.num_envs + 1,), 0, 1_000_000_000)
    env_seeds, train_state_seed = seeds[:args.num_envs], seeds[args.num_envs]
    
    # Create optimizer
    optimizer = create_optimizer(
        learning_rate = main_step_size,
        momentum = args.momentum,
    )
    
    # Create environment
    env_state = jax.vmap(
        partial(CatchEnvironmentState,
            rows = 10,
            cols = 5,
            hot_prob = min(2.0 / args.num_envs, 1.0),
            reset_prob = 0.2,
            paddle_noise = 0.2,
            reward_delivery_prob = 0.2,
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
        input_replace_threshold = args.tau_inputs,
        cumulant_replace_threshold = args.tau_cumulants,
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
    train_state = train_model(
        train_state = train_state,
        num_steps = args.num_steps,
    )
    
    mlflow.end_run()
    print('Training complete!')


if __name__ == '__main__':
    main()
