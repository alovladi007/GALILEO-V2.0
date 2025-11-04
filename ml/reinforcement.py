"""
Reinforcement Learning for Adaptive Formation Control
Session 3: ML-Enhanced Satellite Operations

Implements:
- PPO for formation optimization
- SAC for continuous control
- Multi-agent RL for coordination
- Transfer learning for mission adaptation
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import haiku as hk
import optax
from typing import Tuple, Dict, Any, Optional, NamedTuple, Callable
from dataclasses import dataclass
import numpy as np
from functools import partial


class RLState(NamedTuple):
    """State for RL training"""
    position: jnp.ndarray  # [num_satellites, 3]
    velocity: jnp.ndarray  # [num_satellites, 3]
    time: float
    fuel_used: jnp.ndarray  # [num_satellites]
    

class RLConfig(NamedTuple):
    """Configuration for RL algorithms"""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    learning_rate: float = 3e-4
    buffer_size: int = 10000
    batch_size: int = 256
    update_epochs: int = 10
    

class PPOAgent(hk.Module):
    """
    Proximal Policy Optimization for formation control.
    
    Learns optimal control policies for fuel-efficient formation maintenance.
    """
    
    def __init__(
        self,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 128),
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
    def policy_head(self, features: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Policy network outputting action distribution.
        
        Args:
            features: Hidden features
            
        Returns:
            mean: Action mean
            log_std: Action log standard deviation
        """
        mean = hk.Linear(self.action_dim)(features)
        log_std = hk.get_parameter(
            "log_std",
            shape=[self.action_dim],
            init=hk.initializers.Constant(-0.5)
        )
        return mean, log_std
    
    def value_head(self, features: jnp.ndarray) -> jnp.ndarray:
        """
        Value network for advantage estimation.
        
        Args:
            features: Hidden features
            
        Returns:
            value: State value estimate
        """
        return hk.Linear(1)(features)[..., 0]
    
    def __call__(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through actor-critic network.
        
        Args:
            observation: State observation
            action: Optional action for log probability calculation
            
        Returns:
            Dictionary with policy outputs
        """
        # Shared backbone
        hidden = observation
        for dim in self.hidden_dims:
            hidden = hk.Linear(dim)(hidden)
            hidden = jax.nn.relu(hidden)
            
        # Policy and value outputs
        mean, log_std = self.policy_head(hidden)
        value = self.value_head(hidden)
        
        # Sample action if not provided
        if action is None:
            std = jnp.exp(log_std)
            eps = jax.random.normal(hk.next_rng_key(), mean.shape)
            action = mean + std * eps
            
        # Compute log probability
        std = jnp.exp(log_std)
        log_prob = -0.5 * jnp.sum(
            ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1
        )
        
        # Entropy for exploration
        entropy = 0.5 * jnp.sum(log_std + jnp.log(2 * jnp.pi * jnp.e), axis=-1)
        
        return {
            'action': action,
            'mean': mean,
            'log_std': log_std,
            'log_prob': log_prob,
            'value': value,
            'entropy': entropy
        }


class SACAgent(hk.Module):
    """
    Soft Actor-Critic for continuous control.
    
    Maximum entropy RL for robust formation control.
    """
    
    def __init__(
        self,
        action_dim: int = 3,
        hidden_dims: Tuple[int, ...] = (256, 256),
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
    def actor_network(
        self,
        observation: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Actor network with reparameterization trick.
        
        Args:
            observation: State observation
            
        Returns:
            mean: Action mean
            log_std: Action log standard deviation
        """
        hidden = observation
        for dim in self.hidden_dims:
            hidden = hk.Linear(dim)(hidden)
            hidden = jax.nn.relu(hidden)
            
        mean = hk.Linear(self.action_dim)(hidden)
        log_std = hk.Linear(self.action_dim)(hidden)
        log_std = jnp.clip(log_std, -20, 2)
        
        return mean, log_std
    
    def critic_network(
        self,
        observation: jnp.ndarray,
        action: jnp.ndarray,
        name_suffix: str = ""
    ) -> jnp.ndarray:
        """
        Q-function network.
        
        Args:
            observation: State observation
            action: Action
            name_suffix: Suffix for parameter names (for double Q)
            
        Returns:
            q_value: Q-value estimate
        """
        inputs = jnp.concatenate([observation, action])
        hidden = inputs
        
        for i, dim in enumerate(self.hidden_dims):
            hidden = hk.Linear(dim, name=f"q_linear_{i}{name_suffix}")(hidden)
            hidden = jax.nn.relu(hidden)
            
        return hk.Linear(1, name=f"q_output{name_suffix}")(hidden)[..., 0]
    
    def __call__(
        self,
        observation: jnp.ndarray,
        action: Optional[jnp.ndarray] = None,
        deterministic: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through SAC networks.
        
        Args:
            observation: State observation
            action: Optional action for Q-value calculation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with SAC outputs
        """
        # Actor outputs
        mean, log_std = self.actor_network(observation)
        
        if deterministic:
            action = mean
            log_prob = jnp.zeros_like(mean[..., 0])
        else:
            # Reparameterization trick
            std = jnp.exp(log_std)
            eps = jax.random.normal(hk.next_rng_key(), mean.shape)
            action = mean + std * eps
            
            # Tanh squashing
            action = jnp.tanh(action)
            
            # Compute log probability with correction for tanh
            gaussian_log_prob = -0.5 * jnp.sum(
                ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
                axis=-1
            )
            log_prob = gaussian_log_prob - jnp.sum(
                jnp.log(1 - action ** 2 + 1e-6), axis=-1
            )
            
        # Double Q-networks for stability
        q1 = self.critic_network(observation, action, "_1")
        q2 = self.critic_network(observation, action, "_2")
        
        return {
            'action': action,
            'mean': mean,
            'log_std': log_std,
            'log_prob': log_prob,
            'q1': q1,
            'q2': q2,
            'q_min': jnp.minimum(q1, q2)
        }


class MultiAgentCoordinator(hk.Module):
    """
    Multi-agent reinforcement learning for satellite coordination.
    
    Uses communication and centralized training with decentralized execution.
    """
    
    def __init__(
        self,
        num_agents: int = 2,
        action_dim: int = 3,
        hidden_dim: int = 128,
        communication_rounds: int = 3,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.num_agents = num_agents
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.communication_rounds = communication_rounds
        
    def agent_encoder(
        self,
        observation: jnp.ndarray,
        agent_id: int
    ) -> jnp.ndarray:
        """
        Encode individual agent observation.
        
        Args:
            observation: Agent's local observation
            agent_id: Agent identifier
            
        Returns:
            encoding: Agent encoding
        """
        # Add agent ID as one-hot
        agent_one_hot = jax.nn.one_hot(agent_id, self.num_agents)
        inputs = jnp.concatenate([observation, agent_one_hot])
        
        hidden = hk.Linear(self.hidden_dim)(inputs)
        hidden = jax.nn.relu(hidden)
        hidden = hk.Linear(self.hidden_dim)(hidden)
        
        return hidden
    
    def communication_module(
        self,
        agent_encodings: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Inter-agent communication via attention.
        
        Args:
            agent_encodings: [num_agents, hidden_dim] Agent encodings
            
        Returns:
            updated_encodings: [num_agents, hidden_dim] After communication
        """
        for round_idx in range(self.communication_rounds):
            # Self-attention for communication
            attention = hk.MultiHeadAttention(
                num_heads=4,
                key_size=self.hidden_dim // 4,
                model_size=self.hidden_dim,
                name=f"comm_attention_{round_idx}"
            )
            
            # Add batch dimension for attention
            encodings_batched = agent_encodings[None, :, :]
            attended = attention(
                encodings_batched,
                encodings_batched,
                encodings_batched
            )[0]
            
            # Residual connection
            agent_encodings = agent_encodings + 0.5 * attended
            
        return agent_encodings
    
    def action_decoder(
        self,
        encoding: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Decode action from agent encoding.
        
        Args:
            encoding: Agent's encoding after communication
            
        Returns:
            action_mean: Mean of action distribution
            action_log_std: Log standard deviation of action distribution
        """
        hidden = hk.Linear(self.hidden_dim)(encoding)
        hidden = jax.nn.relu(hidden)
        
        action_mean = hk.Linear(self.action_dim)(hidden)
        action_log_std = hk.Linear(self.action_dim)(hidden)
        action_log_std = jnp.clip(action_log_std, -2, 0.5)
        
        return action_mean, action_log_std
    
    def __call__(
        self,
        observations: jnp.ndarray,
        deterministic: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Coordinated action selection for all agents.
        
        Args:
            observations: [num_agents, obs_dim] All agent observations
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with actions and auxiliary outputs
        """
        # Encode all agents
        encodings = []
        for i in range(self.num_agents):
            enc = self.agent_encoder(observations[i], i)
            encodings.append(enc)
        agent_encodings = jnp.stack(encodings)
        
        # Communication phase
        communicated_encodings = self.communication_module(agent_encodings)
        
        # Decode actions for each agent
        actions = []
        log_probs = []
        
        for i in range(self.num_agents):
            mean, log_std = self.action_decoder(communicated_encodings[i])
            
            if deterministic:
                action = mean
                log_prob = 0.0
            else:
                std = jnp.exp(log_std)
                eps = jax.random.normal(hk.next_rng_key(), mean.shape)
                action = mean + std * eps
                
                log_prob = -0.5 * jnp.sum(
                    ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi)
                )
                
            actions.append(action)
            log_probs.append(log_prob)
            
        return {
            'actions': jnp.stack(actions),
            'log_probs': jnp.array(log_probs),
            'encodings': communicated_encodings
        }


# Training utilities

@jit
def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: Reward sequence
        values: Value estimates
        dones: Done flags
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        advantages: Advantage estimates
        returns: Return estimates
    """
    T = len(rewards)
    advantages = jnp.zeros_like(rewards)
    
    # Compute advantages backwards
    last_advantage = 0
    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages = advantages.at[t].set(
            delta + gamma * gae_lambda * (1 - dones[t]) * last_advantage
        )
        last_advantage = advantages[t]
        
    returns = advantages + values
    return advantages, returns


@partial(jit, static_argnums=(4, 5))
def ppo_update_step(
    params: hk.Params,
    opt_state: optax.OptState,
    batch: Dict[str, jnp.ndarray],
    rng: jax.random.PRNGKey,
    apply_fn: Callable,
    optimizer: optax.GradientTransformation,
    config: RLConfig = RLConfig()
) -> Tuple[hk.Params, optax.OptState, Dict[str, float]]:
    """
    Single PPO update step.
    
    Args:
        params: Network parameters
        opt_state: Optimizer state
        batch: Training batch
        rng: Random key
        apply_fn: Network apply function
        optimizer: Optimizer
        config: PPO configuration
        
    Returns:
        Updated parameters, optimizer state, and metrics
    """
    def loss_fn(params):
        # Forward pass
        outputs = apply_fn(params, rng, batch['observations'])
        
        # Policy loss (clipped surrogate objective)
        ratio = jnp.exp(outputs['log_prob'] - batch['old_log_probs'])
        surr1 = ratio * batch['advantages']
        surr2 = jnp.clip(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio) * batch['advantages']
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))
        
        # Value loss
        value_loss = jnp.mean((outputs['value'] - batch['returns']) ** 2)
        
        # Entropy bonus
        entropy = jnp.mean(outputs['entropy'])
        
        # Total loss
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy
        
        metrics = {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': loss
        }
        
        return loss, metrics
    
    # Compute gradients
    grads, metrics = grad(loss_fn, has_aux=True)(params)
    
    # Clip gradients
    grads = optax.clip_by_global_norm(grads, config.max_grad_norm)[0]
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, metrics


def create_rl_trainer(
    agent_type: str = "ppo",
    config: RLConfig = RLConfig(),
    observation_dim: int = 12,
    action_dim: int = 3,
    num_agents: int = 1
) -> Dict[str, Any]:
    """
    Create RL trainer with specified agent type.
    
    Args:
        agent_type: Type of agent ("ppo", "sac", "multi_agent")
        config: RL configuration
        observation_dim: Observation dimension
        action_dim: Action dimension
        num_agents: Number of agents (for multi-agent)
        
    Returns:
        Dictionary with model, params, optimizer, and apply function
    """
    rng = jax.random.PRNGKey(0)
    
    if agent_type == "ppo":
        def model_fn(observation, action=None):
            agent = PPOAgent(action_dim=action_dim)
            return agent(observation, action)
            
        model = hk.transform(model_fn)
        dummy_obs = jnp.zeros((1, observation_dim))
        params = model.init(rng, dummy_obs)
        
    elif agent_type == "sac":
        def model_fn(observation, action=None, deterministic=False):
            agent = SACAgent(action_dim=action_dim)
            return agent(observation, action, deterministic)
            
        model = hk.transform(model_fn)
        dummy_obs = jnp.zeros((1, observation_dim))
        params = model.init(rng, dummy_obs)
        
    elif agent_type == "multi_agent":
        def model_fn(observations, deterministic=False):
            coordinator = MultiAgentCoordinator(
                num_agents=num_agents,
                action_dim=action_dim
            )
            return coordinator(observations, deterministic)
            
        model = hk.transform(model_fn)
        dummy_obs = jnp.zeros((num_agents, observation_dim))
        params = model.init(rng, dummy_obs)
        
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
        
    # Create optimizer
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(params)
    
    return {
        'model': model,
        'params': params,
        'optimizer': optimizer,
        'opt_state': opt_state,
        'apply_fn': model.apply,
        'config': config
    }


# Environment rewards

@jit
def formation_reward(
    state: RLState,
    target_formation: jnp.ndarray,
    control: jnp.ndarray
) -> float:
    """
    Compute reward for formation maintenance.
    
    Args:
        state: Current RL state
        target_formation: Target formation configuration
        control: Applied control
        
    Returns:
        reward: Scalar reward
    """
    # Formation error
    formation_error = jnp.linalg.norm(state.position - target_formation)
    
    # Fuel penalty
    fuel_penalty = jnp.linalg.norm(control) * 0.1
    
    # Stability bonus
    velocity_magnitude = jnp.linalg.norm(state.velocity)
    stability_bonus = jnp.exp(-velocity_magnitude)
    
    # Collision penalty
    min_distance = jnp.min(
        jnp.linalg.norm(
            state.position[None, :] - state.position[:, None],
            axis=-1
        ) + jnp.eye(len(state.position)) * 1e6  # Ignore self-distances
    )
    collision_penalty = jnp.where(min_distance < 10.0, -100.0, 0.0)
    
    reward = (
        -formation_error
        - fuel_penalty
        + stability_bonus
        + collision_penalty
    )
    
    return reward
