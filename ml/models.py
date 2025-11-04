"""
Machine Learning Models for GeoSense Platform
Session 3: ML-Enhanced Satellite Formation Control

This module provides neural network models for:
- Orbit prediction and propagation
- Anomaly detection for satellite health
- Formation optimization
- Sensor fusion and state estimation
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers
import haiku as hk
from typing import Tuple, Dict, Any, Optional, Callable, NamedTuple
import numpy as np
from dataclasses import dataclass

# Type definitions
State = jnp.ndarray
Control = jnp.ndarray
Measurement = jnp.ndarray


@dataclass
class MLConfig:
    """Configuration for ML models"""
    hidden_dims: Tuple[int, ...] = (256, 128, 64)
    learning_rate: float = 1e-3
    dropout_rate: float = 0.1
    sequence_length: int = 100
    prediction_horizon: int = 10
    latent_dim: int = 32
    

class OrbitPredictor(hk.Module):
    """
    Deep neural network for orbit prediction.
    
    Uses LSTM/GRU architecture for temporal dynamics learning
    with physics-informed constraints.
    """
    
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        prediction_horizon: int = 10,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.prediction_horizon = prediction_horizon
        
    def __call__(
        self,
        state_history: jnp.ndarray,
        control_history: Optional[jnp.ndarray] = None,
        is_training: bool = False
    ) -> jnp.ndarray:
        """
        Predict future orbit states from history.
        
        Args:
            state_history: [batch, time, state_dim] Past states
            control_history: [batch, time, control_dim] Past controls
            is_training: Whether in training mode
            
        Returns:
            predictions: [batch, horizon, state_dim] Future state predictions
        """
        batch_size, seq_len, state_dim = state_history.shape
        
        # Concatenate states and controls if provided
        if control_history is not None:
            inputs = jnp.concatenate([state_history, control_history], axis=-1)
        else:
            inputs = state_history
            
        # LSTM encoder
        lstm = hk.LSTM(self.hidden_dims[0])
        initial_state = lstm.initial_state(batch_size)
        outputs, state = hk.dynamic_unroll(
            lstm, inputs, initial_state, time_major=False
        )
        
        # Take last hidden state
        hidden = outputs[:, -1, :]
        
        # Decoder MLP
        for dim in self.hidden_dims[1:]:
            hidden = hk.Linear(dim)(hidden)
            hidden = jax.nn.relu(hidden)
            if is_training:
                hidden = hk.dropout(hk.next_rng_key(), 0.1, hidden)
                
        # Output predictions for multiple timesteps
        predictions = []
        current_hidden = hidden
        
        for _ in range(self.prediction_horizon):
            # Predict next state
            next_state = hk.Linear(state_dim)(current_hidden)
            predictions.append(next_state)
            
            # Update hidden state for autoregressive prediction
            current_hidden = hk.Linear(self.hidden_dims[-1])(
                jnp.concatenate([current_hidden, next_state])
            )
            current_hidden = jax.nn.relu(current_hidden)
            
        return jnp.stack(predictions, axis=1)


class AnomalyDetector(hk.Module):
    """
    Variational Autoencoder for satellite health anomaly detection.
    
    Learns normal operational patterns and detects deviations.
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dims: Tuple[int, ...] = (128, 64),
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
    def encode(
        self,
        telemetry: jnp.ndarray,
        is_training: bool = False
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Encode telemetry to latent distribution.
        
        Args:
            telemetry: [batch, features] Telemetry data
            is_training: Whether in training mode
            
        Returns:
            mean: [batch, latent_dim] Latent mean
            log_var: [batch, latent_dim] Latent log variance
        """
        hidden = telemetry
        
        for dim in self.hidden_dims:
            hidden = hk.Linear(dim)(hidden)
            hidden = jax.nn.relu(hidden)
            if is_training:
                hidden = hk.dropout(hk.next_rng_key(), 0.1, hidden)
                
        mean = hk.Linear(self.latent_dim)(hidden)
        log_var = hk.Linear(self.latent_dim)(hidden)
        
        return mean, log_var
    
    def decode(
        self,
        z: jnp.ndarray,
        output_dim: int,
        is_training: bool = False
    ) -> jnp.ndarray:
        """
        Decode from latent space to reconstruction.
        
        Args:
            z: [batch, latent_dim] Latent codes
            output_dim: Dimension of output
            is_training: Whether in training mode
            
        Returns:
            reconstruction: [batch, output_dim] Reconstructed telemetry
        """
        hidden = z
        
        for dim in reversed(self.hidden_dims):
            hidden = hk.Linear(dim)(hidden)
            hidden = jax.nn.relu(hidden)
            if is_training:
                hidden = hk.dropout(hk.next_rng_key(), 0.1, hidden)
                
        return hk.Linear(output_dim)(hidden)
    
    def __call__(
        self,
        telemetry: jnp.ndarray,
        is_training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass through VAE.
        
        Args:
            telemetry: [batch, features] Input telemetry
            is_training: Whether in training mode
            
        Returns:
            Dictionary with reconstruction, mean, log_var, and anomaly scores
        """
        input_dim = telemetry.shape[-1]
        
        # Encode
        mean, log_var = self.encode(telemetry, is_training)
        
        # Reparameterization trick
        if is_training:
            std = jnp.exp(0.5 * log_var)
            eps = jax.random.normal(hk.next_rng_key(), mean.shape)
            z = mean + std * eps
        else:
            z = mean
            
        # Decode
        reconstruction = self.decode(z, input_dim, is_training)
        
        # Compute anomaly score (reconstruction error + KL divergence)
        recon_error = jnp.mean((telemetry - reconstruction) ** 2, axis=-1)
        kl_div = -0.5 * jnp.sum(
            1 + log_var - mean ** 2 - jnp.exp(log_var), axis=-1
        )
        anomaly_score = recon_error + 0.1 * kl_div
        
        return {
            'reconstruction': reconstruction,
            'mean': mean,
            'log_var': log_var,
            'anomaly_score': anomaly_score,
            'latent': z
        }


class FormationOptimizer(hk.Module):
    """
    Neural network for learning optimal formation configurations.
    
    Uses graph neural networks to model satellite interactions.
    """
    
    def __init__(
        self,
        hidden_dims: Tuple[int, ...] = (128, 64, 32),
        num_satellites: int = 2,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.hidden_dims = hidden_dims
        self.num_satellites = num_satellites
        
    def edge_model(
        self,
        sender_features: jnp.ndarray,
        receiver_features: jnp.ndarray,
        edge_features: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute edge messages between satellites.
        
        Args:
            sender_features: Features of sending satellite
            receiver_features: Features of receiving satellite
            edge_features: Optional edge attributes (e.g., distance)
            
        Returns:
            Edge message
        """
        if edge_features is not None:
            inputs = jnp.concatenate(
                [sender_features, receiver_features, edge_features]
            )
        else:
            inputs = jnp.concatenate([sender_features, receiver_features])
            
        hidden = inputs
        for dim in self.hidden_dims:
            hidden = hk.Linear(dim)(hidden)
            hidden = jax.nn.relu(hidden)
            
        return hidden
    
    def node_model(
        self,
        node_features: jnp.ndarray,
        aggregated_messages: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update node features based on messages.
        
        Args:
            node_features: Current node features
            aggregated_messages: Sum of incoming messages
            
        Returns:
            Updated node features
        """
        inputs = jnp.concatenate([node_features, aggregated_messages])
        
        hidden = inputs
        for dim in self.hidden_dims:
            hidden = hk.Linear(dim)(hidden)
            hidden = jax.nn.relu(hidden)
            
        # Residual connection
        return node_features + hk.Linear(node_features.shape[-1])(hidden)
    
    def __call__(
        self,
        states: jnp.ndarray,
        objectives: jnp.ndarray,
        num_message_passing: int = 3
    ) -> jnp.ndarray:
        """
        Compute optimal control commands for formation.
        
        Args:
            states: [num_satellites, state_dim] Current states
            objectives: [num_objectives] Formation objectives
            num_message_passing: Number of message passing iterations
            
        Returns:
            controls: [num_satellites, control_dim] Optimal controls
        """
        # Initial node features
        node_features = states
        
        # Add objective information to each node
        objectives_broadcast = jnp.tile(
            objectives[None, :], (self.num_satellites, 1)
        )
        node_features = jnp.concatenate(
            [node_features, objectives_broadcast], axis=-1
        )
        
        # Message passing iterations
        for _ in range(num_message_passing):
            messages = []
            
            # Compute all edge messages
            for i in range(self.num_satellites):
                for j in range(self.num_satellites):
                    if i != j:
                        # Compute relative position as edge feature
                        rel_pos = states[j, :3] - states[i, :3]
                        distance = jnp.linalg.norm(rel_pos)
                        edge_feat = jnp.array([distance])
                        
                        msg = self.edge_model(
                            node_features[i],
                            node_features[j],
                            edge_feat
                        )
                        messages.append((j, msg))
            
            # Aggregate messages for each node
            new_features = []
            for i in range(self.num_satellites):
                # Sum incoming messages
                incoming = [m for (j, m) in messages if j == i]
                if incoming:
                    aggregated = jnp.sum(jnp.stack(incoming), axis=0)
                else:
                    aggregated = jnp.zeros_like(node_features[i])
                    
                # Update node
                updated = self.node_model(node_features[i], aggregated)
                new_features.append(updated)
                
            node_features = jnp.stack(new_features)
        
        # Output control commands
        controls = []
        for i in range(self.num_satellites):
            control = hk.Linear(3)(node_features[i])  # 3D control
            controls.append(control)
            
        return jnp.stack(controls)


class NeuralStateEstimator(hk.Module):
    """
    Learned state estimator using attention mechanisms.
    
    Fuses multiple sensor measurements adaptively.
    """
    
    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        name: Optional[str] = None
    ):
        super().__init__(name=name)
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
    def __call__(
        self,
        measurements: Dict[str, jnp.ndarray],
        measurement_covariances: Dict[str, jnp.ndarray],
        prior_state: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Estimate state from multiple measurements.
        
        Args:
            measurements: Dictionary of sensor measurements
            measurement_covariances: Measurement uncertainties
            prior_state: Optional prior state estimate
            
        Returns:
            state_estimate: Estimated state
            uncertainty: Estimation uncertainty
        """
        # Encode each measurement
        encoded_measurements = []
        encoded_uncertainties = []
        
        for sensor_name, measurement in measurements.items():
            # Encode measurement
            encoded = hk.Linear(self.hidden_dim)(measurement)
            encoded = jax.nn.relu(encoded)
            encoded_measurements.append(encoded)
            
            # Encode uncertainty
            cov = measurement_covariances[sensor_name]
            unc_encoded = hk.Linear(self.hidden_dim)(cov.flatten())
            unc_encoded = jax.nn.sigmoid(unc_encoded)  # Normalize to [0,1]
            encoded_uncertainties.append(unc_encoded)
            
        # Stack for attention
        meas_stack = jnp.stack(encoded_measurements)  # [num_sensors, hidden_dim]
        unc_stack = jnp.stack(encoded_uncertainties)
        
        # Multi-head attention to weight measurements
        attention = hk.MultiHeadAttention(
            num_heads=self.num_heads,
            key_size=self.hidden_dim // self.num_heads,
            model_size=self.hidden_dim
        )
        
        # Query is based on uncertainties (attend to most certain)
        query = unc_stack[None, :, :]  # [1, num_sensors, hidden_dim]
        attended = attention(query, meas_stack[None, :, :], meas_stack[None, :, :])
        attended = attended[0]  # Remove batch dimension
        
        # Aggregate attended measurements
        weights = jax.nn.softmax(-jnp.mean(unc_stack, axis=-1))  # Lower uncertainty = higher weight
        weighted_sum = jnp.sum(attended * weights[:, None], axis=0)
        
        # Include prior if available
        if prior_state is not None:
            prior_encoded = hk.Linear(self.hidden_dim)(prior_state)
            weighted_sum = 0.7 * weighted_sum + 0.3 * prior_encoded
            
        # Decode to state estimate
        hidden = weighted_sum
        hidden = hk.Linear(self.hidden_dim)(hidden)
        hidden = jax.nn.relu(hidden)
        
        state_dim = 6  # Position + velocity
        state_estimate = hk.Linear(state_dim)(hidden)
        
        # Estimate uncertainty
        uncertainty_logits = hk.Linear(state_dim)(hidden)
        uncertainty = jax.nn.softplus(uncertainty_logits)
        
        return state_estimate, uncertainty


# Utility functions for model creation and training

def create_orbit_predictor(
    rng: jax.random.PRNGKey,
    config: MLConfig
) -> Tuple[hk.Params, Callable]:
    """
    Create orbit predictor model.
    
    Args:
        rng: Random key
        config: Model configuration
        
    Returns:
        params: Initial parameters
        apply_fn: Model apply function
    """
    def model_fn(state_history, control_history=None, is_training=False):
        predictor = OrbitPredictor(
            hidden_dims=config.hidden_dims,
            prediction_horizon=config.prediction_horizon
        )
        return predictor(state_history, control_history, is_training)
    
    # Transform to pure function
    model = hk.transform(model_fn)
    
    # Initialize with dummy data
    dummy_states = jnp.zeros((1, config.sequence_length, 6))
    dummy_controls = jnp.zeros((1, config.sequence_length, 3))
    params = model.init(rng, dummy_states, dummy_controls)
    
    return params, model.apply


def create_anomaly_detector(
    rng: jax.random.PRNGKey,
    config: MLConfig,
    telemetry_dim: int = 20
) -> Tuple[hk.Params, Callable]:
    """
    Create anomaly detector model.
    
    Args:
        rng: Random key
        config: Model configuration
        telemetry_dim: Dimension of telemetry vector
        
    Returns:
        params: Initial parameters
        apply_fn: Model apply function
    """
    def model_fn(telemetry, is_training=False):
        detector = AnomalyDetector(
            latent_dim=config.latent_dim,
            hidden_dims=config.hidden_dims
        )
        return detector(telemetry, is_training)
    
    model = hk.transform(model_fn)
    
    dummy_telemetry = jnp.zeros((1, telemetry_dim))
    params = model.init(rng, dummy_telemetry)
    
    return params, model.apply


def create_formation_optimizer(
    rng: jax.random.PRNGKey,
    config: MLConfig,
    num_satellites: int = 2
) -> Tuple[hk.Params, Callable]:
    """
    Create formation optimizer model.
    
    Args:
        rng: Random key
        config: Model configuration
        num_satellites: Number of satellites in formation
        
    Returns:
        params: Initial parameters
        apply_fn: Model apply function
    """
    def model_fn(states, objectives, num_message_passing=3):
        optimizer = FormationOptimizer(
            hidden_dims=config.hidden_dims,
            num_satellites=num_satellites
        )
        return optimizer(states, objectives, num_message_passing)
    
    model = hk.transform(model_fn)
    
    dummy_states = jnp.zeros((num_satellites, 6))
    dummy_objectives = jnp.zeros(5)  # 5 objective parameters
    params = model.init(rng, dummy_states, dummy_objectives)
    
    return params, model.apply


def create_neural_estimator(
    rng: jax.random.PRNGKey,
    config: MLConfig
) -> Tuple[hk.Params, Callable]:
    """
    Create neural state estimator model.
    
    Args:
        rng: Random key
        config: Model configuration
        
    Returns:
        params: Initial parameters
        apply_fn: Model apply function
    """
    def model_fn(measurements, covariances, prior_state=None):
        estimator = NeuralStateEstimator(
            hidden_dim=config.hidden_dims[0],
            num_heads=4
        )
        return estimator(measurements, covariances, prior_state)
    
    model = hk.transform(model_fn)
    
    # Dummy data for initialization
    dummy_meas = {
        'gps': jnp.zeros(3),
        'laser': jnp.zeros(1)
    }
    dummy_cov = {
        'gps': jnp.eye(3),
        'laser': jnp.ones((1, 1))
    }
    params = model.init(rng, dummy_meas, dummy_cov)
    
    return params, model.apply
