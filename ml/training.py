"""
Training Infrastructure for ML Models
Session 3: Data generation, training loops, and evaluation

Provides:
- Synthetic data generation from physics simulations
- Training loops for supervised and RL models
- Model evaluation and validation
- Transfer learning utilities
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import haiku as hk
import optax
import numpy as np
from typing import Tuple, Dict, Any, Optional, Callable, List
from dataclasses import dataclass
import pickle
import time
from pathlib import Path

# Import from other modules (assuming they exist)
# from sim.dynamics import propagate_orbit_jax
# from control.controllers import FormationLQRController
# from interferometry.noise import NoiseModel


@dataclass
class TrainingConfig:
    """Configuration for model training"""
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    validation_split: float = 0.2
    early_stopping_patience: int = 10
    gradient_clip: float = 1.0
    weight_decay: float = 1e-4
    warmup_steps: int = 1000
    save_frequency: int = 10
    log_frequency: int = 100
    

class DataGenerator:
    """
    Generate synthetic training data from physics simulations.
    """
    
    def __init__(
        self,
        rng: jax.random.PRNGKey,
        num_satellites: int = 2,
        orbital_period: float = 5400.0,  # 90 minutes in seconds
        dt: float = 1.0
    ):
        self.rng = rng
        self.num_satellites = num_satellites
        self.orbital_period = orbital_period
        self.dt = dt
        self.mean_motion = 2 * jnp.pi / orbital_period
        
    def generate_orbit_data(
        self,
        num_trajectories: int = 1000,
        trajectory_length: int = 100
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate orbit trajectory data for supervised learning.
        
        Args:
            num_trajectories: Number of trajectories to generate
            trajectory_length: Length of each trajectory
            
        Returns:
            states: [num_trajectories, trajectory_length, state_dim]
            controls: [num_trajectories, trajectory_length, control_dim]
        """
        states = []
        controls = []
        
        for _ in range(num_trajectories):
            # Random initial conditions
            self.rng, subkey = jax.random.split(self.rng)
            initial_state = jax.random.normal(subkey, (self.num_satellites * 6,)) * 0.1
            
            trajectory_states = [initial_state]
            trajectory_controls = []
            
            state = initial_state
            for t in range(trajectory_length - 1):
                # Simple Hill-Clohessy-Wiltshire dynamics
                state_dot = self._hcw_dynamics(state)
                
                # Random control (or from controller)
                self.rng, subkey = jax.random.split(self.rng)
                control = jax.random.normal(subkey, (self.num_satellites * 3,)) * 1e-3
                
                # Propagate
                state = state + (state_dot + jnp.concatenate([
                    jnp.zeros(self.num_satellites * 3),
                    control
                ])) * self.dt
                
                trajectory_states.append(state)
                trajectory_controls.append(control)
                
            states.append(jnp.stack(trajectory_states))
            controls.append(jnp.stack(trajectory_controls + [jnp.zeros_like(control)]))
            
        return jnp.stack(states), jnp.stack(controls)
    
    def generate_telemetry_data(
        self,
        num_samples: int = 10000,
        telemetry_dim: int = 20,
        anomaly_rate: float = 0.05
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Generate telemetry data with anomalies.
        
        Args:
            num_samples: Number of telemetry samples
            telemetry_dim: Dimension of telemetry vector
            anomaly_rate: Fraction of anomalous samples
            
        Returns:
            telemetry: [num_samples, telemetry_dim]
            labels: [num_samples] Binary anomaly labels
        """
        telemetry = []
        labels = []
        
        num_anomalies = int(num_samples * anomaly_rate)
        
        # Generate normal data
        self.rng, subkey = jax.random.split(self.rng)
        normal_data = jax.random.normal(subkey, (num_samples - num_anomalies, telemetry_dim))
        
        # Add systematic patterns to normal data
        t = jnp.linspace(0, 10 * jnp.pi, num_samples - num_anomalies)
        for i in range(telemetry_dim):
            freq = 0.1 * (i + 1)
            normal_data = normal_data.at[:, i].add(
                jnp.sin(freq * t) * 0.5
            )
            
        telemetry.append(normal_data)
        labels.append(jnp.zeros(num_samples - num_anomalies))
        
        # Generate anomalous data
        self.rng, subkey = jax.random.split(self.rng)
        anomaly_data = jax.random.normal(subkey, (num_anomalies, telemetry_dim))
        
        # Make anomalies more extreme
        anomaly_data = anomaly_data * 3.0 + jax.random.choice(
            subkey, jnp.array([-5.0, 5.0]), (num_anomalies, 1)
        )
        
        telemetry.append(anomaly_data)
        labels.append(jnp.ones(num_anomalies))
        
        # Combine and shuffle
        telemetry = jnp.concatenate(telemetry)
        labels = jnp.concatenate(labels)
        
        self.rng, subkey = jax.random.split(self.rng)
        perm = jax.random.permutation(subkey, num_samples)
        
        return telemetry[perm], labels[perm]
    
    def generate_formation_data(
        self,
        num_samples: int = 5000
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Generate formation optimization data.
        
        Args:
            num_samples: Number of formation configurations
            
        Returns:
            states: Current satellite states
            objectives: Formation objectives
            optimal_controls: Optimal control commands
        """
        states = []
        objectives = []
        controls = []
        
        for _ in range(num_samples):
            # Random satellite states
            self.rng, subkey = jax.random.split(self.rng)
            state = jax.random.normal(
                subkey, (self.num_satellites, 6)
            ) * jnp.array([100, 100, 10, 0.1, 0.1, 0.01])
            
            # Random formation objectives
            self.rng, subkey = jax.random.split(self.rng)
            objective = jax.random.uniform(
                subkey, (5,)
            )  # baseline, inclination, etc.
            
            # Compute "optimal" control (simplified)
            target_separation = objective[0] * 200  # km
            current_separation = jnp.linalg.norm(
                state[1, :3] - state[0, :3]
            )
            
            control = jnp.zeros((self.num_satellites, 3))
            if self.num_satellites > 1:
                direction = (state[1, :3] - state[0, :3]) / (current_separation + 1e-6)
                error = target_separation - current_separation
                control = control.at[0, :].set(-direction * error * 1e-5)
                control = control.at[1, :].set(direction * error * 1e-5)
                
            states.append(state)
            objectives.append(objective)
            controls.append(control)
            
        return (
            jnp.stack(states),
            jnp.stack(objectives),
            jnp.stack(controls)
        )
    
    def _hcw_dynamics(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Hill-Clohessy-Wiltshire relative dynamics.
        
        Args:
            state: Combined state vector [position, velocity]
            
        Returns:
            state_dot: State derivative
        """
        n = self.mean_motion
        num_sats = self.num_satellites
        
        state_dot = jnp.zeros_like(state)
        
        for i in range(num_sats):
            idx = i * 6
            x, y, z = state[idx:idx+3]
            vx, vy, vz = state[idx+3:idx+6]
            
            # HCW equations
            ax = 3 * n**2 * x + 2 * n * vy
            ay = -2 * n * vx
            az = -n**2 * z
            
            state_dot = state_dot.at[idx:idx+3].set(jnp.array([vx, vy, vz]))
            state_dot = state_dot.at[idx+3:idx+6].set(jnp.array([ax, ay, az]))
            
        return state_dot


class SupervisedTrainer:
    """
    Trainer for supervised learning models.
    """
    
    def __init__(
        self,
        model: hk.Transformed,
        config: TrainingConfig = TrainingConfig()
    ):
        self.model = model
        self.config = config
        self.optimizer = self._create_optimizer()
        
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with learning rate schedule."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            decay_steps=10000,
            end_value=self.config.learning_rate * 0.1
        )
        
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.gradient_clip),
            optax.adamw(learning_rate=schedule, weight_decay=self.config.weight_decay)
        )
        
        return optimizer
    
    @partial(jit, static_argnums=(0,))
    def train_step(
        self,
        params: hk.Params,
        opt_state: optax.OptState,
        batch: Dict[str, jnp.ndarray],
        rng: jax.random.PRNGKey
    ) -> Tuple[hk.Params, optax.OptState, Dict[str, float]]:
        """
        Single training step.
        
        Args:
            params: Model parameters
            opt_state: Optimizer state
            batch: Training batch
            rng: Random key
            
        Returns:
            Updated parameters, optimizer state, and metrics
        """
        def loss_fn(params):
            predictions = self.model.apply(
                params, rng, batch['inputs'], is_training=True
            )
            
            # MSE loss for regression
            if 'targets' in batch:
                loss = jnp.mean((predictions - batch['targets']) ** 2)
            # VAE loss for anomaly detection
            elif 'reconstruction' in predictions:
                recon_loss = jnp.mean(
                    (batch['inputs'] - predictions['reconstruction']) ** 2
                )
                kl_loss = -0.5 * jnp.mean(
                    1 + predictions['log_var'] -
                    predictions['mean'] ** 2 -
                    jnp.exp(predictions['log_var'])
                )
                loss = recon_loss + 0.1 * kl_loss
            else:
                loss = jnp.mean(predictions['loss'])
                
            return loss, {'loss': loss}
            
        # Compute gradients
        grads, metrics = grad(loss_fn, has_aux=True)(params)
        
        # Update parameters
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, metrics
    
    def train(
        self,
        train_data: Dict[str, jnp.ndarray],
        val_data: Optional[Dict[str, jnp.ndarray]] = None,
        rng: jax.random.PRNGKey = jax.random.PRNGKey(0)
    ) -> Tuple[hk.Params, Dict[str, List[float]]]:
        """
        Full training loop.
        
        Args:
            train_data: Training data dictionary
            val_data: Optional validation data
            rng: Random key
            
        Returns:
            Trained parameters and training history
        """
        # Initialize parameters
        dummy_input = jax.tree_map(
            lambda x: x[:1], train_data
        )
        params = self.model.init(rng, dummy_input['inputs'])
        opt_state = self.optimizer.init(params)
        
        # Training history
        history = {'train_loss': [], 'val_loss': []}
        
        # Calculate number of batches
        num_samples = len(train_data['inputs'])
        num_batches = num_samples // self.config.batch_size
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Shuffle data
            rng, subkey = jax.random.split(rng)
            perm = jax.random.permutation(subkey, num_samples)
            train_data_shuffled = jax.tree_map(
                lambda x: x[perm], train_data
            )
            
            # Training
            epoch_loss = 0.0
            for batch_idx in range(num_batches):
                start = batch_idx * self.config.batch_size
                end = start + self.config.batch_size
                
                batch = jax.tree_map(
                    lambda x: x[start:end], train_data_shuffled
                )
                
                rng, subkey = jax.random.split(rng)
                params, opt_state, metrics = self.train_step(
                    params, opt_state, batch, subkey
                )
                
                epoch_loss += metrics['loss']
                
            avg_train_loss = epoch_loss / num_batches
            history['train_loss'].append(float(avg_train_loss))
            
            # Validation
            if val_data is not None:
                val_loss = self.evaluate(params, val_data, rng)
                history['val_loss'].append(float(val_loss))
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_params = params
                else:
                    patience_counter += 1
                    
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    return best_params, history
                    
            # Logging
            if epoch % self.config.log_frequency == 0:
                print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}", end="")
                if val_data is not None:
                    print(f", val_loss={val_loss:.4f}")
                else:
                    print()
                    
        return params if val_data is None else best_params, history
    
    @partial(jit, static_argnums=(0,))
    def evaluate(
        self,
        params: hk.Params,
        data: Dict[str, jnp.ndarray],
        rng: jax.random.PRNGKey
    ) -> float:
        """
        Evaluate model on data.
        
        Args:
            params: Model parameters
            data: Evaluation data
            rng: Random key
            
        Returns:
            Average loss
        """
        predictions = self.model.apply(
            params, rng, data['inputs'], is_training=False
        )
        
        if 'targets' in data:
            loss = jnp.mean((predictions - data['targets']) ** 2)
        elif isinstance(predictions, dict) and 'anomaly_score' in predictions:
            # For anomaly detection, use anomaly score
            loss = jnp.mean(predictions['anomaly_score'])
        else:
            loss = jnp.mean(predictions.get('loss', 0.0))
            
        return float(loss)


class ModelCheckpoint:
    """
    Save and load model checkpoints.
    """
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save(
        self,
        params: hk.Params,
        metadata: Dict[str, Any],
        name: str = "checkpoint"
    ):
        """Save model checkpoint."""
        checkpoint = {
            'params': params,
            'metadata': metadata,
            'timestamp': time.time()
        }
        
        filepath = self.checkpoint_dir / f"{name}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(checkpoint, f)
            
        print(f"Saved checkpoint to {filepath}")
        
    def load(self, name: str = "checkpoint") -> Tuple[hk.Params, Dict[str, Any]]:
        """Load model checkpoint."""
        filepath = self.checkpoint_dir / f"{name}.pkl"
        
        with open(filepath, 'rb') as f:
            checkpoint = pickle.load(f)
            
        return checkpoint['params'], checkpoint['metadata']
    
    def save_best(
        self,
        params: hk.Params,
        metric: float,
        metric_name: str = "loss"
    ):
        """Save checkpoint if metric is best so far."""
        best_metric_file = self.checkpoint_dir / f"best_{metric_name}.txt"
        
        should_save = True
        if best_metric_file.exists():
            with open(best_metric_file, 'r') as f:
                best_metric = float(f.read())
                should_save = metric < best_metric
                
        if should_save:
            self.save(
                params,
                {metric_name: metric},
                name=f"best_{metric_name}"
            )
            with open(best_metric_file, 'w') as f:
                f.write(str(metric))


class TransferLearning:
    """
    Transfer learning utilities for mission adaptation.
    """
    
    @staticmethod
    def freeze_layers(
        params: hk.Params,
        frozen_patterns: List[str]
    ) -> Tuple[hk.Params, hk.Params]:
        """
        Freeze specified layers.
        
        Args:
            params: Model parameters
            frozen_patterns: List of parameter name patterns to freeze
            
        Returns:
            frozen_params: Parameters to freeze
            trainable_params: Parameters to train
        """
        frozen_params = {}
        trainable_params = {}
        
        def _split_params(path, param):
            path_str = '/'.join(path)
            is_frozen = any(pattern in path_str for pattern in frozen_patterns)
            
            if is_frozen:
                frozen_params[path_str] = param
            else:
                trainable_params[path_str] = param
                
        jax.tree_map_with_path(_split_params, params)
        
        return frozen_params, trainable_params
    
    @staticmethod
    def adapt_to_new_mission(
        source_params: hk.Params,
        target_data: Dict[str, jnp.ndarray],
        adaptation_steps: int = 100,
        learning_rate: float = 1e-4
    ) -> hk.Params:
        """
        Adapt model to new mission using few-shot learning.
        
        Args:
            source_params: Pre-trained parameters
            target_data: Target mission data
            adaptation_steps: Number of adaptation steps
            learning_rate: Learning rate for adaptation
            
        Returns:
            Adapted parameters
        """
        # Simple few-shot adaptation with lower learning rate
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(source_params)
        
        params = source_params
        rng = jax.random.PRNGKey(0)
        
        for step in range(adaptation_steps):
            # Compute gradients on target data
            def loss_fn(params):
                # Simplified loss computation
                predictions = params  # Would apply model here
                loss = jnp.mean(jnp.sum((predictions - target_data['targets']) ** 2))
                return loss
                
            grads = grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            
        return params


# Example usage functions

def train_orbit_predictor(
    num_trajectories: int = 1000,
    trajectory_length: int = 100
) -> hk.Params:
    """
    Train orbit prediction model.
    
    Args:
        num_trajectories: Number of training trajectories
        trajectory_length: Length of each trajectory
        
    Returns:
        Trained model parameters
    """
    from ml.models import create_orbit_predictor, MLConfig
    
    # Generate data
    rng = jax.random.PRNGKey(42)
    generator = DataGenerator(rng)
    states, controls = generator.generate_orbit_data(
        num_trajectories, trajectory_length
    )
    
    # Prepare data
    train_data = {
        'inputs': states[:, :-10],  # Use all but last 10 steps as input
        'targets': states[:, -10:]  # Predict last 10 steps
    }
    
    # Create model
    config = MLConfig()
    params, apply_fn = create_orbit_predictor(rng, config)
    model = hk.transform(lambda x: apply_fn(params, rng, x))
    
    # Train
    trainer = SupervisedTrainer(model, TrainingConfig(num_epochs=50))
    trained_params, history = trainer.train(train_data, rng=rng)
    
    return trained_params


def train_anomaly_detector(
    num_samples: int = 10000,
    telemetry_dim: int = 20
) -> hk.Params:
    """
    Train anomaly detection model.
    
    Args:
        num_samples: Number of training samples
        telemetry_dim: Dimension of telemetry
        
    Returns:
        Trained model parameters
    """
    from ml.models import create_anomaly_detector, MLConfig
    
    # Generate data
    rng = jax.random.PRNGKey(42)
    generator = DataGenerator(rng)
    telemetry, labels = generator.generate_telemetry_data(
        num_samples, telemetry_dim
    )
    
    # Use only normal data for training (unsupervised)
    normal_mask = labels == 0
    train_data = {
        'inputs': telemetry[normal_mask]
    }
    
    # Create model
    config = MLConfig()
    params, apply_fn = create_anomaly_detector(rng, config, telemetry_dim)
    model = hk.transform(lambda x: apply_fn(params, rng, x))
    
    # Train
    trainer = SupervisedTrainer(model, TrainingConfig(num_epochs=100))
    trained_params, history = trainer.train(train_data, rng=rng)
    
    return trained_params
