"""
Model Predictive Control with ML Enhancement
Session 3: Advanced control integration

Combines traditional MPC with neural network predictions for
improved performance and adaptation.
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import cvxpy as cp
import numpy as np
from typing import Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
import haiku as hk


@dataclass
class MLMPCConfig:
    """Configuration for ML-enhanced MPC"""
    horizon: int = 20
    dt: float = 1.0
    Q: jnp.ndarray = None  # State cost
    R: jnp.ndarray = None  # Control cost
    state_dim: int = 6
    control_dim: int = 3
    use_ml_predictions: bool = True
    ml_weight: float = 0.3
    control_limits: float = 0.1
    collision_radius: float = 10.0
    

class MLEnhancedMPC:
    """
    Model Predictive Control enhanced with ML predictions.
    
    Combines optimization-based control with learned dynamics models
    for better prediction accuracy and adaptation.
    """
    
    def __init__(
        self,
        config: MLMPCConfig,
        dynamics_model: Optional[Callable] = None,
        ml_predictor: Optional[Callable] = None
    ):
        self.config = config
        self.dynamics_model = dynamics_model or self._default_dynamics
        self.ml_predictor = ml_predictor
        
        # Default cost matrices if not provided
        if config.Q is None:
            self.Q = jnp.eye(config.state_dim) * jnp.array([1, 1, 1, 0.1, 0.1, 0.1])
        else:
            self.Q = config.Q
            
        if config.R is None:
            self.R = jnp.eye(config.control_dim) * 0.01
        else:
            self.R = config.R
            
    def _default_dynamics(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        """
        Default Hill-Clohessy-Wiltshire dynamics.
        
        Args:
            state: Current state [x, y, z, vx, vy, vz]
            control: Control input [ax, ay, az]
            
        Returns:
            next_state: State after dt
        """
        n = 0.001  # Mean motion
        
        # State transition matrix (discrete time)
        A = jnp.array([
            [4-3*jnp.cos(n*self.config.dt), 0, 0, jnp.sin(n*self.config.dt)/n, 2*(1-jnp.cos(n*self.config.dt))/n, 0],
            [6*(jnp.sin(n*self.config.dt)-n*self.config.dt), 1, 0, -2*(1-jnp.cos(n*self.config.dt))/n, (4*jnp.sin(n*self.config.dt)-3*n*self.config.dt)/n, 0],
            [0, 0, jnp.cos(n*self.config.dt), 0, 0, jnp.sin(n*self.config.dt)/n],
            [3*n*jnp.sin(n*self.config.dt), 0, 0, jnp.cos(n*self.config.dt), 2*jnp.sin(n*self.config.dt), 0],
            [-6*n*(1-jnp.cos(n*self.config.dt)), 0, 0, -2*jnp.sin(n*self.config.dt), 4*jnp.cos(n*self.config.dt)-3, 0],
            [0, 0, -n*jnp.sin(n*self.config.dt), 0, 0, jnp.cos(n*self.config.dt)]
        ])
        
        # Control input matrix
        B = jnp.array([
            [(1-jnp.cos(n*self.config.dt))/(n**2), 2*(n*self.config.dt-jnp.sin(n*self.config.dt))/(n**2), 0],
            [2*(n*self.config.dt-jnp.sin(n*self.config.dt))/(n**2), (4*(1-jnp.cos(n*self.config.dt))-3*n**2*self.config.dt**2)/(2*n**2), 0],
            [0, 0, (1-jnp.cos(n*self.config.dt))/(n**2)],
            [jnp.sin(n*self.config.dt)/n, 2*(1-jnp.cos(n*self.config.dt))/n, 0],
            [-2*(1-jnp.cos(n*self.config.dt))/n, (4*jnp.sin(n*self.config.dt)-3*n*self.config.dt)/n, 0],
            [0, 0, jnp.sin(n*self.config.dt)/n]
        ])
        
        return A @ state + B @ control
    
    def solve_mpc(
        self,
        initial_state: jnp.ndarray,
        target_state: jnp.ndarray,
        state_history: Optional[jnp.ndarray] = None,
        obstacles: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Solve MPC problem with optional ML enhancement.
        
        Args:
            initial_state: Current state
            target_state: Target state
            state_history: Historical states for ML prediction
            obstacles: Obstacle positions for collision avoidance
            
        Returns:
            optimal_control: First control action
            info: Solution information
        """
        N = self.config.horizon
        nx = self.config.state_dim
        nu = self.config.control_dim
        
        # Decision variables
        x = cp.Variable((N+1, nx))
        u = cp.Variable((N, nu))
        
        # ML predictions if available
        ml_predictions = None
        if self.ml_predictor is not None and state_history is not None:
            ml_predictions = self.ml_predictor(state_history)
            
        # Objective function
        objective = 0
        for k in range(N):
            # State error
            state_error = x[k] - target_state
            objective += cp.quad_form(state_error, self.Q)
            
            # Control cost
            objective += cp.quad_form(u[k], self.R)
            
            # ML guidance term if available
            if ml_predictions is not None and k < len(ml_predictions):
                ml_error = x[k] - ml_predictions[k]
                objective += self.config.ml_weight * cp.sum_squares(ml_error)
                
        # Terminal cost
        terminal_error = x[N] - target_state
        objective += 10 * cp.quad_form(terminal_error, self.Q)
        
        # Constraints
        constraints = []
        
        # Initial condition
        constraints.append(x[0] == initial_state)
        
        # Dynamics constraints
        for k in range(N):
            # Use numpy arrays for cvxpy
            A_np = np.array(self._get_dynamics_matrix())
            B_np = np.array(self._get_control_matrix())
            constraints.append(x[k+1] == A_np @ x[k] + B_np @ u[k])
            
        # Control limits
        for k in range(N):
            constraints.append(cp.norm(u[k]) <= self.config.control_limits)
            
        # Collision avoidance if obstacles provided
        if obstacles is not None:
            for k in range(N+1):
                for obs in obstacles:
                    constraints.append(
                        cp.norm(x[k][:3] - obs[:3]) >= self.config.collision_radius
                    )
                    
        # Solve problem
        problem = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            problem.solve(solver=cp.OSQP, verbose=False)
            
            if problem.status == 'optimal':
                optimal_states = x.value
                optimal_controls = u.value
                optimal_control = optimal_controls[0]
                
                info = {
                    'status': 'optimal',
                    'cost': problem.value,
                    'states': optimal_states,
                    'controls': optimal_controls,
                    'ml_used': ml_predictions is not None
                }
            else:
                # Fallback to simple proportional control
                error = target_state - initial_state
                optimal_control = np.clip(
                    0.01 * error[:3],
                    -self.config.control_limits,
                    self.config.control_limits
                )
                
                info = {
                    'status': problem.status,
                    'cost': float('inf'),
                    'fallback': True
                }
                
        except Exception as e:
            # Emergency fallback
            optimal_control = np.zeros(nu)
            info = {
                'status': 'error',
                'error': str(e),
                'fallback': True
            }
            
        return jnp.array(optimal_control), info
    
    def _get_dynamics_matrix(self) -> np.ndarray:
        """Get discrete-time dynamics matrix."""
        n = 0.001
        dt = self.config.dt
        
        A = np.array([
            [4-3*np.cos(n*dt), 0, 0, np.sin(n*dt)/n, 2*(1-np.cos(n*dt))/n, 0],
            [6*(np.sin(n*dt)-n*dt), 1, 0, -2*(1-np.cos(n*dt))/n, (4*np.sin(n*dt)-3*n*dt)/n, 0],
            [0, 0, np.cos(n*dt), 0, 0, np.sin(n*dt)/n],
            [3*n*np.sin(n*dt), 0, 0, np.cos(n*dt), 2*np.sin(n*dt), 0],
            [-6*n*(1-np.cos(n*dt)), 0, 0, -2*np.sin(n*dt), 4*np.cos(n*dt)-3, 0],
            [0, 0, -n*np.sin(n*dt), 0, 0, np.cos(n*dt)]
        ])
        
        return A
    
    def _get_control_matrix(self) -> np.ndarray:
        """Get discrete-time control matrix."""
        n = 0.001
        dt = self.config.dt
        
        B = np.array([
            [(1-np.cos(n*dt))/(n**2), 2*(n*dt-np.sin(n*dt))/(n**2), 0],
            [2*(n*dt-np.sin(n*dt))/(n**2), (4*(1-np.cos(n*dt))-3*n**2*dt**2)/(2*n**2), 0],
            [0, 0, (1-np.cos(n*dt))/(n**2)],
            [np.sin(n*dt)/n, 2*(1-np.cos(n*dt))/n, 0],
            [-2*(1-np.cos(n*dt))/n, (4*np.sin(n*dt)-3*n*dt)/n, 0],
            [0, 0, np.sin(n*dt)/n]
        ])
        
        return B


class AdaptiveMPC:
    """
    Adaptive MPC that learns from experience.
    
    Updates its internal model based on prediction errors and
    adjusts control strategies accordingly.
    """
    
    def __init__(
        self,
        config: MLMPCConfig,
        learning_rate: float = 0.01
    ):
        self.config = config
        self.learning_rate = learning_rate
        
        # Initialize adaptive parameters
        self.model_params = {
            'A_correction': jnp.zeros((config.state_dim, config.state_dim)),
            'B_correction': jnp.zeros((config.state_dim, config.control_dim))
        }
        
        # History for learning
        self.prediction_errors = []
        self.state_history = []
        self.control_history = []
        
        # Base MPC
        self.base_mpc = MLEnhancedMPC(config)
        
    def update_model(
        self,
        predicted_state: jnp.ndarray,
        actual_state: jnp.ndarray,
        previous_state: jnp.ndarray,
        control: jnp.ndarray
    ):
        """
        Update model based on prediction error.
        
        Args:
            predicted_state: What the model predicted
            actual_state: What actually happened
            previous_state: State before control
            control: Applied control
        """
        # Compute prediction error
        error = actual_state - predicted_state
        self.prediction_errors.append(error)
        
        # Gradient-based update (simplified)
        # In practice, would use proper system identification
        state_gradient = (actual_state - previous_state) / previous_state.clip(min=1e-6)
        control_gradient = control / jnp.linalg.norm(control).clip(min=1e-6)
        
        # Update corrections
        self.model_params['A_correction'] += (
            self.learning_rate * jnp.outer(error, state_gradient)
        )
        self.model_params['B_correction'] += (
            self.learning_rate * jnp.outer(error, control_gradient)
        )
        
        # Decay old corrections
        self.model_params['A_correction'] *= 0.99
        self.model_params['B_correction'] *= 0.99
        
    def adaptive_dynamics(
        self,
        state: jnp.ndarray,
        control: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Dynamics with learned corrections.
        
        Args:
            state: Current state
            control: Control input
            
        Returns:
            next_state: Predicted next state with corrections
        """
        # Base dynamics
        next_state = self.base_mpc._default_dynamics(state, control)
        
        # Apply learned corrections
        next_state += self.model_params['A_correction'] @ state
        next_state += self.model_params['B_correction'] @ control
        
        return next_state
    
    def solve_adaptive(
        self,
        current_state: jnp.ndarray,
        target_state: jnp.ndarray,
        confidence: float = 1.0
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Solve adaptive MPC with confidence weighting.
        
        Args:
            current_state: Current state
            target_state: Target state
            confidence: Confidence in the adapted model (0-1)
            
        Returns:
            control: Optimal control
            info: Solution information
        """
        # Blend adapted and base dynamics based on confidence
        if confidence > 0.5 and len(self.prediction_errors) > 10:
            # Use adapted model
            self.base_mpc.dynamics_model = self.adaptive_dynamics
        else:
            # Use base model
            self.base_mpc.dynamics_model = self.base_mpc._default_dynamics
            
        # Solve MPC
        control, info = self.base_mpc.solve_mpc(
            current_state,
            target_state,
            state_history=jnp.array(self.state_history) if self.state_history else None
        )
        
        # Store history
        self.state_history.append(current_state)
        self.control_history.append(control)
        
        # Keep history bounded
        if len(self.state_history) > 100:
            self.state_history.pop(0)
            self.control_history.pop(0)
            
        info['adapted'] = confidence > 0.5
        info['model_corrections'] = {
            'A_norm': float(jnp.linalg.norm(self.model_params['A_correction'])),
            'B_norm': float(jnp.linalg.norm(self.model_params['B_correction']))
        }
        
        return control, info


# Utility functions

def create_ml_mpc(
    horizon: int = 20,
    use_ml: bool = True,
    ml_model: Optional[Callable] = None
) -> MLEnhancedMPC:
    """
    Create ML-enhanced MPC controller.
    
    Args:
        horizon: Prediction horizon
        use_ml: Whether to use ML predictions
        ml_model: Optional ML predictor model
        
    Returns:
        controller: ML-enhanced MPC controller
    """
    config = MLMPCConfig(
        horizon=horizon,
        use_ml_predictions=use_ml,
        ml_weight=0.3 if use_ml else 0.0
    )
    
    return MLEnhancedMPC(config, ml_predictor=ml_model)


def create_adaptive_mpc(
    horizon: int = 20,
    learning_rate: float = 0.01
) -> AdaptiveMPC:
    """
    Create adaptive MPC controller.
    
    Args:
        horizon: Prediction horizon
        learning_rate: Model adaptation rate
        
    Returns:
        controller: Adaptive MPC controller
    """
    config = MLMPCConfig(horizon=horizon)
    return AdaptiveMPC(config, learning_rate)


@jit
def evaluate_mpc_performance(
    states: jnp.ndarray,
    controls: jnp.ndarray,
    target: jnp.ndarray
) -> Dict[str, float]:
    """
    Evaluate MPC performance metrics.
    
    Args:
        states: State trajectory
        controls: Control trajectory
        target: Target state
        
    Returns:
        metrics: Performance metrics
    """
    # Tracking error
    tracking_error = jnp.mean(jnp.linalg.norm(states - target, axis=-1))
    
    # Control effort
    control_effort = jnp.sum(jnp.linalg.norm(controls, axis=-1))
    
    # Stability (variance of states)
    stability = jnp.mean(jnp.var(states, axis=0))
    
    # Convergence rate
    final_error = jnp.linalg.norm(states[-1] - target)
    initial_error = jnp.linalg.norm(states[0] - target)
    convergence_rate = 1.0 - (final_error / (initial_error + 1e-6))
    
    return {
        'tracking_error': float(tracking_error),
        'control_effort': float(control_effort),
        'stability': float(stability),
        'convergence_rate': float(convergence_rate),
        'final_error': float(final_error)
    }
