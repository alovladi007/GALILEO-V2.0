"""
ML-Enhanced Station-Keeping and Collision Avoidance
Session 3: Advanced safety and maintenance algorithms

Combines traditional methods with ML for improved:
- Fuel efficiency
- Prediction accuracy
- Adaptive behavior
"""

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from typing import Tuple, Optional, Dict, Any, List, Callable
from dataclasses import dataclass
from functools import partial


@dataclass
class StationKeepingConfig:
    """Configuration for station-keeping"""
    deadband_x: float = 5.0  # meters
    deadband_y: float = 5.0
    deadband_z: float = 2.0
    min_impulse: float = 0.001  # m/s
    max_impulse: float = 0.1
    prediction_horizon: int = 100  # orbits
    fuel_budget: float = 5.0  # m/s per year
    use_ml_prediction: bool = True
    

@dataclass
class CollisionConfig:
    """Configuration for collision avoidance"""
    min_separation: float = 10.0  # meters
    warning_distance: float = 50.0
    critical_distance: float = 20.0
    max_avoidance_dv: float = 1.0  # m/s
    probability_threshold: float = 1e-6
    prediction_horizon: int = 10  # orbits
    use_ml_prediction: bool = True
    

class MLStationKeeping:
    """
    Station-keeping with ML-based drift prediction.
    
    Uses neural networks to predict long-term drift and optimize
    maneuver timing for fuel efficiency.
    """
    
    def __init__(
        self,
        config: StationKeepingConfig,
        drift_predictor: Optional[Callable] = None
    ):
        self.config = config
        self.drift_predictor = drift_predictor
        
        # Maneuver history for learning
        self.maneuver_history = []
        self.drift_history = []
        
    def predict_drift(
        self,
        current_state: jnp.ndarray,
        time_horizon: int
    ) -> jnp.ndarray:
        """
        Predict drift over time horizon.
        
        Args:
            current_state: Current satellite state
            time_horizon: Prediction horizon in orbits
            
        Returns:
            drift_trajectory: Predicted drift trajectory
        """
        if self.drift_predictor is not None and self.config.use_ml_prediction:
            # Use ML predictor
            return self.drift_predictor(current_state, time_horizon)
        else:
            # Simple linear drift model
            return self._linear_drift_model(current_state, time_horizon)
            
    def _linear_drift_model(
        self,
        state: jnp.ndarray,
        horizon: int
    ) -> jnp.ndarray:
        """
        Simple linear drift model.
        
        Args:
            state: Current state
            horizon: Time horizon
            
        Returns:
            drift: Predicted drift
        """
        # Simplified atmospheric drag and solar pressure effects
        drift_rate = jnp.array([
            0.001,  # Along-track drift (m/orbit)
            0.0002,  # Cross-track drift
            0.0001   # Radial drift
        ])
        
        drift_trajectory = []
        current = state[:3]
        
        for t in range(horizon):
            current = current + drift_rate * (t + 1)
            drift_trajectory.append(current)
            
        return jnp.stack(drift_trajectory)
    
    def check_deadband(
        self,
        position: jnp.ndarray
    ) -> Tuple[bool, jnp.ndarray]:
        """
        Check if position is within deadband.
        
        Args:
            position: Current position relative to target
            
        Returns:
            in_deadband: Whether position is within deadband
            violation: Deadband violation vector
        """
        deadband = jnp.array([
            self.config.deadband_x,
            self.config.deadband_y,
            self.config.deadband_z
        ])
        
        violation = jnp.abs(position) - deadband
        in_deadband = jnp.all(violation <= 0)
        
        return in_deadband, jnp.maximum(violation, 0)
    
    def plan_maneuver(
        self,
        current_state: jnp.ndarray,
        target_state: jnp.ndarray
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Plan optimal station-keeping maneuver.
        
        Args:
            current_state: Current state [position, velocity]
            target_state: Target state
            
        Returns:
            maneuver: Planned velocity change
            info: Planning information
        """
        position_error = current_state[:3] - target_state[:3]
        velocity_error = current_state[3:6] - target_state[3:6]
        
        # Check deadband
        in_deadband, violation = self.check_deadband(position_error)
        
        if in_deadband:
            # No maneuver needed
            return jnp.zeros(3), {'status': 'in_deadband', 'fuel_used': 0.0}
            
        # Predict future drift
        drift = self.predict_drift(current_state, self.config.prediction_horizon)
        
        # Find optimal maneuver timing using ML predictions
        if self.config.use_ml_prediction and self.drift_predictor is not None:
            maneuver = self._ml_optimal_maneuver(
                current_state, target_state, drift
            )
        else:
            maneuver = self._tangential_maneuver(position_error, velocity_error)
            
        # Apply constraints
        maneuver_magnitude = jnp.linalg.norm(maneuver)
        
        if maneuver_magnitude < self.config.min_impulse:
            # Too small, wait
            maneuver = jnp.zeros(3)
            status = 'below_minimum'
        elif maneuver_magnitude > self.config.max_impulse:
            # Limit magnitude
            maneuver = maneuver * (self.config.max_impulse / maneuver_magnitude)
            status = 'limited'
        else:
            status = 'nominal'
            
        # Store for learning
        self.maneuver_history.append(maneuver)
        self.drift_history.append(position_error)
        
        info = {
            'status': status,
            'fuel_used': float(jnp.linalg.norm(maneuver)),
            'violation': violation,
            'predicted_drift': drift[-1] if len(drift) > 0 else None
        }
        
        return maneuver, info
    
    def _ml_optimal_maneuver(
        self,
        current_state: jnp.ndarray,
        target_state: jnp.ndarray,
        drift_prediction: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute optimal maneuver using ML predictions.
        
        Args:
            current_state: Current state
            target_state: Target state
            drift_prediction: Predicted drift trajectory
            
        Returns:
            maneuver: Optimal velocity change
        """
        # Find when drift will be maximum
        drift_magnitudes = jnp.linalg.norm(drift_prediction, axis=1)
        max_drift_time = jnp.argmax(drift_magnitudes)
        
        # Compute maneuver to counteract maximum drift
        max_drift = drift_prediction[max_drift_time]
        
        # Tangential component (most efficient)
        orbital_velocity = jnp.linalg.norm(current_state[3:6])
        tangential_direction = current_state[3:6] / (orbital_velocity + 1e-6)
        
        # Compute required velocity change
        # For along-track: Δv = (3/2) * n * Δx / orbital_period
        n = 0.001  # Mean motion
        delta_v_along = 1.5 * n * max_drift[0]
        
        # Add corrections for cross-track and radial
        delta_v_cross = 0.5 * n * max_drift[1]
        delta_v_radial = n * max_drift[2]
        
        maneuver = (
            delta_v_along * tangential_direction +
            delta_v_cross * jnp.array([0, 1, 0]) +
            delta_v_radial * jnp.array([0, 0, 1])
        )
        
        return -maneuver  # Negative to counter drift
    
    def _tangential_maneuver(
        self,
        position_error: jnp.ndarray,
        velocity_error: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Compute tangential maneuver (fuel-optimal).
        
        Args:
            position_error: Position error
            velocity_error: Velocity error
            
        Returns:
            maneuver: Tangential velocity change
        """
        # Simplified tangential burn calculation
        n = 0.001  # Mean motion
        
        # Along-track correction (tangential burn)
        delta_v_along = -1.5 * n * position_error[0]
        
        # Cross-track correction (normal burn)
        delta_v_cross = -0.5 * n * position_error[1]
        
        # Radial correction (usually avoided)
        delta_v_radial = -n * position_error[2] * 0.1  # Reduced weight
        
        return jnp.array([delta_v_along, delta_v_cross, delta_v_radial])
    
    def annual_fuel_estimate(self) -> float:
        """
        Estimate annual fuel consumption.
        
        Returns:
            fuel_estimate: Estimated m/s per year
        """
        if not self.maneuver_history:
            # Use theoretical estimate
            return 2.0  # m/s per year typical for LEO
            
        # Based on recent history
        recent_maneuvers = self.maneuver_history[-100:]
        total_dv = sum(jnp.linalg.norm(m) for m in recent_maneuvers)
        
        # Extrapolate to year
        orbits_per_year = 365 * 15  # ~15 orbits/day in LEO
        maneuvers_per_year = orbits_per_year / max(len(recent_maneuvers), 1)
        
        return float(total_dv * maneuvers_per_year)


class MLCollisionAvoidance:
    """
    Collision avoidance with ML-based conjunction assessment.
    
    Uses neural networks to predict collision probability and
    optimize avoidance maneuvers.
    """
    
    def __init__(
        self,
        config: CollisionConfig,
        conjunction_predictor: Optional[Callable] = None
    ):
        self.config = config
        self.conjunction_predictor = conjunction_predictor
        
        # Collision event history
        self.conjunction_history = []
        self.maneuver_history = []
        
    def assess_collision_risk(
        self,
        primary_state: jnp.ndarray,
        secondary_state: jnp.ndarray,
        primary_cov: Optional[jnp.ndarray] = None,
        secondary_cov: Optional[jnp.ndarray] = None
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Assess collision risk between two objects.
        
        Args:
            primary_state: Primary object state
            secondary_state: Secondary object state
            primary_cov: Primary covariance matrix
            secondary_cov: Secondary covariance matrix
            
        Returns:
            probability: Collision probability
            info: Assessment information
        """
        # Relative position and velocity
        rel_position = primary_state[:3] - secondary_state[:3]
        rel_velocity = primary_state[3:6] - secondary_state[3:6]
        
        # Current separation
        current_distance = jnp.linalg.norm(rel_position)
        
        # Time to closest approach (simplified)
        tca = -jnp.dot(rel_position, rel_velocity) / (jnp.linalg.norm(rel_velocity)**2 + 1e-10)
        tca = jnp.maximum(tca, 0)  # Future events only
        
        # Position at closest approach
        rel_pos_tca = rel_position + rel_velocity * tca
        miss_distance = jnp.linalg.norm(rel_pos_tca)
        
        # Collision probability
        if self.conjunction_predictor is not None and self.config.use_ml_prediction:
            # ML-based probability
            features = jnp.concatenate([
                rel_position, rel_velocity,
                jnp.array([miss_distance, tca])
            ])
            probability = float(self.conjunction_predictor(features))
        else:
            # Analytical probability (simplified 2D)
            probability = self._analytical_collision_probability(
                miss_distance, primary_cov, secondary_cov
            )
            
        # Determine risk level
        if miss_distance < self.config.critical_distance:
            risk_level = 'critical'
        elif miss_distance < self.config.warning_distance:
            risk_level = 'warning'
        else:
            risk_level = 'nominal'
            
        info = {
            'miss_distance': float(miss_distance),
            'time_to_closest_approach': float(tca),
            'current_distance': float(current_distance),
            'risk_level': risk_level,
            'relative_velocity': float(jnp.linalg.norm(rel_velocity))
        }
        
        return probability, info
    
    def _analytical_collision_probability(
        self,
        miss_distance: float,
        primary_cov: Optional[jnp.ndarray],
        secondary_cov: Optional[jnp.ndarray]
    ) -> float:
        """
        Analytical collision probability calculation.
        
        Args:
            miss_distance: Miss distance at TCA
            primary_cov: Primary covariance
            secondary_cov: Secondary covariance
            
        Returns:
            probability: Collision probability
        """
        # Combined object radius
        combined_radius = self.config.min_separation
        
        # Simplified 2D probability
        if primary_cov is not None and secondary_cov is not None:
            # Combined covariance
            combined_cov = primary_cov[:2, :2] + secondary_cov[:2, :2]
            sigma = jnp.sqrt(jnp.trace(combined_cov) / 2)
        else:
            # Default uncertainty
            sigma = 10.0  # meters
            
        # Probability using error function approximation
        if miss_distance < combined_radius:
            probability = 1.0
        else:
            z = (miss_distance - combined_radius) / (sigma * jnp.sqrt(2))
            probability = jnp.exp(-z**2)
            
        return float(probability)
    
    def compute_avoidance_maneuver(
        self,
        primary_state: jnp.ndarray,
        secondary_state: jnp.ndarray,
        time_to_conjunction: float
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute optimal collision avoidance maneuver.
        
        Args:
            primary_state: Primary object state
            secondary_state: Secondary object state
            time_to_conjunction: Time until closest approach
            
        Returns:
            maneuver: Avoidance maneuver delta-v
            info: Maneuver information
        """
        # Relative state
        rel_position = primary_state[:3] - secondary_state[:3]
        rel_velocity = primary_state[3:6] - secondary_state[3:6]
        
        # Compute required separation
        required_separation = self.config.min_separation * 2  # Safety factor
        
        # Maneuver strategies
        strategies = []
        
        # 1. Radial separation (move up/down)
        radial_direction = rel_position / (jnp.linalg.norm(rel_position) + 1e-6)
        radial_maneuver = radial_direction * required_separation / time_to_conjunction
        strategies.append(('radial', radial_maneuver))
        
        # 2. Along-track separation (speed up/slow down)
        velocity_direction = primary_state[3:6] / (jnp.linalg.norm(primary_state[3:6]) + 1e-6)
        along_track_maneuver = velocity_direction * required_separation / time_to_conjunction
        strategies.append(('along_track', along_track_maneuver))
        
        # 3. Cross-track separation (move sideways)
        cross_direction = jnp.cross(radial_direction, velocity_direction)
        cross_direction = cross_direction / (jnp.linalg.norm(cross_direction) + 1e-6)
        cross_track_maneuver = cross_direction * required_separation / time_to_conjunction
        strategies.append(('cross_track', cross_track_maneuver))
        
        # 4. ML-optimized maneuver if available
        if self.conjunction_predictor is not None and self.config.use_ml_prediction:
            ml_maneuver = self._ml_optimal_avoidance(
                primary_state, secondary_state, time_to_conjunction
            )
            strategies.append(('ml_optimal', ml_maneuver))
            
        # Select minimum fuel strategy
        best_strategy = None
        min_dv = float('inf')
        
        for name, maneuver in strategies:
            dv = jnp.linalg.norm(maneuver)
            if dv < min_dv and dv <= self.config.max_avoidance_dv:
                min_dv = dv
                best_strategy = (name, maneuver)
                
        if best_strategy is None:
            # Emergency maneuver
            maneuver = radial_direction * self.config.max_avoidance_dv
            strategy_name = 'emergency'
        else:
            strategy_name, maneuver = best_strategy
            
        # Store for learning
        self.conjunction_history.append({
            'relative_state': jnp.concatenate([rel_position, rel_velocity]),
            'time_to_conjunction': time_to_conjunction
        })
        self.maneuver_history.append(maneuver)
        
        info = {
            'strategy': strategy_name,
            'delta_v': float(jnp.linalg.norm(maneuver)),
            'time_to_conjunction': float(time_to_conjunction),
            'strategies_evaluated': len(strategies)
        }
        
        return maneuver, info
    
    def _ml_optimal_avoidance(
        self,
        primary_state: jnp.ndarray,
        secondary_state: jnp.ndarray,
        time_to_conjunction: float
    ) -> jnp.ndarray:
        """
        ML-optimized avoidance maneuver.
        
        Args:
            primary_state: Primary state
            secondary_state: Secondary state
            time_to_conjunction: Time to conjunction
            
        Returns:
            maneuver: Optimal maneuver from ML model
        """
        # This would use a trained neural network
        # For now, return a weighted combination
        rel_pos = primary_state[:3] - secondary_state[:3]
        rel_vel = primary_state[3:6] - secondary_state[3:6]
        
        # Perpendicular to both position and velocity
        avoidance_direction = jnp.cross(rel_pos, rel_vel)
        avoidance_direction = avoidance_direction / (jnp.linalg.norm(avoidance_direction) + 1e-6)
        
        # Magnitude based on urgency
        urgency = jnp.exp(-time_to_conjunction / 100)  # Exponential urgency
        magnitude = self.config.min_separation * urgency / time_to_conjunction
        
        return avoidance_direction * magnitude
    
    def formation_safety_check(
        self,
        formation_states: List[jnp.ndarray]
    ) -> Dict[str, Any]:
        """
        Check safety of entire formation.
        
        Args:
            formation_states: List of satellite states
            
        Returns:
            safety_report: Formation safety assessment
        """
        num_satellites = len(formation_states)
        conjunctions = []
        min_distance = float('inf')
        
        # Check all pairs
        for i in range(num_satellites):
            for j in range(i+1, num_satellites):
                distance = jnp.linalg.norm(
                    formation_states[i][:3] - formation_states[j][:3]
                )
                
                if distance < min_distance:
                    min_distance = distance
                    
                if distance < self.config.warning_distance:
                    prob, info = self.assess_collision_risk(
                        formation_states[i],
                        formation_states[j]
                    )
                    
                    conjunctions.append({
                        'satellites': (i, j),
                        'distance': float(distance),
                        'probability': prob,
                        'risk_level': info['risk_level']
                    })
                    
        return {
            'safe': len(conjunctions) == 0,
            'min_distance': min_distance,
            'conjunctions': conjunctions,
            'num_warnings': sum(1 for c in conjunctions if c['risk_level'] == 'warning'),
            'num_critical': sum(1 for c in conjunctions if c['risk_level'] == 'critical')
        }


# Combined safety controller

class IntegratedSafetyController:
    """
    Integrated controller combining station-keeping and collision avoidance
    with ML enhancements.
    """
    
    def __init__(
        self,
        sk_config: StationKeepingConfig = StationKeepingConfig(),
        ca_config: CollisionConfig = CollisionConfig(),
        ml_models: Optional[Dict[str, Callable]] = None
    ):
        # Initialize sub-controllers
        drift_predictor = ml_models.get('drift_predictor') if ml_models else None
        conjunction_predictor = ml_models.get('conjunction_predictor') if ml_models else None
        
        self.station_keeping = MLStationKeeping(sk_config, drift_predictor)
        self.collision_avoidance = MLCollisionAvoidance(ca_config, conjunction_predictor)
        
    def compute_safe_control(
        self,
        primary_state: jnp.ndarray,
        target_state: jnp.ndarray,
        secondary_states: Optional[List[jnp.ndarray]] = None
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """
        Compute control that maintains station while avoiding collisions.
        
        Args:
            primary_state: Current satellite state
            target_state: Target state for station-keeping
            secondary_states: Other objects to avoid
            
        Returns:
            control: Safe control command
            info: Control information
        """
        info = {}
        
        # Station-keeping maneuver
        sk_maneuver, sk_info = self.station_keeping.plan_maneuver(
            primary_state, target_state
        )
        info['station_keeping'] = sk_info
        
        # Collision avoidance if needed
        ca_maneuver = jnp.zeros(3)
        if secondary_states:
            max_probability = 0
            critical_conjunction = None
            
            for secondary in secondary_states:
                prob, ca_info = self.collision_avoidance.assess_collision_risk(
                    primary_state, secondary
                )
                
                if prob > max_probability:
                    max_probability = prob
                    critical_conjunction = (secondary, ca_info)
                    
            if max_probability > self.collision_avoidance.config.probability_threshold:
                secondary, ca_info = critical_conjunction
                ca_maneuver, maneuver_info = self.collision_avoidance.compute_avoidance_maneuver(
                    primary_state, secondary, ca_info['time_to_closest_approach']
                )
                info['collision_avoidance'] = maneuver_info
                info['collision_probability'] = max_probability
                
        # Combine maneuvers with priority to collision avoidance
        if jnp.linalg.norm(ca_maneuver) > 0:
            # Collision avoidance takes priority
            control = ca_maneuver
            info['control_mode'] = 'collision_avoidance'
        else:
            # Normal station-keeping
            control = sk_maneuver
            info['control_mode'] = 'station_keeping'
            
        info['total_delta_v'] = float(jnp.linalg.norm(control))
        
        return control, info
