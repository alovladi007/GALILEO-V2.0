"""
Collision avoidance system for satellite formations.

Implements collision detection, probability computation, and 
avoidance maneuver planning for safe formation operations.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, List, Any
from dataclasses import dataclass
import numpy as np
from functools import partial


@dataclass
class CollisionEvent:
    """
    Represents a potential collision event.
    """
    time: float                    # Time of closest approach (s)
    miss_distance: float           # Minimum separation (km)
    probability: float             # Collision probability
    sat1_id: int                   # First satellite ID
    sat2_id: int                   # Second satellite ID
    sat1_position: jnp.ndarray    # Position at TCA
    sat2_position: jnp.ndarray    # Position at TCA
    relative_velocity: float       # Relative velocity (km/s)
    
    @property
    def is_critical(self) -> bool:
        """Check if collision risk is critical."""
        return self.probability > 1e-4 or self.miss_distance < 0.001  # 1 meter


class CollisionDetector:
    """
    Detects potential collisions between satellites.
    """
    
    def __init__(
        self,
        safety_radius: float = 0.010,  # 10 meters
        screening_radius: float = 1.0,  # 1 km initial screening
        time_horizon: float = 86400.0   # 1 day ahead
    ):
        """
        Initialize collision detector.
        
        Args:
            safety_radius: Combined radius for collision (km)
            screening_radius: Initial screening distance (km)
            time_horizon: Look-ahead time (s)
        """
        self.safety_radius = safety_radius
        self.screening_radius = screening_radius
        self.time_horizon = time_horizon
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_closest_approach(
        self,
        state1: jnp.ndarray,
        state2: jnp.ndarray,
        dt: float = 1.0
    ) -> Tuple[float, float, jnp.ndarray, jnp.ndarray]:
        """
        Compute time and distance of closest approach.
        
        Args:
            state1: First satellite state [x, y, z, vx, vy, vz]
            state2: Second satellite state
            dt: Time step for search (s)
            
        Returns:
            tca: Time of closest approach
            dca: Distance at closest approach
            pos1_tca: Position of sat1 at TCA
            pos2_tca: Position of sat2 at TCA
        """
        r1 = state1[:3]
        v1 = state1[3:]
        r2 = state2[:3]
        v2 = state2[3:]
        
        # Relative state
        dr = r1 - r2
        dv = v1 - v2
        
        # For linear motion, TCA occurs when d(|dr|²)/dt = 0
        # This gives: tca = -(dr · dv) / |dv|²
        
        dv_norm_sq = jnp.dot(dv, dv)
        
        # Handle parallel motion
        tca = jax.lax.cond(
            dv_norm_sq > 1e-10,
            lambda: -jnp.dot(dr, dv) / dv_norm_sq,
            lambda: 0.0
        )
        
        # Constrain to future times
        tca = jnp.clip(tca, 0, self.time_horizon)
        
        # Positions at TCA
        pos1_tca = r1 + v1 * tca
        pos2_tca = r2 + v2 * tca
        
        # Distance at TCA
        dca = jnp.linalg.norm(pos1_tca - pos2_tca)
        
        return tca, dca, pos1_tca, pos2_tca
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_collision_probability(
        self,
        miss_distance: float,
        position_covariance1: jnp.ndarray,
        position_covariance2: jnp.ndarray,
        combined_radius: Optional[float] = None
    ) -> float:
        """
        Compute collision probability using 2D Pc method.
        
        Args:
            miss_distance: Predicted miss distance (km)
            position_covariance1: Position uncertainty of sat1 (3x3)
            position_covariance2: Position uncertainty of sat2 (3x3)
            combined_radius: Combined object radius (km)
            
        Returns:
            Pc: Collision probability
        """
        if combined_radius is None:
            combined_radius = self.safety_radius
        
        # Combined covariance
        C = position_covariance1[:2, :2] + position_covariance2[:2, :2]  # Use 2D projection
        
        # Simplified 2D probability (assumes circular cross-section)
        # Pc ≈ exp(-d²/(2σ²)) * (R²/σ²) for d >> R
        
        sigma_sq = jnp.trace(C) / 2  # Average variance
        
        # Foster's approximation
        u = miss_distance**2 / (2 * sigma_sq)
        pc = jnp.exp(-u) * (combined_radius**2 / sigma_sq)
        
        return jnp.clip(pc, 0, 1)
    
    def screen_conjunctions(
        self,
        states: jnp.ndarray,
        covariances: Optional[jnp.ndarray] = None
    ) -> List[CollisionEvent]:
        """
        Screen all satellite pairs for potential collisions.
        
        Args:
            states: Satellite states (N x 6)
            covariances: Position covariances (N x 3 x 3)
            
        Returns:
            List of potential collision events
        """
        n_sats = states.shape[0]
        events = []
        
        if covariances is None:
            # Default uncertainty: 10m in each direction
            covariances = jnp.tile(jnp.eye(3) * (0.01**2), (n_sats, 1, 1))
        
        # Check all pairs
        for i in range(n_sats):
            for j in range(i + 1, n_sats):
                # Quick screening based on current separation
                current_sep = jnp.linalg.norm(states[i, :3] - states[j, :3])
                
                if current_sep < self.screening_radius:
                    # Detailed analysis
                    tca, dca, pos1, pos2 = self.compute_closest_approach(
                        states[i], states[j]
                    )
                    
                    if dca < self.screening_radius / 10:  # Refined threshold
                        # Compute collision probability
                        pc = self.compute_collision_probability(
                            dca, covariances[i], covariances[j]
                        )
                        
                        # Relative velocity at TCA
                        v_rel = jnp.linalg.norm(states[i, 3:] - states[j, 3:])
                        
                        event = CollisionEvent(
                            time=tca,
                            miss_distance=dca,
                            probability=pc,
                            sat1_id=i,
                            sat2_id=j,
                            sat1_position=pos1,
                            sat2_position=pos2,
                            relative_velocity=v_rel
                        )
                        
                        events.append(event)
        
        # Sort by probability (highest risk first)
        events.sort(key=lambda e: e.probability, reverse=True)
        
        return events


class CollisionAvoidanceManeuver:
    """
    Plans and executes collision avoidance maneuvers.
    """
    
    def __init__(
        self,
        n: float,
        min_separation: float = 0.050,  # 50 meters
        max_delta_v: float = 0.010      # 10 m/s max per maneuver
    ):
        """
        Initialize collision avoidance maneuver planner.
        
        Args:
            n: Mean motion (rad/s)
            min_separation: Minimum safe separation (km)
            max_delta_v: Maximum ΔV for avoidance (km/s)
        """
        self.n = n
        self.min_separation = min_separation
        self.max_delta_v = max_delta_v
    
    @jax.jit
    def compute_avoidance_burn(
        self,
        state: jnp.ndarray,
        threat_position: jnp.ndarray,
        threat_velocity: jnp.ndarray,
        time_to_collision: float
    ) -> Tuple[jnp.ndarray, float]:
        """
        Compute optimal avoidance burn.
        
        Args:
            state: Own satellite state
            threat_position: Threat position at TCA
            threat_velocity: Threat velocity
            time_to_collision: Time until TCA (s)
            
        Returns:
            delta_v: Velocity change vector (km/s)
            new_miss_distance: Expected miss distance after maneuver
        """
        own_pos = state[:3]
        own_vel = state[3:]
        
        # Compute current trajectory to TCA
        own_pos_tca = own_pos + own_vel * time_to_collision
        
        # Vector from threat to own position at TCA
        separation_vector = own_pos_tca - threat_position
        sep_distance = jnp.linalg.norm(separation_vector)
        
        # If already safe, no maneuver needed
        if sep_distance > self.min_separation:
            return jnp.zeros(3), sep_distance
        
        # Compute perpendicular direction for maximum efficiency
        relative_velocity = own_vel - threat_velocity
        
        # Perpendicular to both separation and relative velocity
        avoidance_direction = jnp.cross(separation_vector, relative_velocity)
        avoidance_direction = avoidance_direction / (jnp.linalg.norm(avoidance_direction) + 1e-10)
        
        # Also consider radial separation
        if jnp.linalg.norm(avoidance_direction) < 0.1:
            # Use radial direction if perpendicular is not well defined
            avoidance_direction = separation_vector / (sep_distance + 1e-10)
        
        # Compute required ΔV
        # Δd ≈ ΔV * time_to_collision
        required_separation_change = self.min_separation - sep_distance
        required_dv_magnitude = required_separation_change / (time_to_collision + 1.0)
        
        # Limit to maximum ΔV
        dv_magnitude = jnp.clip(required_dv_magnitude, 0, self.max_delta_v)
        
        delta_v = avoidance_direction * dv_magnitude
        
        # Estimate new miss distance
        new_pos_tca = own_pos + (own_vel + delta_v) * time_to_collision
        new_miss_distance = jnp.linalg.norm(new_pos_tca - threat_position)
        
        return delta_v, new_miss_distance
    
    @jax.jit
    def compute_optimal_avoidance_time(
        self,
        time_to_collision: float,
        relative_velocity: float,
        current_separation: float
    ) -> float:
        """
        Compute optimal time to perform avoidance maneuver.
        
        Earlier maneuvers are more fuel-efficient but have more uncertainty.
        
        Args:
            time_to_collision: Time until TCA (s)
            relative_velocity: Relative velocity (km/s)
            current_separation: Current separation (km)
            
        Returns:
            Optimal maneuver time (s from now)
        """
        # Balance efficiency vs uncertainty
        # Efficiency: ΔV ∝ 1/time_remaining
        # Uncertainty: σ ∝ sqrt(time_remaining)
        
        # Optimal is typically 1-2 orbits before TCA
        orbital_period = 2 * jnp.pi / self.n
        
        optimal_lead_time = jnp.clip(
            2 * orbital_period,
            3600,  # At least 1 hour
            time_to_collision * 0.5  # At most half the time
        )
        
        optimal_maneuver_time = time_to_collision - optimal_lead_time
        
        return jnp.maximum(0, optimal_maneuver_time)
    
    def plan_coordinated_avoidance(
        self,
        events: List[CollisionEvent],
        states: jnp.ndarray,
        fuel_allocation: Optional[Dict[int, float]] = None
    ) -> Dict[int, jnp.ndarray]:
        """
        Plan coordinated avoidance for multiple satellites.
        
        Args:
            events: List of collision events
            states: Current satellite states
            fuel_allocation: Available fuel per satellite (km/s)
            
        Returns:
            Dictionary mapping satellite ID to planned ΔV
        """
        maneuvers = {}
        
        for event in events:
            if not event.is_critical:
                continue
            
            # Decide which satellite to maneuver (or both)
            if fuel_allocation:
                fuel1 = fuel_allocation.get(event.sat1_id, float('inf'))
                fuel2 = fuel_allocation.get(event.sat2_id, float('inf'))
                
                if fuel1 > fuel2:
                    maneuvering_sat = event.sat1_id
                    threat_pos = event.sat2_position
                    threat_vel = states[event.sat2_id, 3:]
                else:
                    maneuvering_sat = event.sat2_id
                    threat_pos = event.sat1_position
                    threat_vel = states[event.sat1_id, 3:]
            else:
                # Default: maneuver satellite with higher ID
                maneuvering_sat = max(event.sat1_id, event.sat2_id)
                other_sat = min(event.sat1_id, event.sat2_id)
                threat_pos = event.sat1_position if maneuvering_sat == event.sat2_id else event.sat2_position
                threat_vel = states[other_sat, 3:]
            
            # Skip if already has maneuver planned
            if maneuvering_sat in maneuvers:
                continue
            
            # Compute avoidance burn
            delta_v, new_miss = self.compute_avoidance_burn(
                states[maneuvering_sat],
                threat_pos,
                threat_vel,
                event.time
            )
            
            maneuvers[maneuvering_sat] = delta_v
        
        return maneuvers


class FormationSafetyMonitor:
    """
    Monitors formation safety and triggers avoidance when needed.
    """
    
    def __init__(
        self,
        n: float,
        n_satellites: int,
        safety_radius: float = 0.010,
        check_interval: float = 60.0  # Check every minute
    ):
        """
        Initialize safety monitor.
        
        Args:
            n: Mean motion (rad/s)
            n_satellites: Number of satellites
            safety_radius: Minimum safe separation (km)
            check_interval: Time between safety checks (s)
        """
        self.n = n
        self.n_satellites = n_satellites
        self.safety_radius = safety_radius
        self.check_interval = check_interval
        
        self.detector = CollisionDetector(safety_radius)
        self.avoider = CollisionAvoidanceManeuver(n)
        
        # Safety metrics
        self.collision_history = []
        self.maneuver_history = []
    
    def assess_formation_safety(
        self,
        states: jnp.ndarray,
        covariances: Optional[jnp.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Assess overall formation safety.
        
        Args:
            states: Satellite states
            covariances: Position uncertainties
            
        Returns:
            Safety assessment dictionary
        """
        # Find all potential collisions
        events = self.detector.screen_conjunctions(states, covariances)
        
        # Compute safety metrics
        min_separation = float('inf')
        max_collision_prob = 0.0
        n_critical = 0
        
        for i in range(self.n_satellites):
            for j in range(i + 1, self.n_satellites):
                sep = jnp.linalg.norm(states[i, :3] - states[j, :3])
                min_separation = min(min_separation, sep)
        
        for event in events:
            max_collision_prob = max(max_collision_prob, event.probability)
            if event.is_critical:
                n_critical += 1
        
        # Formation dispersion
        positions = states[:, :3]
        centroid = jnp.mean(positions, axis=0)
        dispersion = jnp.mean(jnp.linalg.norm(positions - centroid, axis=1))
        
        assessment = {
            'safe': len(events) == 0 or max_collision_prob < 1e-5,
            'min_separation': min_separation,
            'max_collision_probability': max_collision_prob,
            'n_collision_risks': len(events),
            'n_critical_events': n_critical,
            'formation_dispersion': dispersion,
            'events': events
        }
        
        return assessment
    
    def compute_keepout_zones(
        self,
        reference_state: jnp.ndarray,
        other_states: jnp.ndarray,
        time_horizon: float = 3600.0
    ) -> List[Tuple[jnp.ndarray, float]]:
        """
        Compute keepout zones around other satellites.
        
        Args:
            reference_state: Reference satellite state
            other_states: Other satellites' states
            time_horizon: Prediction horizon (s)
            
        Returns:
            List of (center, radius) for keepout spheres
        """
        keepout_zones = []
        
        for state in other_states:
            # Predict future position
            future_pos = state[:3] + state[3:] * time_horizon
            
            # Keepout radius includes uncertainty growth
            base_radius = self.safety_radius
            uncertainty_growth = 0.001 * jnp.sqrt(time_horizon / 3600)  # 1m/sqrt(hr)
            
            radius = base_radius + uncertainty_growth
            
            keepout_zones.append((future_pos, radius))
        
        return keepout_zones


# Utility functions
@jax.jit
def compute_separation_matrix(states: jnp.ndarray) -> jnp.ndarray:
    """
    Compute pairwise separation distances.
    
    Args:
        states: Satellite states (N x 6)
        
    Returns:
        Separation matrix (N x N)
    """
    n = states.shape[0]
    positions = states[:, :3]
    
    # Broadcast subtraction
    diff = positions[:, None, :] - positions[None, :, :]
    distances = jnp.linalg.norm(diff, axis=2)
    
    return distances


@jax.jit
def predict_collision_risk(
    states: jnp.ndarray,
    time_horizon: float,
    dt: float = 60.0
) -> float:
    """
    Predict maximum collision risk over time horizon.
    
    Simple linear propagation for quick assessment.
    
    Args:
        states: Satellite states
        time_horizon: Prediction time (s)
        dt: Time step
        
    Returns:
        Maximum collision risk metric
    """
    max_risk = 0.0
    steps = int(time_horizon / dt)
    
    for step in range(steps):
        t = step * dt
        
        # Linear propagation
        future_positions = states[:, :3] + states[:, 3:] * t
        
        # Minimum separation
        n_sats = states.shape[0]
        for i in range(n_sats):
            for j in range(i + 1, n_sats):
                sep = jnp.linalg.norm(future_positions[i] - future_positions[j])
                risk = jnp.exp(-sep / 0.010)  # Exponential risk function
                max_risk = jnp.maximum(max_risk, risk)
    
    return max_risk


# Example usage and testing
if __name__ == "__main__":
    print("Collision Avoidance System Test")
    print("=" * 50)
    
    # Create formation with potential collision
    states = jnp.array([
        [0.100, 0.000, 0.000, -0.0001, 0.0000, 0.000],  # Sat 1
        [0.000, 0.100, 0.000, 0.0001, -0.0001, 0.000],  # Sat 2
        [0.095, 0.005, 0.000, -0.0001, 0.0000, 0.000],  # Sat 3 (close to Sat 1)
    ])
    
    n = 0.001  # rad/s (LEO)
    
    # Test collision detection
    detector = CollisionDetector(safety_radius=0.010)
    events = detector.screen_conjunctions(states)
    
    print(f"Collision Detection:")
    print(f"  Found {len(events)} potential collisions")
    
    for event in events:
        print(f"\n  Event: Sat {event.sat1_id} - Sat {event.sat2_id}")
        print(f"    TCA: {event.time:.1f} s")
        print(f"    Miss distance: {event.miss_distance*1000:.1f} m")
        print(f"    Probability: {event.probability:.2e}")
        print(f"    Critical: {event.is_critical}")
    
    # Test avoidance maneuver
    if events:
        print(f"\nCollision Avoidance:")
        avoider = CollisionAvoidanceManeuver(n)
        
        event = events[0]  # Most critical
        
        # Plan avoidance
        dv, new_miss = avoider.compute_avoidance_burn(
            states[event.sat1_id],
            event.sat2_position,
            states[event.sat2_id, 3:],
            event.time
        )
        
        print(f"  Required ΔV: {jnp.linalg.norm(dv)*1000:.3f} m/s")
        print(f"  Direction: {dv/jnp.linalg.norm(dv)}")
        print(f"  New miss distance: {new_miss*1000:.1f} m")
    
    # Test formation safety
    print(f"\nFormation Safety Assessment:")
    monitor = FormationSafetyMonitor(n, 3)
    safety = monitor.assess_formation_safety(states)
    
    print(f"  Formation safe: {safety['safe']}")
    print(f"  Minimum separation: {safety['min_separation']*1000:.1f} m")
    print(f"  Max collision prob: {safety['max_collision_probability']:.2e}")
    print(f"  Formation dispersion: {safety['formation_dispersion']*1000:.1f} m")
    
    # Test separation matrix
    sep_matrix = compute_separation_matrix(states)
    print(f"\nSeparation Matrix (m):")
    print(sep_matrix * 1000)
