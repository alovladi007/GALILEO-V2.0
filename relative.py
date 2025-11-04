"""
Relative orbital dynamics module.

Implements equations for relative motion between satellites in formation:
- Hill/Clohessy-Wiltshire (CW) equations (linearized relative dynamics)
- Nonlinear relative dynamics
- Coordinate transformations between inertial and Hill (LVLH) frames

The Hill frame (also called LVLH - Local Vertical Local Horizontal) is a
moving reference frame centered on the leader satellite:
- x-axis: radial direction (away from Earth)
- y-axis: along-track direction (velocity direction for circular orbits)
- z-axis: cross-track direction (normal to orbital plane)
"""

from typing import Tuple
import jax
import jax.numpy as jnp
from jax import Array

from sim.dynamics.keplerian import GM_EARTH


def hill_clohessy_wiltshire_dynamics(
    t: float,
    delta_state: Array,
    n: float
) -> Array:
    """
    Hill-Clohessy-Wiltshire (CW) equations for relative orbital dynamics.
    
    The CW equations describe the linearized relative motion between two satellites
    in neighboring circular orbits. They're valid when:
    1. Leader orbit is circular
    2. Relative separation << orbital radius
    3. Follower eccentricity is small
    
    State Variables (in Hill/LVLH frame):
        - δx: radial separation (positive = away from Earth)
        - δy: along-track separation (positive = ahead)
        - δz: cross-track separation (positive = normal to plane)
    
    Equations:
        δẍ - 3n²δx - 2nδẏ = 0
        δÿ + 2nδẋ = 0
        δz̈ + n²δz = 0
    
    where n is the mean motion (orbital angular velocity) of the leader.
    
    Solution Properties:
        - Natural modes: in-plane ellipse + out-of-plane oscillation
        - In-plane motion couples radial and along-track
        - Out-of-plane motion is independent, simple harmonic oscillator
        - Relative orbit is periodic (period = leader's orbital period)
    
    Args:
        t: Time (s) - not used but included for ODE compatibility
        delta_state: Relative state [δx, δy, δz, δvx, δvy, δvz] (km, km/s)
        n: Mean motion of leader orbit (rad/s)
    
    Returns:
        Relative state derivative (km/s, km/s²)
    
    Reference:
        - Clohessy & Wiltshire (1960), "Terminal Guidance System for Satellite Rendezvous"
        - Hill (1878), "Researches in the Lunar Theory"
    
    Example:
        >>> # 1 km radial separation, no relative velocity
        >>> delta_state = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        >>> n = 0.001  # rad/s (~100 min orbit)
        >>> d_delta_state = hill_clohessy_wiltshire_dynamics(0.0, delta_state, n)
    """
    # Extract relative position and velocity
    dx, dy, dz = delta_state[0], delta_state[1], delta_state[2]
    dvx, dvy, dvz = delta_state[3], delta_state[4], delta_state[5]
    
    # Compute accelerations from CW equations
    # Radial: ẍ = 3n²x + 2nẏ
    ddx = 3 * (n ** 2) * dx + 2 * n * dvy
    
    # Along-track: ÿ = -2nẋ
    ddy = -2 * n * dvx
    
    # Cross-track: z̈ = -n²z
    ddz = -(n ** 2) * dz
    
    return jnp.array([dvx, dvy, dvz, ddx, ddy, ddz])


def relative_dynamics_nonlinear(
    t: float,
    combined_state: Array,
    mu: float = GM_EARTH
) -> Array:
    """
    Nonlinear relative orbital dynamics.
    
    Computes the exact (nonlinear) relative motion between two satellites
    without the linearization assumptions of the CW equations. This is more
    accurate for:
    - Large relative separations (> 10% of orbital radius)
    - Eccentric orbits
    - Long propagation times
    
    The relative state is computed in the Hill frame but propagation is done
    in the inertial frame for both satellites, then transformed.
    
    Args:
        t: Time (s)
        combined_state: Combined state [r1, v1, r2, v2] where:
            - r1, v1: Leader satellite position and velocity (km, km/s)
            - r2, v2: Follower satellite position and velocity (km, km/s)
        mu: Gravitational parameter (km³/s²)
    
    Returns:
        Combined state derivative (km/s, km/s²)
    
    Algorithm:
        1. Extract leader and follower states
        2. Compute accelerations in inertial frame for both
        3. Return combined derivatives
    
    Note:
        For formation flying analysis, this can be post-processed to
        compute relative state in Hill frame using inertial_to_hill_frame().
    
    Example:
        >>> # Leader at 7000 km, follower 1 km away
        >>> r1 = jnp.array([7000.0, 0.0, 0.0])
        >>> v1 = jnp.array([0.0, 7.5, 0.0])
        >>> r2 = jnp.array([7001.0, 0.0, 0.0])
        >>> v2 = jnp.array([0.0, 7.5, 0.0])
        >>> combined = jnp.concatenate([r1, v1, r2, v2])
        >>> d_combined = relative_dynamics_nonlinear(0.0, combined)
    """
    # Extract states
    r1 = combined_state[0:3]
    v1 = combined_state[3:6]
    r2 = combined_state[6:9]
    v2 = combined_state[9:12]
    
    # Compute accelerations (two-body)
    r1_mag = jnp.linalg.norm(r1)
    r2_mag = jnp.linalg.norm(r2)
    
    a1 = -mu / (r1_mag ** 3) * r1
    a2 = -mu / (r2_mag ** 3) * r2
    
    # Return combined derivative
    return jnp.concatenate([v1, a1, v2, a2])


def hill_frame_to_inertial(
    r_leader: Array,
    v_leader: Array,
    delta_r_hill: Array,
    delta_v_hill: Array
) -> Tuple[Array, Array]:
    """
    Transform relative state from Hill frame to inertial frame.
    
    Hill Frame (LVLH - Local Vertical Local Horizontal):
        - Origin at leader satellite
        - x-axis: radial (away from Earth)
        - y-axis: along-track (velocity direction for circular orbit)
        - z-axis: cross-track (completes right-handed triad)
    
    Transformation:
        1. Construct Hill frame basis vectors from leader state
        2. Rotate relative position/velocity to inertial frame
        3. Add leader position/velocity
    
    Args:
        r_leader: Leader position in inertial frame (km)
        v_leader: Leader velocity in inertial frame (km/s)
        delta_r_hill: Relative position in Hill frame [δx, δy, δz] (km)
        delta_v_hill: Relative velocity in Hill frame [δvx, δvy, δvz] (km/s)
    
    Returns:
        Tuple of (r_follower, v_follower) in inertial frame (km, km/s)
    
    Example:
        >>> r_leader = jnp.array([7000.0, 0.0, 0.0])
        >>> v_leader = jnp.array([0.0, 7.5, 0.0])
        >>> delta_r = jnp.array([1.0, 0.0, 0.0])  # 1 km radial
        >>> delta_v = jnp.array([0.0, 0.0, 0.0])
        >>> r_follower, v_follower = hill_frame_to_inertial(
        ...     r_leader, v_leader, delta_r, delta_v
        ... )
    """
    # Construct Hill frame basis vectors
    # h = angular momentum vector
    h = jnp.cross(r_leader, v_leader)
    h_mag = jnp.linalg.norm(h)
    
    # z-axis: cross-track (normal to orbital plane)
    z_hill = h / h_mag
    
    # x-axis: radial (toward Earth, but we define positive as away)
    x_hill = r_leader / jnp.linalg.norm(r_leader)
    
    # y-axis: along-track (completes right-handed triad)
    y_hill = jnp.cross(z_hill, x_hill)
    
    # Rotation matrix: Hill -> Inertial
    # Each column is a basis vector of the Hill frame expressed in inertial frame
    R_hill_to_inertial = jnp.column_stack([x_hill, y_hill, z_hill])
    
    # Transform relative position to inertial frame
    delta_r_inertial = R_hill_to_inertial @ delta_r_hill
    
    # Transform relative velocity to inertial frame
    # Need to account for rotation of Hill frame
    # ω = h / r²
    r_leader_mag = jnp.linalg.norm(r_leader)
    omega_hill = h / (r_leader_mag ** 2)
    
    # Velocity transformation includes Coriolis term
    delta_v_inertial = R_hill_to_inertial @ delta_v_hill + jnp.cross(omega_hill, delta_r_inertial)
    
    # Compute follower state in inertial frame
    r_follower = r_leader + delta_r_inertial
    v_follower = v_leader + delta_v_inertial
    
    return r_follower, v_follower


def inertial_to_hill_frame(
    r_leader: Array,
    v_leader: Array,
    r_follower: Array,
    v_follower: Array
) -> Tuple[Array, Array]:
    """
    Transform follower state from inertial frame to Hill frame relative to leader.
    
    This is the inverse transformation of hill_frame_to_inertial(). It's useful
    for analyzing formation flying dynamics and computing relative states for
    control algorithms.
    
    Args:
        r_leader: Leader position in inertial frame (km)
        v_leader: Leader velocity in inertial frame (km/s)
        r_follower: Follower position in inertial frame (km)
        v_follower: Follower velocity in inertial frame (km/s)
    
    Returns:
        Tuple of (delta_r_hill, delta_v_hill):
            - delta_r_hill: Relative position in Hill frame [δx, δy, δz] (km)
            - delta_v_hill: Relative velocity in Hill frame [δvx, δvy, δvz] (km/s)
    
    Example:
        >>> r_leader = jnp.array([7000.0, 0.0, 0.0])
        >>> v_leader = jnp.array([0.0, 7.5, 0.0])
        >>> r_follower = jnp.array([7001.0, 0.0, 0.0])
        >>> v_follower = jnp.array([0.0, 7.5, 0.0])
        >>> delta_r, delta_v = inertial_to_hill_frame(
        ...     r_leader, v_leader, r_follower, v_follower
        ... )
    """
    # Relative state in inertial frame
    delta_r_inertial = r_follower - r_leader
    delta_v_inertial = v_follower - v_leader
    
    # Construct Hill frame basis vectors
    h = jnp.cross(r_leader, v_leader)
    h_mag = jnp.linalg.norm(h)
    
    z_hill = h / h_mag
    x_hill = r_leader / jnp.linalg.norm(r_leader)
    y_hill = jnp.cross(z_hill, x_hill)
    
    # Rotation matrix: Inertial -> Hill
    # Rows are basis vectors (transpose of Hill->Inertial)
    R_inertial_to_hill = jnp.row_stack([x_hill, y_hill, z_hill])
    
    # Transform relative position to Hill frame
    delta_r_hill = R_inertial_to_hill @ delta_r_inertial
    
    # Transform relative velocity to Hill frame
    # Need to remove Coriolis term
    r_leader_mag = jnp.linalg.norm(r_leader)
    omega_hill = h / (r_leader_mag ** 2)
    
    delta_v_hill = R_inertial_to_hill @ (delta_v_inertial - jnp.cross(omega_hill, delta_r_inertial))
    
    return delta_r_hill, delta_v_hill


# JIT-compile for performance
hill_clohessy_wiltshire_dynamics = jax.jit(hill_clohessy_wiltshire_dynamics)
relative_dynamics_nonlinear = jax.jit(relative_dynamics_nonlinear)
hill_frame_to_inertial = jax.jit(hill_frame_to_inertial)
inertial_to_hill_frame = jax.jit(inertial_to_hill_frame)
