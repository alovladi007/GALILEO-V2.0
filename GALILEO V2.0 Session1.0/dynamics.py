"""
Orbital dynamics and perturbations for satellite formation flying.

This module implements:
- Two-body Keplerian motion
- J2 perturbation (Earth oblateness)
- Atmospheric drag
- Solar radiation pressure
- Relative motion dynamics (Hill-Clohessy-Wiltshire equations)

All implementations are JAX-accelerated for GPU computation and automatic differentiation.
"""

from typing import Tuple, Optional
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from enum import Enum


# Physical constants
GM_EARTH = 3.986004418e14  # m³/s² - Earth gravitational parameter
R_EARTH = 6378137.0  # m - Earth equatorial radius
J2 = 1.08263e-3  # J2 coefficient (Earth oblateness)
OMEGA_EARTH = 7.2921159e-5  # rad/s - Earth rotation rate
AU = 1.495978707e11  # m - Astronomical unit
SOLAR_FLUX = 1367.0  # W/m² - Solar constant at 1 AU
C_LIGHT = 299792458.0  # m/s - Speed of light


@dataclass
class OrbitalState:
    """Complete orbital state representation."""
    
    position: jnp.ndarray  # [x, y, z] in ECI frame (m)
    velocity: jnp.ndarray  # [vx, vy, vz] in ECI frame (m/s)
    time: float  # epoch time (s since J2000)
    
    def to_array(self) -> jnp.ndarray:
        """Convert to state vector [x, y, z, vx, vy, vz]."""
        return jnp.concatenate([self.position, self.velocity])
    
    @classmethod
    def from_array(cls, state: jnp.ndarray, time: float) -> 'OrbitalState':
        """Create from state vector."""
        return cls(
            position=state[:3],
            velocity=state[3:6],
            time=time
        )


@dataclass
class SatelliteProperties:
    """Physical properties affecting perturbations."""
    
    mass: float  # kg
    area: float  # m² - cross-sectional area
    cd: float = 2.2  # drag coefficient
    cr: float = 1.3  # reflectivity coefficient (1=perfect absorber, 2=perfect reflector)
    
    @property
    def area_to_mass(self) -> float:
        """Area-to-mass ratio (m²/kg)."""
        return self.area / self.mass


class PerturbationType(Enum):
    """Types of orbital perturbations."""
    J2 = "j2"
    DRAG = "drag"
    SRP = "solar_radiation_pressure"
    THIRD_BODY = "third_body"


@jax.jit
def two_body_acceleration(position: jnp.ndarray, gm: float = GM_EARTH) -> jnp.ndarray:
    """
    Compute two-body gravitational acceleration.
    
    Args:
        position: Position vector [x, y, z] (m)
        gm: Gravitational parameter (m³/s²)
    
    Returns:
        Acceleration vector (m/s²)
    
    Notes:
        a = -GM * r / |r|³
    """
    r = jnp.linalg.norm(position)
    return -gm * position / (r**3)


@jax.jit
def j2_acceleration(position: jnp.ndarray) -> jnp.ndarray:
    """
    Compute J2 perturbation acceleration (Earth oblateness).
    
    Args:
        position: Position vector [x, y, z] in ECI frame (m)
    
    Returns:
        Acceleration vector (m/s²)
    
    Notes:
        J2 is the second zonal harmonic representing Earth's equatorial bulge.
        
        a_J2 = (3/2) * J2 * (GM/r²) * (R_E/r)² * [
            (5z²/r² - 1) * x_hat,
            (5z²/r² - 1) * y_hat,
            (5z²/r² - 3) * z_hat
        ]
    """
    x, y, z = position[0], position[1], position[2]
    r = jnp.linalg.norm(position)
    
    # Precompute common terms
    r2 = r * r
    r_ratio = R_EARTH / r
    z2_over_r2 = (z * z) / r2
    
    # J2 coefficient factor
    factor = 1.5 * J2 * GM_EARTH * r_ratio**2 / r2
    
    # Acceleration components
    ax = factor * (5.0 * z2_over_r2 - 1.0) * x
    ay = factor * (5.0 * z2_over_r2 - 1.0) * y
    az = factor * (5.0 * z2_over_r2 - 3.0) * z
    
    return jnp.array([ax, ay, az])


@jax.jit
def atmospheric_drag_acceleration(
    position: jnp.ndarray,
    velocity: jnp.ndarray,
    sat_props: SatelliteProperties,
    rho: float
) -> jnp.ndarray:
    """
    Compute atmospheric drag acceleration.
    
    Args:
        position: Position vector [x, y, z] in ECI frame (m)
        velocity: Velocity vector [vx, vy, vz] in ECI frame (m/s)
        sat_props: Satellite physical properties
        rho: Atmospheric density at satellite altitude (kg/m³)
    
    Returns:
        Acceleration vector (m/s²)
    
    Notes:
        a_drag = -0.5 * (Cd * A/m) * rho * v_rel * |v_rel|
        
        where v_rel accounts for Earth's rotation.
    """
    # Relative velocity (accounting for Earth rotation)
    omega_vec = jnp.array([0.0, 0.0, OMEGA_EARTH])
    v_rot = jnp.cross(omega_vec, position)
    v_rel = velocity - v_rot
    v_rel_mag = jnp.linalg.norm(v_rel)
    
    # Drag acceleration
    drag_coeff = -0.5 * sat_props.cd * sat_props.area_to_mass * rho
    return drag_coeff * v_rel_mag * v_rel


def exponential_atmosphere(altitude: float) -> float:
    """
    Compute atmospheric density using exponential model.
    
    Args:
        altitude: Altitude above Earth surface (m)
    
    Returns:
        Atmospheric density (kg/m³)
    
    Notes:
        This is a simplified exponential model. For production use,
        consider NRLMSISE-00 or JB-2008 models.
        
        ρ(h) = ρ₀ * exp(-h/H)
        
        where H is the scale height (~8500m for low altitudes).
    """
    # Reference values at different altitudes
    if altitude < 200e3:
        rho_0 = 2.46e-10  # kg/m³ at 200 km
        h_0 = 200e3
        H = 58.515e3  # scale height
    elif altitude < 500e3:
        rho_0 = 6.07e-13  # kg/m³ at 500 km
        h_0 = 500e3
        H = 71.835e3
    else:
        rho_0 = 1.45e-15  # kg/m³ at 800 km
        h_0 = 800e3
        H = 173.0e3
    
    return rho_0 * np.exp(-(altitude - h_0) / H)


@jax.jit
def solar_radiation_pressure_acceleration(
    position: jnp.ndarray,
    sun_position: jnp.ndarray,
    sat_props: SatelliteProperties,
    in_shadow: bool = False
) -> jnp.ndarray:
    """
    Compute solar radiation pressure acceleration.
    
    Args:
        position: Satellite position [x, y, z] in ECI frame (m)
        sun_position: Sun position [x, y, z] in ECI frame (m)
        sat_props: Satellite physical properties
        in_shadow: Whether satellite is in Earth's shadow
    
    Returns:
        Acceleration vector (m/s²)
    
    Notes:
        a_srp = (P/c) * Cr * (A/m) * (r_sun/|r_sun|)
        
        where P = solar flux / distance²
    """
    if in_shadow:
        return jnp.zeros(3)
    
    # Vector from satellite to sun
    r_sat_sun = sun_position - position
    dist = jnp.linalg.norm(r_sat_sun)
    
    # Solar pressure at satellite
    pressure = SOLAR_FLUX / C_LIGHT * (AU / dist)**2
    
    # SRP acceleration
    srp_coeff = pressure * sat_props.cr * sat_props.area_to_mass
    return srp_coeff * (r_sat_sun / dist)


def check_shadow(
    sat_position: jnp.ndarray,
    sun_position: jnp.ndarray
) -> bool:
    """
    Check if satellite is in Earth's shadow (cylindrical shadow model).
    
    Args:
        sat_position: Satellite position in ECI (m)
        sun_position: Sun position in ECI (m)
    
    Returns:
        True if satellite is in shadow
    """
    # Sun direction
    sun_dir = sun_position / np.linalg.norm(sun_position)
    
    # Project satellite position onto sun direction
    proj = np.dot(sat_position, sun_dir)
    
    # If satellite is on sunlit side, not in shadow
    if proj > 0:
        return False
    
    # Distance from shadow axis
    perp_dist = np.linalg.norm(sat_position - proj * sun_dir)
    
    # Check if within Earth's shadow cylinder
    return perp_dist < R_EARTH


@jax.jit
def orbital_energy(state: jnp.ndarray, gm: float = GM_EARTH) -> float:
    """
    Compute specific orbital energy (energy per unit mass).
    
    Args:
        state: State vector [x, y, z, vx, vy, vz]
        gm: Gravitational parameter
    
    Returns:
        Specific orbital energy (m²/s²)
    
    Notes:
        E = v²/2 - GM/r
    """
    pos = state[:3]
    vel = state[3:6]
    
    r = jnp.linalg.norm(pos)
    v2 = jnp.dot(vel, vel)
    
    return 0.5 * v2 - gm / r


@jax.jit
def angular_momentum(state: jnp.ndarray) -> jnp.ndarray:
    """
    Compute specific angular momentum vector.
    
    Args:
        state: State vector [x, y, z, vx, vy, vz]
    
    Returns:
        Angular momentum vector (m²/s)
    
    Notes:
        h = r × v
    """
    pos = state[:3]
    vel = state[3:6]
    return jnp.cross(pos, vel)


@jax.jit
def orbital_period(state: jnp.ndarray, gm: float = GM_EARTH) -> float:
    """
    Compute orbital period for elliptical orbit.
    
    Args:
        state: State vector [x, y, z, vx, vy, vz]
        gm: Gravitational parameter
    
    Returns:
        Orbital period (seconds)
    
    Notes:
        T = 2π√(a³/GM)
        
        where a = -GM/(2E) is the semi-major axis
    """
    energy = orbital_energy(state, gm)
    a = -gm / (2.0 * energy)
    return 2.0 * jnp.pi * jnp.sqrt(a**3 / gm)


@jax.jit
def hill_acceleration(
    relative_state: jnp.ndarray,
    chief_orbit_rate: float
) -> jnp.ndarray:
    """
    Compute relative acceleration using Hill-Clohessy-Wiltshire (HCW) equations.
    
    Args:
        relative_state: Relative state [x, y, z, vx, vy, vz] in LVLH frame
        chief_orbit_rate: Mean motion of chief orbit (rad/s)
    
    Returns:
        Relative acceleration [ax, ay, az] (m/s²)
    
    Notes:
        The HCW equations describe relative motion in a circular reference orbit:
        
        ẍ - 2nẏ - 3n²x = 0
        ÿ + 2nẋ = 0
        z̈ + n²z = 0
        
        where n is the mean motion (orbit rate) and (x,y,z) are in the
        Local-Vertical-Local-Horizontal (LVLH) frame:
        - x: radial (away from Earth)
        - y: along-track (direction of motion)
        - z: cross-track (normal to orbit plane)
    """
    x, y, z = relative_state[0], relative_state[1], relative_state[2]
    vx, vy, vz = relative_state[3], relative_state[4], relative_state[5]
    
    n = chief_orbit_rate
    n2 = n * n
    
    # HCW equations
    ax = 3.0 * n2 * x + 2.0 * n * vy
    ay = -2.0 * n * vx
    az = -n2 * z
    
    return jnp.array([ax, ay, az])


class OrbitPropagator:
    """Numerical orbit propagator with perturbations."""
    
    def __init__(
        self,
        perturbations: Optional[list[PerturbationType]] = None,
        sat_properties: Optional[SatelliteProperties] = None
    ):
        """
        Initialize orbit propagator.
        
        Args:
            perturbations: List of perturbations to include
            sat_properties: Satellite physical properties (required for drag/SRP)
        """
        self.perturbations = perturbations or []
        self.sat_properties = sat_properties or SatelliteProperties(
            mass=100.0,  # kg
            area=0.1,    # m²
            cd=2.2,
            cr=1.3
        )
    
    def acceleration(
        self,
        state: jnp.ndarray,
        t: float,
        sun_position: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute total acceleration including all perturbations.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz]
            t: Current time (s)
            sun_position: Sun position for SRP (optional)
        
        Returns:
            Total acceleration vector (m/s²)
        """
        pos = state[:3]
        vel = state[3:6]
        
        # Two-body acceleration (always included)
        acc = two_body_acceleration(pos)
        
        # Add perturbations
        if PerturbationType.J2 in self.perturbations:
            acc = acc + j2_acceleration(pos)
        
        if PerturbationType.DRAG in self.perturbations:
            r = jnp.linalg.norm(pos)
            altitude = r - R_EARTH
            rho = exponential_atmosphere(float(altitude))
            acc = acc + atmospheric_drag_acceleration(pos, vel, self.sat_properties, rho)
        
        if PerturbationType.SRP in self.perturbations and sun_position is not None:
            in_shadow = check_shadow(np.array(pos), np.array(sun_position))
            acc = acc + solar_radiation_pressure_acceleration(
                pos, sun_position, self.sat_properties, in_shadow
            )
        
        return acc
    
    def dynamics(
        self,
        state: jnp.ndarray,
        t: float,
        sun_position: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Compute state derivative [v, a] for integration.
        
        Args:
            state: State vector [x, y, z, vx, vy, vz]
            t: Current time (s)
            sun_position: Sun position for SRP (optional)
        
        Returns:
            State derivative [vx, vy, vz, ax, ay, az]
        """
        pos = state[:3]
        vel = state[3:6]
        acc = self.acceleration(state, t, sun_position)
        
        return jnp.concatenate([vel, acc])
    
    def propagate_rk4(
        self,
        state0: jnp.ndarray,
        t0: float,
        dt: float,
        n_steps: int,
        sun_position: Optional[jnp.ndarray] = None
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Propagate orbit using 4th-order Runge-Kutta integration.
        
        Args:
            state0: Initial state vector [x, y, z, vx, vy, vz]
            t0: Initial time (s)
            dt: Time step (s)
            n_steps: Number of steps
            sun_position: Sun position (constant for simplicity)
        
        Returns:
            times: Array of times (s)
            states: Array of states [n_steps+1, 6]
        """
        states = [state0]
        times = [t0]
        
        state = state0
        t = t0
        
        for _ in range(n_steps):
            # RK4 integration
            k1 = self.dynamics(state, t, sun_position)
            k2 = self.dynamics(state + 0.5 * dt * k1, t + 0.5 * dt, sun_position)
            k3 = self.dynamics(state + 0.5 * dt * k2, t + 0.5 * dt, sun_position)
            k4 = self.dynamics(state + dt * k3, t + dt, sun_position)
            
            state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t = t + dt
            
            states.append(state)
            times.append(t)
        
        return jnp.array(times), jnp.array(states)


def keplerian_to_cartesian(
    a: float,
    e: float,
    i: float,
    omega: float,
    w: float,
    nu: float,
    gm: float = GM_EARTH
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert Keplerian orbital elements to Cartesian state.
    
    Args:
        a: Semi-major axis (m)
        e: Eccentricity
        i: Inclination (rad)
        omega: Right ascension of ascending node (RAAN) (rad)
        w: Argument of periapsis (rad)
        nu: True anomaly (rad)
        gm: Gravitational parameter (m³/s²)
    
    Returns:
        position: Position vector [x, y, z] (m)
        velocity: Velocity vector [vx, vy, vz] (m/s)
    """
    # Orbital radius
    p = a * (1 - e**2)  # semi-latus rectum
    r = p / (1 + e * jnp.cos(nu))
    
    # Position in perifocal frame
    r_pqw = jnp.array([
        r * jnp.cos(nu),
        r * jnp.sin(nu),
        0.0
    ])
    
    # Velocity in perifocal frame
    v_pqw = jnp.sqrt(gm / p) * jnp.array([
        -jnp.sin(nu),
        e + jnp.cos(nu),
        0.0
    ])
    
    # Rotation matrices
    cos_omega, sin_omega = jnp.cos(omega), jnp.sin(omega)
    cos_i, sin_i = jnp.cos(i), jnp.sin(i)
    cos_w, sin_w = jnp.cos(w), jnp.sin(w)
    
    # PQW to ECI rotation matrix
    R = jnp.array([
        [cos_omega*cos_w - sin_omega*sin_w*cos_i, -cos_omega*sin_w - sin_omega*cos_w*cos_i, sin_omega*sin_i],
        [sin_omega*cos_w + cos_omega*sin_w*cos_i, -sin_omega*sin_w + cos_omega*cos_w*cos_i, -cos_omega*sin_i],
        [sin_w*sin_i, cos_w*sin_i, cos_i]
    ])
    
    # Transform to ECI
    position = R @ r_pqw
    velocity = R @ v_pqw
    
    return position, velocity


# JIT-compile key functions for performance
two_body_acceleration = jax.jit(two_body_acceleration)
j2_acceleration = jax.jit(j2_acceleration)
atmospheric_drag_acceleration = jax.jit(atmospheric_drag_acceleration)
solar_radiation_pressure_acceleration = jax.jit(solar_radiation_pressure_acceleration)
hill_acceleration = jax.jit(hill_acceleration)
orbital_energy = jax.jit(orbital_energy)
angular_momentum = jax.jit(angular_momentum)
orbital_period = jax.jit(orbital_period)
keplerian_to_cartesian = jax.jit(keplerian_to_cartesian)
