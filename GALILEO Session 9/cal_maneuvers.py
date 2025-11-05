"""
Calibration Maneuvers and Synthetic Data Generation

Generate synthetic calibration data for testing orbit determination:
- Thruster firings
- Attitude maneuvers
- Ballistic coefficient changes
- Solar panel articulation
- Measurement geometry optimization
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum


class ManeuverType(Enum):
    """Types of calibration maneuvers."""
    TRANSLATION = "translation"  # Delta-V maneuver
    ROTATION = "rotation"  # Attitude change
    BALLISTIC_CHANGE = "ballistic_change"  # Area or mass change
    SOLAR_PANEL = "solar_panel"  # Panel articulation
    COAST = "coast"  # Passive arc


@dataclass
class Maneuver:
    """Calibration maneuver specification."""
    type: ManeuverType
    start_time: float  # seconds
    duration: float  # seconds
    parameters: Dict  # Maneuver-specific parameters
    
    @property
    def end_time(self) -> float:
        """End time of maneuver."""
        return self.start_time + self.duration
    
    def is_active(self, time: float) -> bool:
        """Check if maneuver is active at given time."""
        return self.start_time <= time <= self.end_time


class CalibrationManeuverGenerator:
    """
    Generate synthetic calibration maneuver sequences.
    
    Designs maneuver sequences to:
    - Excite specific dynamics
    - Maximize observability
    - Separate parameter correlations
    """
    
    def __init__(self, orbital_period: float):
        """
        Initialize maneuver generator.
        
        Parameters
        ----------
        orbital_period : float
            Orbital period (seconds)
        """
        self.orbital_period = orbital_period
    
    def generate_delta_v_sequence(self, 
                                 n_maneuvers: int = 5,
                                 delta_v_range: Tuple[float, float] = (0.1, 1.0),
                                 spacing: str = 'uniform') -> List[Maneuver]:
        """
        Generate sequence of delta-V maneuvers.
        
        Parameters
        ----------
        n_maneuvers : int
            Number of maneuvers
        delta_v_range : tuple
            Range of delta-V magnitudes (m/s)
        spacing : str
            'uniform' or 'optimal' spacing
        
        Returns
        -------
        list
            Maneuver sequence
        """
        maneuvers = []
        
        if spacing == 'uniform':
            times = np.linspace(0, self.orbital_period * 5, n_maneuvers)
        elif spacing == 'optimal':
            # Space at 1/4 orbital period for maximum observability
            times = np.arange(n_maneuvers) * (self.orbital_period / 4)
        else:
            raise ValueError(f"Unknown spacing: {spacing}")
        
        for i, t in enumerate(times):
            # Random delta-V direction
            direction = np.random.randn(3)
            direction /= np.linalg.norm(direction)
            
            # Random magnitude
            magnitude = np.random.uniform(*delta_v_range)
            
            maneuver = Maneuver(
                type=ManeuverType.TRANSLATION,
                start_time=t,
                duration=60.0,  # 1 minute burn
                parameters={
                    'delta_v': direction * magnitude,
                    'thrust_efficiency': 0.95 + 0.05 * np.random.rand(),
                    'isp': 200 + 50 * np.random.rand()
                }
            )
            
            maneuvers.append(maneuver)
        
        return maneuvers
    
    def generate_attitude_sequence(self,
                                   n_maneuvers: int = 3) -> List[Maneuver]:
        """
        Generate sequence of attitude maneuvers.
        
        Used to calibrate:
        - Center of mass offset
        - Moment of inertia
        - Thrust misalignment
        
        Parameters
        ----------
        n_maneuvers : int
            Number of attitude changes
        
        Returns
        -------
        list
            Maneuver sequence
        """
        maneuvers = []
        
        attitudes = [
            {'roll': 0, 'pitch': 0, 'yaw': 0},  # Nominal
            {'roll': 90, 'pitch': 0, 'yaw': 0},  # Roll 90°
            {'roll': 0, 'pitch': 90, 'yaw': 0},  # Pitch 90°
        ]
        
        for i in range(min(n_maneuvers, len(attitudes))):
            t = i * self.orbital_period / 2
            
            maneuver = Maneuver(
                type=ManeuverType.ROTATION,
                start_time=t,
                duration=300.0,  # 5 minute rotation
                parameters={
                    'target_attitude': attitudes[i],
                    'slew_rate': 0.5  # deg/s
                }
            )
            
            maneuvers.append(maneuver)
        
        return maneuvers
    
    def generate_ballistic_coefficient_changes(self,
                                               n_changes: int = 4) -> List[Maneuver]:
        """
        Generate ballistic coefficient changes.
        
        Simulates deploying/retracting solar panels or drag panels
        to vary cross-sectional area.
        
        Parameters
        ----------
        n_changes : int
            Number of configuration changes
        
        Returns
        -------
        list
            Maneuver sequence
        """
        maneuvers = []
        
        # Configurations with different areas
        configs = [
            {'area_factor': 1.0, 'name': 'nominal'},
            {'area_factor': 1.5, 'name': 'panels_deployed'},
            {'area_factor': 0.8, 'name': 'panels_retracted'},
            {'area_factor': 1.2, 'name': 'mixed'}
        ]
        
        for i in range(min(n_changes, len(configs))):
            t = i * self.orbital_period
            
            maneuver = Maneuver(
                type=ManeuverType.BALLISTIC_CHANGE,
                start_time=t,
                duration=self.orbital_period,  # Hold for one orbit
                parameters={
                    'area_factor': configs[i]['area_factor'],
                    'configuration': configs[i]['name']
                }
            )
            
            maneuvers.append(maneuver)
        
        return maneuvers
    
    def generate_comprehensive_sequence(self) -> List[Maneuver]:
        """
        Generate comprehensive calibration sequence.
        
        Combines different maneuver types for full system calibration.
        
        Returns
        -------
        list
            Complete maneuver sequence
        """
        maneuvers = []
        
        # Coast arc for baseline
        maneuvers.append(Maneuver(
            type=ManeuverType.COAST,
            start_time=0,
            duration=self.orbital_period,
            parameters={}
        ))
        
        # Delta-V maneuvers
        t_offset = self.orbital_period
        dv_maneuvers = self.generate_delta_v_sequence(n_maneuvers=3, spacing='optimal')
        for m in dv_maneuvers:
            m.start_time += t_offset
            maneuvers.append(m)
        
        # Ballistic changes
        t_offset += 2 * self.orbital_period
        bc_maneuvers = self.generate_ballistic_coefficient_changes(n_changes=2)
        for m in bc_maneuvers:
            m.start_time += t_offset
            maneuvers.append(m)
        
        # Final coast
        t_offset += 3 * self.orbital_period
        maneuvers.append(Maneuver(
            type=ManeuverType.COAST,
            start_time=t_offset,
            duration=self.orbital_period,
            parameters={}
        ))
        
        return maneuvers


class SyntheticOrbitGenerator:
    """
    Generate synthetic orbit data with maneuvers and perturbations.
    
    Includes:
    - Keplerian motion
    - Drag and SRP perturbations
    - Maneuver effects
    - Measurement noise
    """
    
    def __init__(self, mu: float = 3.986004418e14):
        """
        Initialize orbit generator.
        
        Parameters
        ----------
        mu : float
            Gravitational parameter (m³/s²)
        """
        self.mu = mu
    
    def propagate_keplerian(self, 
                           state0: np.ndarray,
                           times: np.ndarray) -> np.ndarray:
        """
        Propagate Keplerian orbit.
        
        Parameters
        ----------
        state0 : np.ndarray
            Initial state [x, y, z, vx, vy, vz]
        times : np.ndarray
            Time epochs
        
        Returns
        -------
        np.ndarray
            States at each time (N x 6)
        """
        # Simplified two-body propagation
        r0 = state0[:3]
        v0 = state0[3:]
        
        # Orbital elements (simplified)
        r0_mag = np.linalg.norm(r0)
        v0_mag = np.linalg.norm(v0)
        
        # Semi-major axis
        a = 1.0 / (2.0/r0_mag - v0_mag**2/self.mu)
        
        # Mean motion
        n = np.sqrt(self.mu / a**3)
        
        states = np.zeros((len(times), 6))
        
        for i, t in enumerate(times):
            # Mean anomaly
            M = n * t
            
            # Simple circular approximation for demonstration
            # (real implementation would solve Kepler's equation)
            theta = M
            
            # Position in orbital plane
            r = a * np.array([np.cos(theta), np.sin(theta), 0])
            v = np.sqrt(self.mu/a) * np.array([-np.sin(theta), np.cos(theta), 0])
            
            # Rotate to inertial frame (simplified)
            states[i, :3] = r
            states[i, 3:] = v
        
        return states
    
    def add_drag_perturbation(self,
                             states: np.ndarray,
                             times: np.ndarray,
                             cd: float,
                             area_to_mass: float,
                             density_model: callable) -> np.ndarray:
        """
        Add atmospheric drag perturbation.
        
        Parameters
        ----------
        states : np.ndarray
            Unperturbed states (N x 6)
        times : np.ndarray
            Time epochs
        cd : float
            Drag coefficient
        area_to_mass : float
            Area-to-mass ratio (m²/kg)
        density_model : callable
            Function(altitude) -> density
        
        Returns
        -------
        np.ndarray
            Perturbed states
        """
        dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
        
        perturbed = states.copy()
        
        for i in range(len(times)):
            r = states[i, :3]
            v = states[i, 3:]
            
            altitude = np.linalg.norm(r) - 6371e3  # Simplified
            rho = density_model(altitude)
            
            v_mag = np.linalg.norm(v)
            if v_mag > 0:
                drag_accel = -0.5 * cd * area_to_mass * rho * v_mag * v
                
                # Integrate acceleration
                perturbed[i, 3:] += drag_accel * dt[i]
                perturbed[i, :3] += perturbed[i, 3:] * dt[i]
        
        return perturbed
    
    def add_srp_perturbation(self,
                            states: np.ndarray,
                            times: np.ndarray,
                            cr: float,
                            area_to_mass: float,
                            sun_vectors: np.ndarray) -> np.ndarray:
        """
        Add solar radiation pressure perturbation.
        
        Parameters
        ----------
        states : np.ndarray
            Unperturbed states (N x 6)
        times : np.ndarray
            Time epochs
        cr : float
            Radiation pressure coefficient
        area_to_mass : float
            Area-to-mass ratio
        sun_vectors : np.ndarray
            Unit vectors to sun (N x 3)
        
        Returns
        -------
        np.ndarray
            Perturbed states
        """
        dt = np.diff(times, prepend=times[0] - (times[1] - times[0]))
        
        P_sun = 4.56e-6  # N/m²
        c = 299792458.0  # m/s
        
        perturbed = states.copy()
        
        for i in range(len(times)):
            # SRP acceleration (away from sun)
            srp_mag = cr * area_to_mass * (P_sun / c)
            srp_accel = -srp_mag * sun_vectors[i]
            
            # Integrate
            perturbed[i, 3:] += srp_accel * dt[i]
            perturbed[i, :3] += perturbed[i, 3:] * dt[i]
        
        return perturbed
    
    def apply_maneuvers(self,
                       states: np.ndarray,
                       times: np.ndarray,
                       maneuvers: List[Maneuver]) -> np.ndarray:
        """
        Apply maneuver effects to states.
        
        Parameters
        ----------
        states : np.ndarray
            States before maneuvers (N x 6)
        times : np.ndarray
            Time epochs
        maneuvers : list
            Maneuver sequence
        
        Returns
        -------
        np.ndarray
            States with maneuvers applied
        """
        states_with_maneuvers = states.copy()
        
        for maneuver in maneuvers:
            if maneuver.type == ManeuverType.TRANSLATION:
                # Find times during maneuver
                mask = (times >= maneuver.start_time) & (times <= maneuver.end_time)
                
                if np.any(mask):
                    # Apply delta-V instantaneously at start
                    first_idx = np.where(mask)[0][0]
                    delta_v = maneuver.parameters['delta_v']
                    efficiency = maneuver.parameters.get('thrust_efficiency', 1.0)
                    
                    states_with_maneuvers[first_idx:, 3:] += delta_v * efficiency
            
            elif maneuver.type == ManeuverType.BALLISTIC_CHANGE:
                # This affects future drag calculations (handled externally)
                pass
        
        return states_with_maneuvers
    
    def generate_measurements(self,
                            states: np.ndarray,
                            measurement_type: str = 'range',
                            noise_std: float = 10.0) -> np.ndarray:
        """
        Generate synthetic measurements with noise.
        
        Parameters
        ----------
        states : np.ndarray
            True states (N x 6)
        measurement_type : str
            'range', 'range_rate', or 'position'
        noise_std : float
            Measurement noise standard deviation
        
        Returns
        -------
        np.ndarray
            Noisy measurements
        """
        n = len(states)
        
        if measurement_type == 'range':
            # Range from origin
            measurements = np.linalg.norm(states[:, :3], axis=1)
        elif measurement_type == 'range_rate':
            # Range rate
            r = states[:, :3]
            v = states[:, 3:]
            measurements = np.sum(r * v, axis=1) / np.linalg.norm(r, axis=1)
        elif measurement_type == 'position':
            measurements = states[:, :3]
        else:
            raise ValueError(f"Unknown measurement type: {measurement_type}")
        
        # Add noise
        noise = np.random.randn(*measurements.shape) * noise_std
        noisy_measurements = measurements + noise
        
        return noisy_measurements


def create_example_scenario() -> Dict:
    """
    Create example calibration scenario.
    
    Returns
    -------
    dict
        Scenario specification with maneuvers and conditions
    """
    # LEO satellite
    orbital_period = 5400.0  # 90 minutes
    
    generator = CalibrationManeuverGenerator(orbital_period)
    
    # Generate comprehensive maneuver sequence
    maneuvers = generator.generate_comprehensive_sequence()
    
    # Initial conditions
    r0 = 6371e3 + 400e3  # 400 km altitude
    v0 = np.sqrt(3.986004418e14 / r0)
    
    initial_state = np.array([r0, 0, 0, 0, v0, 0])
    
    # Physical parameters
    spacecraft_params = {
        'mass': 500.0,  # kg
        'area': 5.0,  # m²
        'cd': 2.2,
        'cr': 1.5
    }
    
    # Time span
    total_duration = maneuvers[-1].end_time
    times = np.linspace(0, total_duration, int(total_duration / 10) + 1)
    
    scenario = {
        'initial_state': initial_state,
        'times': times,
        'maneuvers': maneuvers,
        'spacecraft': spacecraft_params,
        'orbital_period': orbital_period
    }
    
    return scenario


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("=" * 60)
    print("CALIBRATION MANEUVER GENERATION")
    print("=" * 60)
    
    # Create scenario
    scenario = create_example_scenario()
    
    print(f"\nOrbital period: {scenario['orbital_period']/60:.1f} minutes")
    print(f"Total duration: {scenario['times'][-1]/3600:.2f} hours")
    print(f"Number of time steps: {len(scenario['times'])}")
    print(f"\nSpacecraft parameters:")
    for key, val in scenario['spacecraft'].items():
        print(f"  {key}: {val}")
    
    print(f"\nManeuver sequence ({len(scenario['maneuvers'])} maneuvers):")
    print("-" * 60)
    
    for i, m in enumerate(scenario['maneuvers']):
        print(f"\n{i+1}. {m.type.value.upper()}")
        print(f"   Time: {m.start_time/60:.1f} - {m.end_time/60:.1f} min")
        print(f"   Duration: {m.duration/60:.1f} min")
        if m.parameters:
            print(f"   Parameters:")
            for key, val in m.parameters.items():
                if isinstance(val, np.ndarray):
                    print(f"     {key}: {val}")
                else:
                    print(f"     {key}: {val}")
    
    # Generate synthetic orbit
    print("\n" + "=" * 60)
    print("SYNTHETIC ORBIT GENERATION")
    print("=" * 60)
    
    orbit_gen = SyntheticOrbitGenerator()
    
    # Propagate
    print("\nPropagating Keplerian orbit...")
    states = orbit_gen.propagate_keplerian(scenario['initial_state'], scenario['times'])
    
    # Add perturbations
    print("Adding drag perturbation...")
    density_model = lambda alt: 1e-12 * np.exp(-alt / 60e3)
    states = orbit_gen.add_drag_perturbation(
        states, scenario['times'],
        scenario['spacecraft']['cd'],
        scenario['spacecraft']['area'] / scenario['spacecraft']['mass'],
        density_model
    )
    
    print("Adding SRP perturbation...")
    sun_vectors = np.random.randn(len(scenario['times']), 3)
    sun_vectors /= np.linalg.norm(sun_vectors, axis=1, keepdims=True)
    states = orbit_gen.add_srp_perturbation(
        states, scenario['times'],
        scenario['spacecraft']['cr'],
        scenario['spacecraft']['area'] / scenario['spacecraft']['mass'],
        sun_vectors
    )
    
    print("Applying maneuvers...")
    states = orbit_gen.apply_maneuvers(states, scenario['times'], scenario['maneuvers'])
    
    # Generate measurements
    print("Generating measurements...")
    measurements = orbit_gen.generate_measurements(states, 'range', noise_std=10.0)
    
    print(f"\nFinal statistics:")
    print(f"  Position range: {np.min(np.linalg.norm(states[:, :3], axis=1))/1e3:.1f} - "
          f"{np.max(np.linalg.norm(states[:, :3], axis=1))/1e3:.1f} km")
    print(f"  Velocity range: {np.min(np.linalg.norm(states[:, 3:], axis=1)):.1f} - "
          f"{np.max(np.linalg.norm(states[:, 3:], axis=1)):.1f} m/s")
    print(f"  Measurement range: {np.min(measurements)/1e3:.1f} - "
          f"{np.max(measurements)/1e3:.1f} km")
    print(f"  Measurement SNR: {np.mean(measurements)/10:.1f}")
    
    print("\n✓ Calibration maneuver generation completed successfully")
