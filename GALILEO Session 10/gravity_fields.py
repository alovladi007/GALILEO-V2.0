"""
Reference Gravity Field Models

Handles loading and computation of reference gravity fields including:
- EGM96 (Earth Gravitational Model 1996)
- EGM2008 (Earth Gravitational Model 2008)
- WGS84 reference ellipsoid
- Gravity anomaly computation
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class GravityFieldModel:
    """
    Container for gravity field model data and metadata.
    
    Attributes:
        name: Model name (e.g., 'EGM96', 'EGM2008')
        degree_max: Maximum spherical harmonic degree
        coefficients: Dictionary containing C_nm and S_nm coefficients
        reference_radius: Reference radius in meters (typically Earth radius)
        gm: Gravitational parameter (GM) in m^3/s^2
        metadata: Additional model metadata
    """
    name: str
    degree_max: int
    coefficients: Dict[str, np.ndarray]
    reference_radius: float = 6378137.0  # WGS84 equatorial radius
    gm: float = 3.986004418e14  # WGS84 GM value
    metadata: Optional[Dict[str, Any]] = None
    
    def get_coefficient(self, n: int, m: int, coefficient_type: str = 'C') -> float:
        """
        Retrieve a specific spherical harmonic coefficient.
        
        Args:
            n: Degree
            m: Order
            coefficient_type: 'C' for cosine, 'S' for sine
            
        Returns:
            Coefficient value
        """
        if coefficient_type not in ['C', 'S']:
            raise ValueError("coefficient_type must be 'C' or 'S'")
        
        key = f"{coefficient_type}_{n}_{m}"
        return self.coefficients.get(key, 0.0)
    
    def compute_geoid_height(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        max_degree: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute geoid heights from spherical harmonic coefficients.
        
        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            max_degree: Maximum degree to use (None = use all)
            
        Returns:
            Geoid heights in meters
        """
        if max_degree is None or max_degree > self.degree_max:
            max_degree = self.degree_max
        
        # Convert to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Simplified placeholder computation
        # In production, would use proper spherical harmonic synthesis
        geoid = np.zeros_like(lat, dtype=float)
        
        # Very simplified approximation for demonstration
        # Real implementation would compute full spherical harmonic series
        for n in range(2, min(max_degree + 1, 20)):  # Limit for placeholder
            for m in range(n + 1):
                C_nm = self.get_coefficient(n, m, 'C')
                S_nm = self.get_coefficient(n, m, 'S')
                
                # Simplified Legendre polynomial (placeholder)
                P_nm = np.cos(n * lat_rad) * np.cos(m * lat_rad)
                
                geoid += C_nm * P_nm * np.cos(m * lon_rad)
                if m > 0:
                    geoid += S_nm * P_nm * np.sin(m * lon_rad)
        
        # Scale by reference radius
        geoid *= self.reference_radius / 100.0  # Rough scaling
        
        return geoid
    
    def save(self, filepath: Path) -> None:
        """Save model to file."""
        data = {
            'name': self.name,
            'degree_max': self.degree_max,
            'reference_radius': self.reference_radius,
            'gm': self.gm,
            'metadata': self.metadata,
            'coefficients': {k: v.tolist() if isinstance(v, np.ndarray) else v 
                           for k, v in self.coefficients.items()}
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'GravityFieldModel':
        """Load model from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        coefficients = {k: np.array(v) if isinstance(v, list) else v 
                       for k, v in data['coefficients'].items()}
        
        return cls(
            name=data['name'],
            degree_max=data['degree_max'],
            coefficients=coefficients,
            reference_radius=data['reference_radius'],
            gm=data['gm'],
            metadata=data.get('metadata')
        )


def load_egm96(data_path: Optional[Path] = None) -> GravityFieldModel:
    """
    Load EGM96 gravity field model.
    
    Args:
        data_path: Path to EGM96 coefficient file (if None, creates placeholder)
        
    Returns:
        GravityFieldModel instance
    """
    if data_path is not None and data_path.exists():
        return GravityFieldModel.load(data_path)
    
    # Create placeholder model with synthetic coefficients
    degree_max = 360
    coefficients = {}
    
    # Generate placeholder coefficients (would load from actual EGM96 file)
    for n in range(2, degree_max + 1):
        for m in range(n + 1):
            # Simplified placeholder values with realistic decay
            C_value = 1e-6 / (n ** 2) * np.random.randn()
            S_value = 1e-6 / (n ** 2) * np.random.randn() if m > 0 else 0.0
            
            coefficients[f"C_{n}_{m}"] = C_value
            if m > 0:
                coefficients[f"S_{n}_{m}"] = S_value
    
    metadata = {
        'description': 'Earth Gravitational Model 1996',
        'reference': 'NIMA (1996)',
        'complete_to_degree': 360,
        'placeholder': True
    }
    
    return GravityFieldModel(
        name='EGM96',
        degree_max=degree_max,
        coefficients=coefficients,
        metadata=metadata
    )


def load_egm2008(data_path: Optional[Path] = None) -> GravityFieldModel:
    """
    Load EGM2008 gravity field model.
    
    Args:
        data_path: Path to EGM2008 coefficient file (if None, creates placeholder)
        
    Returns:
        GravityFieldModel instance
    """
    if data_path is not None and data_path.exists():
        return GravityFieldModel.load(data_path)
    
    # Create placeholder model with higher resolution
    degree_max = 2190
    coefficients = {}
    
    # Generate placeholder coefficients (would load from actual EGM2008 file)
    # For efficiency in placeholder, only generate up to degree 500
    for n in range(2, min(degree_max, 500) + 1):
        for m in range(n + 1):
            C_value = 1e-6 / (n ** 2) * np.random.randn()
            S_value = 1e-6 / (n ** 2) * np.random.randn() if m > 0 else 0.0
            
            coefficients[f"C_{n}_{m}"] = C_value
            if m > 0:
                coefficients[f"S_{n}_{m}"] = S_value
    
    metadata = {
        'description': 'Earth Gravitational Model 2008',
        'reference': 'Pavlis et al. (2012)',
        'complete_to_degree': 2190,
        'placeholder': True,
        'note': 'Placeholder contains coefficients up to degree 500'
    }
    
    return GravityFieldModel(
        name='EGM2008',
        degree_max=degree_max,
        coefficients=coefficients,
        metadata=metadata
    )


def compute_gravity_anomaly(
    lat: np.ndarray,
    lon: np.ndarray,
    observed_gravity: np.ndarray,
    model: GravityFieldModel,
    correction_type: str = 'free_air',
    elevation: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Compute gravity anomaly relative to reference model.
    
    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        observed_gravity: Observed gravity values in mGal
        model: Reference gravity field model
        correction_type: Type of anomaly ('free_air', 'bouguer', 'isostatic')
        elevation: Elevation in meters (required for Bouguer)
        
    Returns:
        Tuple of (anomaly, components_dict)
        - anomaly: Gravity anomaly in mGal
        - components_dict: Dictionary of anomaly components
    """
    # Compute normal gravity on reference ellipsoid (WGS84)
    normal_gravity = compute_normal_gravity(lat)
    
    # Compute free-air correction
    free_air_correction = np.zeros_like(observed_gravity)
    if elevation is not None:
        # Standard free-air gradient: -0.3086 mGal/m
        free_air_correction = -0.3086 * elevation
    
    # Compute reference geoid contribution
    geoid_height = model.compute_geoid_height(lat, lon)
    geoid_correction = -0.3086 * geoid_height
    
    # Compute anomaly based on type
    components = {
        'normal_gravity': normal_gravity,
        'free_air_correction': free_air_correction,
        'geoid_correction': geoid_correction,
    }
    
    if correction_type == 'free_air':
        anomaly = observed_gravity - normal_gravity - free_air_correction
    
    elif correction_type == 'bouguer':
        if elevation is None:
            raise ValueError("Elevation required for Bouguer anomaly")
        
        # Bouguer slab correction (assuming density of 2670 kg/m³)
        bouguer_correction = 0.1119 * elevation
        components['bouguer_correction'] = bouguer_correction
        
        anomaly = (observed_gravity - normal_gravity - 
                  free_air_correction + bouguer_correction)
    
    elif correction_type == 'isostatic':
        # Placeholder for isostatic correction
        # Would compute Airy-Heiskanen or Pratt-Hayford isostasy
        isostatic_correction = np.zeros_like(observed_gravity)
        components['isostatic_correction'] = isostatic_correction
        
        anomaly = observed_gravity - normal_gravity - isostatic_correction
    
    else:
        raise ValueError(f"Unknown correction type: {correction_type}")
    
    return anomaly, components


def compute_normal_gravity(lat: np.ndarray) -> np.ndarray:
    """
    Compute normal gravity on WGS84 ellipsoid using Somigliana's formula.
    
    Args:
        lat: Latitude in degrees
        
    Returns:
        Normal gravity in mGal
    """
    # WGS84 parameters
    g_e = 9.7803253359  # Equatorial gravity (m/s²)
    g_p = 9.8321849378  # Polar gravity (m/s²)
    
    # Convert to radians
    lat_rad = np.radians(lat)
    
    # Somigliana's formula
    sin2_lat = np.sin(lat_rad) ** 2
    
    numerator = g_e * np.cos(lat_rad) ** 2 + g_p * sin2_lat
    denominator = np.sqrt(1 - 0.00669437999014 * sin2_lat)  # WGS84 eccentricity
    
    normal_g = numerator / denominator
    
    # Convert to mGal
    return normal_g * 1e5


def compute_gravity_gradient(
    gravity_field: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    component: str = 'vertical'
) -> np.ndarray:
    """
    Compute gravity gradient tensor components.
    
    Args:
        gravity_field: 2D gravity field array
        lat: 2D latitude array
        lon: 2D longitude array
        component: Gradient component ('vertical', 'horizontal', 'full_tensor')
        
    Returns:
        Gravity gradient(s)
    """
    # Convert to meters for gradient computation
    R_earth = 6371000.0  # Earth radius in meters
    
    # Compute grid spacing
    dlat = np.gradient(lat, axis=0)
    dlon = np.gradient(lon, axis=1)
    
    # Convert to distance
    dy = dlat * R_earth * np.pi / 180
    dx = dlon * R_earth * np.pi / 180 * np.cos(np.radians(lat))
    
    # Compute gradients
    dg_dy = np.gradient(gravity_field, axis=0) / dy
    dg_dx = np.gradient(gravity_field, axis=1) / dx
    
    if component == 'vertical':
        # Vertical gradient (approximation)
        return -2 * gravity_field / R_earth
    
    elif component == 'horizontal':
        return np.sqrt(dg_dx**2 + dg_dy**2)
    
    elif component == 'full_tensor':
        # Return dict with all components
        return {
            'dg_dx': dg_dx,
            'dg_dy': dg_dy,
            'dg_dz': -2 * gravity_field / R_earth,
            'd2g_dx2': np.gradient(dg_dx, axis=1) / dx,
            'd2g_dy2': np.gradient(dg_dy, axis=0) / dy,
        }
    
    else:
        raise ValueError(f"Unknown component: {component}")
