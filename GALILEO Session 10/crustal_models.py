"""
Crustal Density Models and Terrain Corrections

Provides crustal density priors, terrain corrections, and Bouguer reductions
for gravity data processing.
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class CrustalDensityModel:
    """
    Container for crustal density model data.
    
    Attributes:
        name: Model name (e.g., 'CRUST1.0', 'LITHO1.0')
        lat_grid: Latitude grid
        lon_grid: Longitude grid
        layer_densities: Dict of layer densities (kg/m³)
        layer_thicknesses: Dict of layer thicknesses (m)
        basement_depth: Depth to basement (m)
        metadata: Additional model metadata
    """
    name: str
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    layer_densities: Dict[str, np.ndarray]
    layer_thicknesses: Dict[str, np.ndarray]
    basement_depth: Optional[np.ndarray] = None
    metadata: Optional[Dict] = None
    
    def get_density_at_depth(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        depth: float
    ) -> np.ndarray:
        """
        Get crustal density at specified depth via interpolation.
        
        Args:
            lat: Latitude points
            lon: Longitude points
            depth: Depth in meters
            
        Returns:
            Density in kg/m³
        """
        # Simplified placeholder - would use proper 3D interpolation
        # Use upper crustal density as approximation
        if 'upper_crust' in self.layer_densities:
            return np.ones_like(lat) * np.mean(self.layer_densities['upper_crust'])
        return np.ones_like(lat) * 2670.0  # Standard crustal density
    
    def get_integrated_density(
        self,
        lat: np.ndarray,
        lon: np.ndarray,
        depth_range: Tuple[float, float]
    ) -> np.ndarray:
        """
        Get integrated crustal density over depth range.
        
        Args:
            lat: Latitude points
            lon: Longitude points
            depth_range: (min_depth, max_depth) in meters
            
        Returns:
            Integrated density
        """
        depth_min, depth_max = depth_range
        thickness = depth_max - depth_min
        
        # Average density over range (simplified)
        avg_density = self.get_density_at_depth(lat, lon, 
                                                (depth_min + depth_max) / 2)
        
        return avg_density * thickness


def load_crust1(data_path: Optional[Path] = None) -> CrustalDensityModel:
    """
    Load CRUST1.0 global crustal model.
    
    Args:
        data_path: Path to CRUST1.0 data files
        
    Returns:
        CrustalDensityModel instance
    """
    if data_path is not None and data_path.exists():
        # Load from actual CRUST1.0 files
        pass
    
    # Create placeholder model with typical crustal structure
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    
    # Define typical crustal layers with densities
    layer_densities = {
        'water': np.ones_like(lat_2d) * 1030.0,  # Ocean water
        'ice': np.ones_like(lat_2d) * 920.0,     # Ice sheets
        'soft_sediment': np.ones_like(lat_2d) * 2200.0,
        'hard_sediment': np.ones_like(lat_2d) * 2500.0,
        'upper_crust': np.ones_like(lat_2d) * 2670.0,
        'middle_crust': np.ones_like(lat_2d) * 2800.0,
        'lower_crust': np.ones_like(lat_2d) * 2920.0,
    }
    
    # Define typical layer thicknesses (varies by location)
    layer_thicknesses = {
        'water': np.random.uniform(0, 5000, lat_2d.shape),
        'ice': np.zeros_like(lat_2d),  # Only in polar regions
        'soft_sediment': np.random.uniform(0, 3000, lat_2d.shape),
        'hard_sediment': np.random.uniform(0, 2000, lat_2d.shape),
        'upper_crust': np.ones_like(lat_2d) * 10000.0,
        'middle_crust': np.ones_like(lat_2d) * 10000.0,
        'lower_crust': np.ones_like(lat_2d) * 15000.0,
    }
    
    # Ice sheets in polar regions
    layer_thicknesses['ice'][np.abs(lat_2d) > 60] = np.random.uniform(
        0, 3000, np.sum(np.abs(lat_2d) > 60)
    )
    
    # Compute basement depth
    basement_depth = (
        layer_thicknesses['soft_sediment'] +
        layer_thicknesses['hard_sediment']
    )
    
    metadata = {
        'description': 'CRUST1.0 Global Crustal Model',
        'reference': 'Laske et al. (2013)',
        'resolution': '1 degree',
        'placeholder': True
    }
    
    return CrustalDensityModel(
        name='CRUST1.0',
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        layer_densities=layer_densities,
        layer_thicknesses=layer_thicknesses,
        basement_depth=basement_depth,
        metadata=metadata
    )


def terrain_correction(
    lat: np.ndarray,
    lon: np.ndarray,
    elevation: np.ndarray,
    dem: np.ndarray,
    dem_lat: np.ndarray,
    dem_lon: np.ndarray,
    density: float = 2670.0,
    radius: float = 100000.0
) -> np.ndarray:
    """
    Compute terrain correction for gravity measurements.
    
    Uses the Hammer zones method for computing gravitational effect of 
    topography around measurement points.
    
    Args:
        lat: Station latitudes
        lon: Station longitudes
        elevation: Station elevations (m)
        dem: Digital elevation model (2D array)
        dem_lat: DEM latitude grid
        dem_lon: DEM longitude grid
        density: Terrain density in kg/m³
        radius: Maximum radius for correction (m)
        
    Returns:
        Terrain correction in mGal
    """
    # Gravitational constant
    G = 6.67430e-11  # m³/(kg·s²)
    
    # Convert to mGal factor
    G_mgal = G * 1e5
    
    # Initialize correction
    correction = np.zeros_like(lat)
    
    # Earth radius for distance calculation
    R_earth = 6371000.0
    
    # Process each station
    for i, (station_lat, station_lon, station_elev) in enumerate(
        zip(lat.flat, lon.flat, elevation.flat)
    ):
        # Find DEM points within radius
        # Create 2D grid for distances
        lat_2d, lon_2d = np.meshgrid(dem_lat, dem_lon, indexing='ij')
        
        # Compute distances
        dlat = (lat_2d - station_lat) * np.pi / 180 * R_earth
        dlon = (lon_2d - station_lon) * np.pi / 180 * R_earth * \
               np.cos(np.radians(station_lat))
        
        # Distance from station
        distance = np.sqrt(dlat**2 + dlon**2)
        
        # Select points within radius
        mask = (distance < radius) & (distance > 0)
        
        if not np.any(mask):
            # No terrain points within radius
            continue
        
        # Height difference and distances
        dh = dem[mask] - station_elev
        dist = distance[mask]
        
        # Compute correction (positive above station, negative below)
        # Estimate area per DEM cell
        lat_spacing = np.mean(np.diff(dem_lat)) * np.pi / 180 * R_earth
        lon_spacing = np.mean(np.diff(dem_lon)) * np.pi / 180 * R_earth * \
                     np.cos(np.radians(station_lat))
        cell_area = lat_spacing * lon_spacing
        
        # Simplified terrain correction
        correction.flat[i] = np.sum(
            2 * G_mgal * density * cell_area * dh / (dist**2 + dh**2)**1.5
        )
    
    return correction.reshape(lat.shape)


def bouguer_correction(
    elevation: np.ndarray,
    density: float = 2670.0,
    slab_thickness: Optional[float] = None
) -> np.ndarray:
    """
    Compute Bouguer slab correction.
    
    Args:
        elevation: Elevation above reference in meters
        density: Crustal density in kg/m³
        slab_thickness: Slab thickness (if None, uses elevation)
        
    Returns:
        Bouguer correction in mGal
    """
    if slab_thickness is None:
        slab_thickness = elevation
    
    # Bouguer slab correction formula
    # δg = 2π G ρ h = 0.04193 ρ h (for ρ in g/cm³, h in m)
    # For ρ in kg/m³: factor = 2π G / 10
    
    G = 6.67430e-11  # m³/(kg·s²)
    factor = 2 * np.pi * G * 1e5  # Convert to mGal
    
    correction = factor * density * slab_thickness
    
    return correction


def compute_density_contrast(
    crustal_model: CrustalDensityModel,
    lat: np.ndarray,
    lon: np.ndarray,
    reference_density: float = 2670.0
) -> np.ndarray:
    """
    Compute density contrast relative to reference.
    
    Args:
        crustal_model: Crustal density model
        lat: Latitude points
        lon: Longitude points
        reference_density: Reference density in kg/m³
        
    Returns:
        Density contrast in kg/m³
    """
    # Get model density
    model_density = crustal_model.get_density_at_depth(lat, lon, depth=0)
    
    # Compute contrast
    contrast = model_density - reference_density
    
    return contrast


def isostatic_correction(
    lat: np.ndarray,
    lon: np.ndarray,
    elevation: np.ndarray,
    crustal_model: CrustalDensityModel,
    compensation_type: str = 'airy',
    compensation_depth: float = 35000.0
) -> np.ndarray:
    """
    Compute isostatic gravity correction.
    
    Args:
        lat: Latitude points
        lon: Longitude points
        elevation: Topographic elevation (m)
        crustal_model: Crustal density model
        compensation_type: 'airy' or 'pratt'
        compensation_depth: Depth of compensation (m)
        
    Returns:
        Isostatic correction in mGal
    """
    # Get crustal density
    rho_crust = crustal_model.get_density_at_depth(lat, lon, depth=0)
    rho_mantle = 3300.0  # Typical mantle density
    
    if compensation_type == 'airy':
        # Airy-Heiskanen isostatic model
        # Root depth = elevation * ρ_crust / (ρ_mantle - ρ_crust)
        root_depth = elevation * rho_crust / (rho_mantle - rho_crust)
        
        # Compute gravitational effect of root
        G = 6.67430e-11
        G_mgal = G * 1e5
        
        # Simplified computation
        correction = -2 * np.pi * G_mgal * (rho_mantle - rho_crust) * root_depth
        
    elif compensation_type == 'pratt':
        # Pratt-Hayford isostatic model
        # Density varies with elevation
        density_contrast = elevation * rho_crust / compensation_depth
        
        G_mgal = 6.67430e-11 * 1e5
        correction = -2 * np.pi * G_mgal * density_contrast * compensation_depth
        
    else:
        raise ValueError(f"Unknown compensation type: {compensation_type}")
    
    return correction


def complete_bouguer_anomaly(
    lat: np.ndarray,
    lon: np.ndarray,
    observed_gravity: np.ndarray,
    elevation: np.ndarray,
    dem: np.ndarray,
    dem_lat: np.ndarray,
    dem_lon: np.ndarray,
    density: float = 2670.0
) -> Dict[str, np.ndarray]:
    """
    Compute complete Bouguer anomaly with all corrections.
    
    Args:
        lat: Station latitudes
        lon: Station longitudes
        observed_gravity: Observed gravity (mGal)
        elevation: Station elevations (m)
        dem: Digital elevation model
        dem_lat: DEM latitude grid
        dem_lon: DEM longitude grid
        density: Terrain density (kg/m³)
        
    Returns:
        Dictionary with anomaly and all correction components
    """
    from .gravity_fields import compute_normal_gravity
    
    # Compute normal gravity
    normal_g = compute_normal_gravity(lat)
    
    # Free-air correction
    fac = -0.3086 * elevation
    
    # Bouguer slab correction
    bouguer = bouguer_correction(elevation, density)
    
    # Terrain correction
    terrain = terrain_correction(lat, lon, elevation, dem, dem_lat, dem_lon, density)
    
    # Complete Bouguer anomaly
    cba = observed_gravity - normal_g - fac + bouguer - terrain
    
    return {
        'complete_bouguer_anomaly': cba,
        'observed_gravity': observed_gravity,
        'normal_gravity': normal_g,
        'free_air_correction': fac,
        'bouguer_correction': bouguer,
        'terrain_correction': terrain,
        'free_air_anomaly': observed_gravity - normal_g - fac,
        'simple_bouguer_anomaly': observed_gravity - normal_g - fac + bouguer
    }
