"""
Hydrology Models and Corrections

Provides seasonal water storage, groundwater models, and hydrological
corrections for time-variable gravity measurements.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class HydrologyModel:
    """
    Container for hydrological model data.
    
    Attributes:
        name: Model name (e.g., 'GLDAS', 'GRACE')
        lat_grid: Latitude grid
        lon_grid: Longitude grid
        water_storage: Water storage thickness (mm or kg/m²)
        time_stamps: Time stamps for temporal data
        components: Dict of water components (soil, snow, groundwater, etc.)
        metadata: Additional model metadata
    """
    name: str
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    water_storage: np.ndarray  # Shape: (time, lat, lon) or (lat, lon)
    time_stamps: Optional[List[datetime]] = None
    components: Optional[Dict[str, np.ndarray]] = None
    metadata: Optional[Dict] = None
    
    def get_storage_at_time(
        self,
        time_index: int,
        lat: Optional[np.ndarray] = None,
        lon: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get water storage at specific time, optionally at specific locations.
        
        Args:
            time_index: Time index
            lat: Latitude points for interpolation (optional)
            lon: Longitude points for interpolation (optional)
            
        Returns:
            Water storage in mm or kg/m²
        """
        if self.water_storage.ndim == 2:
            storage = self.water_storage
        else:
            storage = self.water_storage[time_index]
        
        if lat is not None and lon is not None:
            # Interpolate to requested points (simplified)
            # Would use proper 2D interpolation in production
            return np.ones_like(lat) * np.mean(storage)
        
        return storage
    
    def compute_seasonal_signal(
        self,
        lat: np.ndarray,
        lon: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute seasonal water storage signal parameters.
        
        Args:
            lat: Latitude grid
            lon: Longitude grid
            
        Returns:
            Dictionary with annual and semi-annual components
        """
        if self.water_storage.ndim != 3:
            raise ValueError("Temporal data required for seasonal analysis")
        
        # Perform harmonic analysis (simplified)
        time_axis = 0
        n_times = self.water_storage.shape[time_axis]
        
        # Assume regular time sampling over 1 year
        t = np.linspace(0, 2 * np.pi, n_times)
        
        # Fit annual and semi-annual harmonics
        # S(t) = A0 + A1*cos(ωt + φ1) + A2*cos(2ωt + φ2)
        
        annual_amp = np.zeros((len(self.lat_grid), len(self.lon_grid)))
        semiannual_amp = np.zeros_like(annual_amp)
        
        for i in range(len(self.lat_grid)):
            for j in range(len(self.lon_grid)):
                series = self.water_storage[:, i, j]
                
                # Fourier components
                annual_cos = np.sum(series * np.cos(t)) / n_times
                annual_sin = np.sum(series * np.sin(t)) / n_times
                annual_amp[i, j] = np.sqrt(annual_cos**2 + annual_sin**2)
                
                semiannual_cos = np.sum(series * np.cos(2*t)) / n_times
                semiannual_sin = np.sum(series * np.sin(2*t)) / n_times
                semiannual_amp[i, j] = np.sqrt(semiannual_cos**2 + semiannual_sin**2)
        
        return {
            'annual_amplitude': annual_amp,
            'semiannual_amplitude': semiannual_amp,
            'mean_storage': np.mean(self.water_storage, axis=0)
        }


def load_seasonal_water(
    source: str = 'GLDAS',
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    data_path: Optional[Path] = None
) -> HydrologyModel:
    """
    Load seasonal water storage model.
    
    Args:
        source: Data source ('GLDAS', 'GRACE', 'ERA5')
        start_date: Start date for data
        end_date: End date for data
        data_path: Path to data files
        
    Returns:
        HydrologyModel instance
    """
    # Create placeholder model with synthetic seasonal signal
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    
    # Generate 12 months of data
    n_months = 12
    time_stamps = [datetime(2024, i+1, 1) for i in range(n_months)]
    
    # Create synthetic seasonal signal
    water_storage = np.zeros((n_months, len(lat_grid), len(lon_grid)))
    
    for month in range(n_months):
        # Phase varies with latitude (northern vs southern hemisphere)
        phase = 2 * np.pi * month / 12
        
        # Seasonal amplitude varies with latitude
        amplitude = 100 * np.abs(np.cos(lat_2d * np.pi / 180))  # mm
        
        # Seasonal signal
        seasonal = amplitude * np.cos(phase - np.pi * np.sign(lat_2d))
        
        # Add random variations
        noise = np.random.normal(0, 20, seasonal.shape)
        
        water_storage[month] = seasonal + noise + 200  # Base + seasonal
    
    # Define components
    components = {
        'soil_moisture': water_storage * 0.6,
        'snow': water_storage * 0.2,
        'groundwater': water_storage * 0.2,
    }
    
    metadata = {
        'description': f'{source} Seasonal Water Storage',
        'temporal_resolution': 'monthly',
        'spatial_resolution': '1 degree',
        'units': 'mm water equivalent',
        'placeholder': True
    }
    
    return HydrologyModel(
        name=source,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        water_storage=water_storage,
        time_stamps=time_stamps,
        components=components,
        metadata=metadata
    )


def load_groundwater_model(
    region: str = 'global',
    data_path: Optional[Path] = None
) -> HydrologyModel:
    """
    Load groundwater storage model.
    
    Args:
        region: Geographic region ('global', 'california', etc.)
        data_path: Path to groundwater data
        
    Returns:
        HydrologyModel instance
    """
    # Create placeholder groundwater model
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    
    # Groundwater storage (simplified)
    # Higher in mid-latitudes, lower in arid regions
    base_storage = 500 * np.exp(-((lat_2d - 30) / 40) ** 2)  # mm
    
    # Add regional variations
    variation = 200 * np.random.randn(*lat_2d.shape)
    groundwater = np.maximum(base_storage + variation, 0)
    
    metadata = {
        'description': f'Groundwater Storage Model - {region}',
        'units': 'mm water equivalent',
        'placeholder': True
    }
    
    return HydrologyModel(
        name=f'Groundwater_{region}',
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        water_storage=groundwater,
        metadata=metadata
    )


def hydrological_correction(
    lat: np.ndarray,
    lon: np.ndarray,
    observation_time: datetime,
    hydrology_model: HydrologyModel,
    reference_time: Optional[datetime] = None
) -> np.ndarray:
    """
    Compute hydrological gravity correction.
    
    Args:
        lat: Station latitudes
        lon: Station longitudes
        observation_time: Time of gravity observation
        hydrology_model: Hydrological model
        reference_time: Reference time (if None, uses mean)
        
    Returns:
        Hydrological correction in mGal
    """
    # Find nearest time index in model
    if hydrology_model.time_stamps is None:
        # Use static model
        water_storage = hydrology_model.water_storage
        ref_storage = water_storage
    else:
        # Find nearest time
        time_diffs = [abs((ts - observation_time).total_seconds()) 
                     for ts in hydrology_model.time_stamps]
        time_idx = np.argmin(time_diffs)
        
        water_storage = hydrology_model.get_storage_at_time(time_idx)
        
        if reference_time is not None:
            # Find reference time
            ref_diffs = [abs((ts - reference_time).total_seconds()) 
                        for ts in hydrology_model.time_stamps]
            ref_idx = np.argmin(ref_diffs)
            ref_storage = hydrology_model.get_storage_at_time(ref_idx)
        else:
            # Use mean as reference
            ref_storage = np.mean(hydrology_model.water_storage, axis=0)
    
    # Compute water storage difference in mm
    delta_storage = water_storage - ref_storage
    
    # Convert to gravity effect using infinite Bouguer slab approximation
    # 1 mm of water = 0.042 µGal (or 0.000042 mGal)
    # More precisely: δg = 2π G ρ_water h
    G = 6.67430e-11  # m³/(kg·s²)
    rho_water = 1000  # kg/m³
    
    # Convert mm to m
    delta_storage_m = delta_storage / 1000.0
    
    # Compute gravity effect in mGal
    correction = 2 * np.pi * G * rho_water * delta_storage_m * 1e5
    
    # Interpolate to station locations (simplified)
    # Would use proper 2D interpolation in production
    correction_at_stations = np.ones_like(lat) * np.mean(correction)
    
    return correction_at_stations


def compute_grace_equivalent(
    gravity_anomaly: np.ndarray,
    lat_grid: np.ndarray,
    lon_grid: np.ndarray
) -> np.ndarray:
    """
    Convert gravity anomaly to water storage equivalent (GRACE-like).
    
    Args:
        gravity_anomaly: Gravity anomaly in mGal
        lat_grid: Latitude grid
        lon_grid: Longitude grid
        
    Returns:
        Water storage in mm
    """
    # Inverse of hydrological correction formula
    # δg = 2π G ρ_water h
    # h = δg / (2π G ρ_water)
    
    G = 6.67430e-11
    rho_water = 1000
    
    # Convert mGal to m/s²
    gravity_ms2 = gravity_anomaly * 1e-5
    
    # Compute water thickness
    thickness_m = gravity_ms2 / (2 * np.pi * G * rho_water)
    
    # Convert to mm
    thickness_mm = thickness_m * 1000
    
    return thickness_mm


def temporal_filtering(
    gravity_series: np.ndarray,
    time_stamps: List[datetime],
    filter_type: str = 'seasonal',
    period_days: Optional[float] = None
) -> np.ndarray:
    """
    Apply temporal filtering to gravity time series.
    
    Args:
        gravity_series: Time series of gravity values
        time_stamps: Corresponding time stamps
        filter_type: 'seasonal', 'trend', 'high_pass', 'low_pass'
        period_days: Period for filtering (days)
        
    Returns:
        Filtered gravity series
    """
    n = len(gravity_series)
    
    if filter_type == 'seasonal':
        # Remove annual and semi-annual signals
        t = np.array([(ts - time_stamps[0]).total_seconds() / (365.25 * 86400) 
                     for ts in time_stamps])
        
        # Fit harmonics
        A = np.column_stack([
            np.ones(n),
            t,
            np.cos(2 * np.pi * t),
            np.sin(2 * np.pi * t),
            np.cos(4 * np.pi * t),
            np.sin(4 * np.pi * t),
        ])
        
        # Least squares fit
        coeffs = np.linalg.lstsq(A, gravity_series, rcond=None)[0]
        
        # Reconstruct without seasonal components
        trend = coeffs[0] + coeffs[1] * t
        return trend
    
    elif filter_type == 'trend':
        # Fit linear trend
        t = np.arange(n)
        coeffs = np.polyfit(t, gravity_series, 1)
        return np.polyval(coeffs, t)
    
    elif filter_type in ['high_pass', 'low_pass']:
        # Simple moving average filter
        if period_days is None:
            period_days = 30
        
        window = max(1, int(period_days / 30))  # Assume monthly data
        
        if filter_type == 'low_pass':
            # Moving average (low-pass)
            filtered = np.convolve(gravity_series, 
                                  np.ones(window) / window, 
                                  mode='same')
        else:
            # High-pass: remove low-pass
            low_pass = np.convolve(gravity_series, 
                                  np.ones(window) / window, 
                                  mode='same')
            filtered = gravity_series - low_pass
        
        return filtered
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")


def estimate_storage_change(
    gravity_change: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    time_period_days: float
) -> Dict[str, np.ndarray]:
    """
    Estimate water storage change from gravity change.
    
    Args:
        gravity_change: Gravity change in mGal
        lat: Latitude points
        lon: Longitude points
        time_period_days: Time period of change (days)
        
    Returns:
        Dictionary with storage change estimates and uncertainties
    """
    # Convert gravity change to storage
    storage_change = compute_grace_equivalent(gravity_change, lat, lon)
    
    # Estimate uncertainty based on time period
    # Longer periods have more cumulative error
    uncertainty = np.abs(storage_change) * 0.1 * np.sqrt(time_period_days / 30)
    
    # Classify change magnitude
    threshold_small = 20  # mm
    threshold_large = 100  # mm
    
    classification = np.zeros_like(storage_change, dtype=int)
    classification[np.abs(storage_change) < threshold_small] = 0  # Insignificant
    classification[(np.abs(storage_change) >= threshold_small) & 
                  (np.abs(storage_change) < threshold_large)] = 1  # Moderate
    classification[np.abs(storage_change) >= threshold_large] = 2  # Large
    
    return {
        'storage_change_mm': storage_change,
        'uncertainty_mm': uncertainty,
        'change_classification': classification,
        'annual_rate_mm_per_year': storage_change * 365.25 / time_period_days
    }
