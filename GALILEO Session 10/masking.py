"""
Ocean/Land Masking and Region Definition

Provides tools for creating and applying geographic masks for
ocean, land, ice sheets, and custom regions.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class OceanLandMask:
    """
    Container for ocean/land mask data.
    
    Attributes:
        name: Mask name
        lat_grid: Latitude grid
        lon_grid: Longitude grid
        mask: Binary mask (0=ocean, 1=land, or more categories)
        categories: Dict mapping mask values to category names
        resolution: Grid resolution in degrees
        metadata: Additional mask metadata
    """
    name: str
    lat_grid: np.ndarray
    lon_grid: np.ndarray
    mask: np.ndarray
    categories: Dict[int, str]
    resolution: float
    metadata: Optional[Dict] = None
    
    def is_land(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Check if points are on land.
        
        Args:
            lat: Latitude points
            lon: Longitude points
            
        Returns:
            Boolean array (True=land)
        """
        # Simplified point-in-grid lookup
        # Would use proper interpolation in production
        
        # Find nearest grid points
        lat_idx = np.searchsorted(self.lat_grid, lat)
        lon_idx = np.searchsorted(self.lon_grid, lon)
        
        # Clip to valid range
        lat_idx = np.clip(lat_idx, 0, len(self.lat_grid) - 1)
        lon_idx = np.clip(lon_idx, 0, len(self.lon_grid) - 1)
        
        # Look up mask values
        mask_values = self.mask[lat_idx, lon_idx]
        
        return mask_values == 1  # 1 = land
    
    def is_ocean(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """Check if points are in ocean."""
        return ~self.is_land(lat, lon)
    
    def get_category(self, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
        """
        Get mask category for points.
        
        Returns:
            Array of category values
        """
        lat_idx = np.searchsorted(self.lat_grid, lat)
        lon_idx = np.searchsorted(self.lon_grid, lon)
        
        lat_idx = np.clip(lat_idx, 0, len(self.lat_grid) - 1)
        lon_idx = np.clip(lon_idx, 0, len(self.lon_grid) - 1)
        
        return self.mask[lat_idx, lon_idx]
    
    def apply_mask(
        self,
        data: np.ndarray,
        mask_value: Union[int, List[int]],
        fill_value: float = np.nan
    ) -> np.ndarray:
        """
        Apply mask to data array.
        
        Args:
            data: Data array (same shape as mask)
            mask_value: Value(s) to mask
            fill_value: Value to use for masked points
            
        Returns:
            Masked data array
        """
        if not isinstance(mask_value, (list, tuple)):
            mask_value = [mask_value]
        
        masked_data = data.copy()
        for val in mask_value:
            masked_data[self.mask == val] = fill_value
        
        return masked_data
    
    def save(self, filepath: Path) -> None:
        """Save mask to file."""
        data = {
            'name': self.name,
            'lat_grid': self.lat_grid.tolist(),
            'lon_grid': self.lon_grid.tolist(),
            'mask': self.mask.tolist(),
            'categories': self.categories,
            'resolution': self.resolution,
            'metadata': self.metadata
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'OceanLandMask':
        """Load mask from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls(
            name=data['name'],
            lat_grid=np.array(data['lat_grid']),
            lon_grid=np.array(data['lon_grid']),
            mask=np.array(data['mask']),
            categories=data['categories'],
            resolution=data['resolution'],
            metadata=data.get('metadata')
        )


def load_ocean_mask(
    resolution: float = 1.0,
    data_path: Optional[Path] = None,
    include_lakes: bool = True
) -> OceanLandMask:
    """
    Load ocean/land mask.
    
    Args:
        resolution: Grid resolution in degrees
        data_path: Path to mask data file
        include_lakes: Include lakes as water
        
    Returns:
        OceanLandMask instance
    """
    if data_path is not None and data_path.exists():
        return OceanLandMask.load(data_path)
    
    # Create placeholder mask
    lat_grid = np.arange(-90, 90 + resolution, resolution)
    lon_grid = np.arange(-180, 180 + resolution, resolution)
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    
    # Create simplified ocean/land mask
    # Ocean = 0, Land = 1, Ice = 2, Lake = 3
    mask = np.ones_like(lat_2d, dtype=int)
    
    # Simple ocean approximation: deep ocean in specific regions
    # Atlantic: lon -70 to 20, lat -60 to 70
    ocean_atlantic = (lon_2d > -70) & (lon_2d < 20) & \
                     (lat_2d > -60) & (lat_2d < 70)
    
    # Pacific: lon 120 to -70 (crossing dateline), lat -60 to 70
    ocean_pacific = ((lon_2d > 120) | (lon_2d < -70)) & \
                    (lat_2d > -60) & (lat_2d < 70)
    
    # Indian: lon 40 to 120, lat -60 to 30
    ocean_indian = (lon_2d > 40) & (lon_2d < 120) & \
                   (lat_2d > -60) & (lat_2d < 30)
    
    # Southern Ocean
    ocean_southern = (lat_2d < -60)
    
    # Arctic Ocean
    ocean_arctic = (lat_2d > 70)
    
    # Apply ocean masks
    mask[ocean_atlantic | ocean_pacific | ocean_indian | 
         ocean_southern | ocean_arctic] = 0
    
    # Ice sheets: Antarctica and Greenland
    ice_antarctica = (lat_2d < -60)
    ice_greenland = (lat_2d > 60) & (lat_2d < 80) & \
                    (lon_2d > -60) & (lon_2d < -20)
    
    mask[ice_antarctica | ice_greenland] = 2
    
    # Add some lakes if requested
    if include_lakes:
        # Great Lakes
        great_lakes = (lat_2d > 41) & (lat_2d < 49) & \
                     (lon_2d > -93) & (lon_2d < -76)
        mask[great_lakes] = 3
        
        # Caspian Sea
        caspian = (lat_2d > 36) & (lat_2d < 47) & \
                 (lon_2d > 46) & (lon_2d < 55)
        mask[caspian] = 3
    
    categories = {
        0: 'ocean',
        1: 'land',
        2: 'ice',
        3: 'lake'
    }
    
    metadata = {
        'description': 'Global Ocean/Land Mask',
        'resolution_deg': resolution,
        'include_lakes': include_lakes,
        'placeholder': True
    }
    
    return OceanLandMask(
        name='global_mask',
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        mask=mask,
        categories=categories,
        resolution=resolution,
        metadata=metadata
    )


def create_region_mask(
    lat_range: Tuple[float, float],
    lon_range: Tuple[float, float],
    resolution: float = 1.0,
    region_name: str = 'custom_region'
) -> OceanLandMask:
    """
    Create a mask for a specific geographic region.
    
    Args:
        lat_range: (min_lat, max_lat) in degrees
        lon_range: (min_lon, max_lon) in degrees
        resolution: Grid resolution in degrees
        region_name: Name for the region
        
    Returns:
        OceanLandMask with region defined
    """
    lat_grid = np.arange(-90, 90 + resolution, resolution)
    lon_grid = np.arange(-180, 180 + resolution, resolution)
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    
    # Create mask: 0=outside region, 1=inside region
    mask = np.zeros_like(lat_2d, dtype=int)
    
    # Define region
    in_region = (lat_2d >= lat_range[0]) & (lat_2d <= lat_range[1]) & \
                (lon_2d >= lon_range[0]) & (lon_2d <= lon_range[1])
    
    mask[in_region] = 1
    
    categories = {
        0: 'outside',
        1: 'inside'
    }
    
    metadata = {
        'description': f'Region mask: {region_name}',
        'lat_range': lat_range,
        'lon_range': lon_range,
    }
    
    return OceanLandMask(
        name=region_name,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        mask=mask,
        categories=categories,
        resolution=resolution,
        metadata=metadata
    )


def create_polygon_mask(
    polygon_vertices: List[Tuple[float, float]],
    resolution: float = 1.0,
    region_name: str = 'polygon_region'
) -> OceanLandMask:
    """
    Create a mask from polygon vertices.
    
    Args:
        polygon_vertices: List of (lat, lon) tuples defining polygon
        resolution: Grid resolution in degrees
        region_name: Name for the region
        
    Returns:
        OceanLandMask with polygon region defined
    """
    lat_grid = np.arange(-90, 90 + resolution, resolution)
    lon_grid = np.arange(-180, 180 + resolution, resolution)
    lat_2d, lon_2d = np.meshgrid(lat_grid, lon_grid, indexing='ij')
    
    # Convert vertices to arrays
    vertices = np.array(polygon_vertices)
    
    # Simple point-in-polygon test (ray casting algorithm)
    mask = np.zeros_like(lat_2d, dtype=int)
    
    for i in range(len(lat_grid)):
        for j in range(len(lon_grid)):
            point = (lat_2d[i, j], lon_2d[i, j])
            if point_in_polygon(point, vertices):
                mask[i, j] = 1
    
    categories = {
        0: 'outside',
        1: 'inside'
    }
    
    metadata = {
        'description': f'Polygon mask: {region_name}',
        'vertices': polygon_vertices,
    }
    
    return OceanLandMask(
        name=region_name,
        lat_grid=lat_grid,
        lon_grid=lon_grid,
        mask=mask,
        categories=categories,
        resolution=resolution,
        metadata=metadata
    )


def point_in_polygon(point: Tuple[float, float], 
                    vertices: np.ndarray) -> bool:
    """
    Check if point is inside polygon using ray casting algorithm.
    
    Args:
        point: (lat, lon) tuple
        vertices: Nx2 array of polygon vertices
        
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(vertices)
    inside = False
    
    p1x, p1y = vertices[0]
    for i in range(1, n + 1):
        p2x, p2y = vertices[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def combine_masks(
    masks: List[OceanLandMask],
    operation: str = 'union',
    result_name: str = 'combined_mask'
) -> OceanLandMask:
    """
    Combine multiple masks using boolean operations.
    
    Args:
        masks: List of OceanLandMask objects
        operation: 'union', 'intersection', 'difference'
        result_name: Name for resulting mask
        
    Returns:
        Combined OceanLandMask
    """
    if not masks:
        raise ValueError("Must provide at least one mask")
    
    # Use first mask as template
    base_mask = masks[0]
    result = base_mask.mask.copy()
    
    if operation == 'union':
        # Union: any mask has 1 -> result is 1
        for mask in masks[1:]:
            result = np.maximum(result, mask.mask)
    
    elif operation == 'intersection':
        # Intersection: all masks have 1 -> result is 1
        for mask in masks[1:]:
            result = np.minimum(result, mask.mask)
    
    elif operation == 'difference':
        # Difference: first mask minus subsequent masks
        for mask in masks[1:]:
            result[mask.mask == 1] = 0
    
    else:
        raise ValueError(f"Unknown operation: {operation}")
    
    return OceanLandMask(
        name=result_name,
        lat_grid=base_mask.lat_grid,
        lon_grid=base_mask.lon_grid,
        mask=result,
        categories=base_mask.categories,
        resolution=base_mask.resolution,
        metadata={'operation': operation, 'source_masks': [m.name for m in masks]}
    )


def mask_statistics(
    data: np.ndarray,
    mask: OceanLandMask,
    category: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute statistics for masked data.
    
    Args:
        data: Data array
        mask: OceanLandMask
        category: Specific category to analyze (None = all)
        
    Returns:
        Dictionary of statistics
    """
    if category is not None:
        selected_data = data[mask.mask == category]
    else:
        selected_data = data[mask.mask > 0]  # All non-ocean
    
    # Remove NaN values
    valid_data = selected_data[~np.isnan(selected_data)]
    
    if len(valid_data) == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan
        }
    
    return {
        'count': len(valid_data),
        'mean': np.mean(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'median': np.median(valid_data),
        'percentile_25': np.percentile(valid_data, 25),
        'percentile_75': np.percentile(valid_data, 75),
    }


def distance_to_coast(
    lat: np.ndarray,
    lon: np.ndarray,
    ocean_mask: OceanLandMask
) -> np.ndarray:
    """
    Compute distance to nearest coastline.
    
    Args:
        lat: Latitude points
        lon: Longitude points
        ocean_mask: Ocean/land mask
        
    Returns:
        Distance to coast in km
    """
    # Find coastline (transition between ocean and land)
    mask_grad_lat = np.abs(np.gradient(ocean_mask.mask, axis=0))
    mask_grad_lon = np.abs(np.gradient(ocean_mask.mask, axis=1))
    coastline = (mask_grad_lat + mask_grad_lon) > 0
    
    # Get coastline coordinates
    coast_lat, coast_lon = np.where(coastline)
    coast_lat_deg = ocean_mask.lat_grid[coast_lat]
    coast_lon_deg = ocean_mask.lon_grid[coast_lon]
    
    # Compute distances (simplified great circle approximation)
    R_earth = 6371.0  # km
    
    distances = np.zeros_like(lat)
    for i, (pt_lat, pt_lon) in enumerate(zip(lat.flat, lon.flat)):
        # Compute distance to all coast points
        dlat = np.radians(coast_lat_deg - pt_lat)
        dlon = np.radians(coast_lon_deg - pt_lon)
        
        a = np.sin(dlat/2)**2 + \
            np.cos(np.radians(pt_lat)) * np.cos(np.radians(coast_lat_deg)) * \
            np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        dist = R_earth * c
        distances.flat[i] = np.min(dist)
    
    return distances.reshape(lat.shape)
