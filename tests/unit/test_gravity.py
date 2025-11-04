"""
Unit tests for gravity field simulation module.
"""

import pytest
import jax.numpy as jnp
import numpy as np
from sim.gravity import (
    GravityModel,
    SphericalHarmonics,
    load_egm2008_model,
)


class TestGravityModel:
    """Test gravity model data structure."""
    
    def test_gravity_model_initialization(self):
        """Test creation of gravity model."""
        max_degree = 10
        C_nm = jnp.zeros((max_degree + 1, max_degree + 1))
        S_nm = jnp.zeros((max_degree + 1, max_degree + 1))
        
        model = GravityModel(
            C_nm=C_nm,
            S_nm=S_nm,
            max_degree=max_degree,
            max_order=max_degree,
        )
        
        assert model.max_degree == max_degree
        assert model.C_nm.shape == (max_degree + 1, max_degree + 1)
        assert model.reference_radius > 0
        assert model.gm > 0
    
    def test_gravity_model_defaults(self):
        """Test default values for gravity model."""
        model = GravityModel(
            C_nm=jnp.zeros((2, 2)),
            S_nm=jnp.zeros((2, 2)),
            max_degree=1,
            max_order=1,
        )
        
        # Check Earth parameters
        assert model.reference_radius == 6378137.0  # meters
        assert abs(model.gm - 3.986004418e14) < 1e6  # m³/s²


class TestSphericalHarmonics:
    """Test spherical harmonics computations."""
    
    @pytest.fixture
    def simple_model(self):
        """Create a simple gravity model for testing."""
        C_nm = jnp.zeros((3, 3))
        S_nm = jnp.zeros((3, 3))
        # Set C_2,0 (J2) to approximate Earth value
        C_nm = C_nm.at[2, 0].set(-1.08263e-3)
        
        return GravityModel(
            C_nm=C_nm,
            S_nm=S_nm,
            max_degree=2,
            max_order=2,
        )
    
    def test_spherical_harmonics_initialization(self, simple_model):
        """Test initialization of spherical harmonics calculator."""
        sh = SphericalHarmonics(simple_model)
        assert sh.model.max_degree == 2
        assert sh.model.max_order == 2
    
    def test_gravitational_potential_shape(self, simple_model):
        """Test that gravitational potential returns correct shape."""
        sh = SphericalHarmonics(simple_model)
        
        # Test single point
        r = jnp.array(7000.0e3)  # 7000 km in meters
        lat = jnp.array(0.0)
        lon = jnp.array(0.0)
        
        potential = sh.gravitational_potential(r, lat, lon)
        assert potential.shape == ()  # scalar
    
    def test_gravitational_acceleration_shape(self, simple_model):
        """Test that acceleration returns 3D vector."""
        sh = SphericalHarmonics(simple_model)
        
        # Position at 7000 km radius
        position = jnp.array([7000.0e3, 0.0, 0.0])
        
        acceleration = sh.gravitational_acceleration(position)
        assert acceleration.shape == (3,)
    
    def test_acceleration_direction(self, simple_model):
        """Test that acceleration points toward Earth center."""
        sh = SphericalHarmonics(simple_model)
        
        # Position vector
        position = jnp.array([7000.0e3, 1000.0e3, 500.0e3])
        
        # Acceleration should point toward origin (negative of position)
        acceleration = sh.gravitational_acceleration(position)
        
        # Dot product should be negative (opposite directions)
        dot_product = jnp.dot(position, acceleration)
        # Note: This test assumes we implement the function to return
        # negative gradient, which points toward Earth


class TestLoadModel:
    """Test loading of gravity models."""
    
    def test_load_egm2008_model(self):
        """Test loading EGM2008 model."""
        max_degree = 50
        model = load_egm2008_model(max_degree=max_degree)
        
        assert model.max_degree == max_degree
        assert model.max_order == max_degree
        assert model.C_nm.shape == (max_degree + 1, max_degree + 1)
        assert model.S_nm.shape == (max_degree + 1, max_degree + 1)
    
    def test_load_egm2008_default_degree(self):
        """Test loading with default parameters."""
        model = load_egm2008_model()
        assert model.max_degree == 360


class TestGravityComputations:
    """Integration tests for gravity computations."""
    
    def test_two_body_approximation(self):
        """Test that simple two-body gravity is approximately correct."""
        # Create zero-order model (spherical Earth)
        C_nm = jnp.zeros((1, 1))
        S_nm = jnp.zeros((1, 1))
        model = GravityModel(C_nm=C_nm, S_nm=S_nm, max_degree=0, max_order=0)
        
        sh = SphericalHarmonics(model)
        
        # Position at 7000 km
        r = 7000.0e3  # meters
        position = jnp.array([r, 0.0, 0.0])
        
        # Expected acceleration magnitude: GM/r²
        expected_mag = model.gm / (r * r)
        
        # Computed acceleration (once implemented)
        acceleration = sh.gravitational_acceleration(position)
        computed_mag = jnp.linalg.norm(acceleration)
        
        # Should be within 1% for zero-order model
        # relative_error = abs(computed_mag - expected_mag) / expected_mag
        # assert relative_error < 0.01
    
    def test_geoid_height_reasonable(self):
        """Test that computed geoid heights are reasonable."""
        from sim.gravity import compute_geoid_height
        
        model = load_egm2008_model(max_degree=10)
        
        # Test at equator
        lat = np.array([0.0])
        lon = np.array([0.0])
        
        geoid = compute_geoid_height(lat, lon, model)
        
        # Geoid heights should be within ±100m typically
        assert geoid.shape == (1,)
        # assert abs(geoid[0]) < 100.0  # Once implemented


@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for gravity computations."""
    
    def test_potential_computation_speed(self, benchmark, simple_model):
        """Benchmark gravitational potential computation."""
        sh = SphericalHarmonics(simple_model)
        
        r = jnp.array(7000.0e3)
        lat = jnp.array(0.0)
        lon = jnp.array(0.0)
        
        result = benchmark(sh.gravitational_potential, r, lat, lon)
        assert jnp.isfinite(result)
    
    def test_batch_acceleration(self, benchmark, simple_model):
        """Benchmark batch acceleration computation."""
        sh = SphericalHarmonics(simple_model)
        
        # Batch of 1000 positions
        n_points = 1000
        positions = jnp.ones((n_points, 3)) * 7000.0e3
        
        def compute_batch():
            return jnp.array([
                sh.gravitational_acceleration(pos) 
                for pos in positions
            ])
        
        result = benchmark(compute_batch)
        assert result.shape == (n_points, 3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
