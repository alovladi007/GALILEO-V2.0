"""
Geophysics Module Test Suite

Quick tests to verify module installation and basic functionality.
"""

import sys
sys.path.insert(0, '/home/claude')

import numpy as np
from datetime import datetime


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from geophysics import (
            load_egm96, load_egm2008, compute_gravity_anomaly,
            load_crust1, terrain_correction, bouguer_correction,
            load_seasonal_water, hydrological_correction,
            load_ocean_mask, create_region_mask,
            setup_joint_inversion, integrate_gravity_seismic
        )
        print("  ✓ All imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_gravity_fields():
    """Test gravity field models."""
    print("\nTesting gravity field models...")
    
    try:
        from geophysics import load_egm96, compute_gravity_anomaly
        
        # Load model
        egm96 = load_egm96()
        assert egm96.name == 'EGM96'
        assert egm96.degree_max == 360
        
        # Test geoid computation
        lat = np.array([40.0, 41.0])
        lon = np.array([-75.0, -74.0])
        geoid = egm96.compute_geoid_height(lat, lon, max_degree=20)
        assert geoid.shape == lat.shape
        
        # Test anomaly computation
        observed_g = np.array([980200, 980150])
        anomaly, components = compute_gravity_anomaly(
            lat, lon, observed_g, egm96, correction_type='free_air'
        )
        assert anomaly.shape == lat.shape
        assert 'normal_gravity' in components
        
        print("  ✓ Gravity field models working")
        return True
    except Exception as e:
        print(f"  ✗ Gravity field test failed: {e}")
        return False


def test_crustal_models():
    """Test crustal density models."""
    print("\nTesting crustal models...")
    
    try:
        from geophysics import load_crust1, bouguer_correction
        
        # Load model
        crust = load_crust1()
        assert crust.name == 'CRUST1.0'
        
        # Test density query
        lat = np.array([40.0])
        lon = np.array([-120.0])
        density = crust.get_density_at_depth(lat, lon, depth=5000)
        assert density.shape == lat.shape
        assert density[0] > 2000  # Reasonable crustal density
        
        # Test Bouguer correction
        elevation = np.array([1000.0])
        correction = bouguer_correction(elevation)
        assert correction[0] > 0  # Positive correction for elevation
        
        print("  ✓ Crustal models working")
        return True
    except Exception as e:
        print(f"  ✗ Crustal model test failed: {e}")
        return False


def test_hydrology():
    """Test hydrology models."""
    print("\nTesting hydrology models...")
    
    try:
        from geophysics import load_seasonal_water, hydrological_correction
        
        # Load model
        hydro = load_seasonal_water(source='GLDAS')
        assert hydro.name == 'GLDAS'
        assert len(hydro.time_stamps) == 12  # Monthly data
        
        # Test correction
        lat = np.array([40.0])
        lon = np.array([-120.0])
        obs_time = datetime(2024, 6, 15)
        
        correction = hydrological_correction(lat, lon, obs_time, hydro)
        assert correction.shape == lat.shape
        
        print("  ✓ Hydrology models working")
        return True
    except Exception as e:
        print(f"  ✗ Hydrology test failed: {e}")
        return False


def test_masking():
    """Test ocean/land masking."""
    print("\nTesting masking...")
    
    try:
        from geophysics import load_ocean_mask, create_region_mask
        
        # Load global mask
        mask = load_ocean_mask(resolution=1.0)
        assert mask.name == 'global_mask'
        assert len(mask.categories) > 0
        
        # Test point queries
        lat = np.array([40.0, 0.0])  # Land, ocean
        lon = np.array([-100.0, -150.0])
        
        is_land = mask.is_land(lat, lon)
        assert is_land.shape == lat.shape
        
        # Test custom region
        region = create_region_mask(
            lat_range=(35, 45),
            lon_range=(-125, -115),
            region_name='test_region'
        )
        assert region.name == 'test_region'
        
        print("  ✓ Masking working")
        return True
    except Exception as e:
        print(f"  ✗ Masking test failed: {e}")
        return False


def test_joint_inversion():
    """Test joint inversion setup."""
    print("\nTesting joint inversion...")
    
    try:
        from geophysics import setup_joint_inversion, integrate_gravity_seismic
        
        # Setup model
        lat = np.array([40.0, 41.0])
        lon = np.array([-120.0, -119.0])
        gravity = np.array([980200, 980150])
        
        joint_model = setup_joint_inversion(gravity, lat, lon)
        assert joint_model.name == 'joint_model'
        assert 'gravity' in joint_model.data_types
        
        # Add seismic data
        velocity = np.array([5000, 5500])
        joint_model = integrate_gravity_seismic(
            joint_model, velocity,
            seismic_type='velocity',
            coupling_type='petrophysical'
        )
        assert 'seismic_velocity' in joint_model.data_types
        
        print("  ✓ Joint inversion working")
        return True
    except Exception as e:
        print(f"  ✗ Joint inversion test failed: {e}")
        return False


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*60)
    print("GEOPHYSICS MODULE TEST SUITE")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Gravity Fields", test_gravity_fields),
        ("Crustal Models", test_crustal_models),
        ("Hydrology", test_hydrology),
        ("Masking", test_masking),
        ("Joint Inversion", test_joint_inversion),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    n_passed = sum(1 for _, result in results if result)
    n_total = len(results)
    
    for name, result in results:
        status = "PASS ✓" if result else "FAIL ✗"
        print(f"  {name:20s}: {status}")
    
    print(f"\n  Total: {n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n  ✓ All tests PASSED!")
        return 0
    else:
        print(f"\n  ✗ {n_total - n_passed} test(s) FAILED")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
