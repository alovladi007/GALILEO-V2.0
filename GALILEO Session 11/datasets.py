"""
Regression Datasets Module
===========================
Provides synthetic test datasets and gold standard outputs for benchmarking.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import json
import pickle


class RegressionDatasets:
    """Manager for regression test datasets and gold outputs."""
    
    def __init__(self, data_dir: Path = Path("bench/datasets")):
        """
        Initialize dataset manager.
        
        Parameters
        ----------
        data_dir : Path
            Directory containing test datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.gold_dir = Path("bench/gold_outputs")
        self.gold_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize datasets on first use
        self._initialize_datasets()
    
    def _initialize_datasets(self):
        """Initialize/generate synthetic test datasets."""
        # Check if datasets exist, generate if not
        if not (self.data_dir / "point_source.npy").exists():
            self._generate_all_datasets()
    
    def _generate_all_datasets(self):
        """Generate all synthetic test datasets."""
        print("Generating synthetic regression datasets...")
        
        # Spatial resolution tests
        self._generate_point_source()
        self._generate_frequency_test()
        self._generate_twin_anomalies()
        self._generate_overlapping_anomalies()
        
        # Localization tests
        self._generate_localization_test()
        self._generate_boundary_test()
        self._generate_multi_target_test()
        self._generate_depth_test()
        
        # Performance tests
        self._generate_standard_model()
        self._generate_inversion_test()
        self._generate_ml_test_input()
        
        print("✓ All datasets generated")
    
    # =========================================================================
    # SPATIAL RESOLUTION DATASETS
    # =========================================================================
    
    def _generate_point_source(self):
        """Generate point source response for PSF characterization."""
        size = 128
        center = size // 2
        
        # Create Gaussian point source response
        y, x = np.ogrid[-center:size-center, -center:size-center]
        sigma = 5.0  # pixels
        psf = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        
        # Add realistic noise
        noise = np.random.randn(size, size) * 0.05
        point_source = psf + noise
        
        np.save(self.data_dir / "point_source.npy", point_source)
        
        # Generate gold PSF (ideal)
        gold_psf = psf[center-32:center+32, center-32:center+32]
        np.save(self.gold_dir / "psf.npy", gold_psf)
    
    def _generate_frequency_test(self):
        """Generate frequency response test patterns."""
        size = 256
        frequencies = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # cycles per pixel
        
        input_patterns = []
        output_patterns = []
        
        for freq in frequencies:
            # Generate sinusoidal pattern
            x = np.arange(size)
            y = np.arange(size)
            X, Y = np.meshgrid(x, y)
            
            input_pattern = np.sin(2 * np.pi * freq * X)
            
            # Simulate system response (low-pass filter)
            sigma = 1.0 / freq
            from scipy.ndimage import gaussian_filter
            output_pattern = gaussian_filter(input_pattern, sigma)
            
            input_patterns.append(input_pattern)
            output_patterns.append(output_pattern)
        
        freq_data = {
            'frequencies': frequencies,
            'input': np.array(input_patterns),
            'output': np.array(output_patterns)
        }
        
        with open(self.data_dir / "frequency_test.pkl", 'wb') as f:
            pickle.dump(freq_data, f)
    
    def _generate_twin_anomalies(self):
        """Generate closely-spaced twin anomalies."""
        size = 128
        
        # Two Gaussian anomalies separated by 10 pixels
        y, x = np.ogrid[:size, :size]
        
        center1 = (size // 2 - 5, size // 2)
        center2 = (size // 2 + 5, size // 2)
        
        sigma = 8.0
        anomaly1 = np.exp(-((x - center1[1])**2 + (y - center1[0])**2) / (2 * sigma**2))
        anomaly2 = np.exp(-((x - center2[1])**2 + (y - center2[0])**2) / (2 * sigma**2))
        
        twin_data = anomaly1 + anomaly2
        
        # Add noise
        noise = np.random.randn(size, size) * 0.1
        twin_data += noise
        
        np.save(self.data_dir / "twin_anomalies.npy", twin_data)
        
        # Gold output: separated anomalies
        gold = np.stack([anomaly1, anomaly2], axis=0)
        np.save(self.gold_dir / "twin_anomalies.npy", gold)
    
    def _generate_overlapping_anomalies(self):
        """Generate overlapping anomalies."""
        size = 128
        y, x = np.ogrid[:size, :size]
        
        # Three overlapping Gaussians
        centers = [(40, 40), (60, 50), (50, 70)]
        sigmas = [12.0, 15.0, 10.0]
        amplitudes = [1.0, 0.8, 1.2]
        
        overlap_data = np.zeros((size, size))
        separated_components = []
        
        for center, sigma, amp in zip(centers, sigmas, amplitudes):
            component = amp * np.exp(
                -((x - center[1])**2 + (y - center[0])**2) / (2 * sigma**2)
            )
            overlap_data += component
            separated_components.append(component)
        
        # Add noise
        noise = np.random.randn(size, size) * 0.05
        overlap_data += noise
        
        np.save(self.data_dir / "overlapping_anomalies.npy", overlap_data)
        
        # Gold output: separated components
        gold = np.array(separated_components)
        np.save(self.gold_dir / "separated_anomalies.npy", gold)
    
    # =========================================================================
    # LOCALIZATION DATASETS
    # =========================================================================
    
    def _generate_localization_test(self):
        """Generate localization test with known centroids."""
        size = 256
        y, x = np.ogrid[:size, :size]
        
        # Five anomalies at known positions
        true_centroids = np.array([
            [50, 50],
            [50, 150],
            [150, 50],
            [150, 150],
            [100, 100]
        ])
        
        anomaly_data = np.zeros((size, size))
        
        for center in true_centroids:
            sigma = 10.0
            anomaly = np.exp(
                -((x - center[1])**2 + (y - center[0])**2) / (2 * sigma**2)
            )
            anomaly_data += anomaly
        
        # Add noise
        noise = np.random.randn(size, size) * 0.05
        anomaly_data += noise
        
        np.save(self.data_dir / "localization_test.npy", anomaly_data)
        np.save(self.gold_dir / "centroids.npy", true_centroids)
    
    def _generate_boundary_test(self):
        """Generate boundary detection test."""
        size = 256
        
        # Create step function boundaries
        field = np.zeros((size, size))
        
        # Rectangular regions with different values
        field[50:100, 50:150] = 1.0
        field[120:200, 80:180] = 0.7
        field[80:120, 180:220] = 1.2
        
        # Smooth boundaries
        from scipy.ndimage import gaussian_filter
        field = gaussian_filter(field, sigma=2.0)
        
        # Add noise
        noise = np.random.randn(size, size) * 0.05
        field += noise
        
        np.save(self.data_dir / "boundary_test.npy", field)
        
        # Gold boundaries (before smoothing)
        boundaries = [
            np.array([[50, 50], [50, 150], [100, 150], [100, 50]]),
            np.array([[120, 80], [120, 180], [200, 180], [200, 80]]),
            np.array([[80, 180], [80, 220], [120, 220], [120, 180]])
        ]
        
        with open(self.gold_dir / "boundaries.pkl", 'wb') as f:
            pickle.dump(boundaries, f)
    
    def _generate_multi_target_test(self):
        """Generate multi-target localization test."""
        size = 512
        y, x = np.ogrid[:size, :size]
        
        # Random positions
        np.random.seed(42)
        n_targets = 15
        true_positions = np.random.rand(n_targets, 2) * (size - 100) + 50
        
        multi_data = np.zeros((size, size))
        
        for pos in true_positions:
            sigma = np.random.rand() * 5 + 8  # 8-13 pixels
            amplitude = np.random.rand() * 0.5 + 0.7  # 0.7-1.2
            
            anomaly = amplitude * np.exp(
                -((x - pos[1])**2 + (y - pos[0])**2) / (2 * sigma**2)
            )
            multi_data += anomaly
        
        # Add significant noise
        noise = np.random.randn(size, size) * 0.1
        multi_data += noise
        
        np.save(self.data_dir / "multi_target_test.npy", multi_data)
        np.save(self.gold_dir / "multi_targets.npy", true_positions)
    
    def _generate_depth_test(self):
        """Generate depth estimation test."""
        size = 256
        y, x = np.ogrid[:size, :size]
        
        # Anomalies at different depths
        positions = np.array([
            [80, 80],
            [80, 180],
            [180, 80],
            [180, 180]
        ])
        
        true_depths = np.array([5.0, 10.0, 15.0, 20.0])  # km
        
        depth_field = np.zeros((size, size))
        
        for pos, depth in zip(positions, true_depths):
            # Width increases with depth
            sigma = depth * 0.8  # pixels
            amplitude = 1.0 / depth  # Amplitude decreases with depth
            
            anomaly = amplitude * np.exp(
                -((x - pos[1])**2 + (y - pos[0])**2) / (2 * sigma**2)
            )
            depth_field += anomaly
        
        # Add noise
        noise = np.random.randn(size, size) * 0.02
        depth_field += noise
        
        depth_data = {
            'field': depth_field,
            'positions': positions
        }
        
        with open(self.data_dir / "depth_test.pkl", 'wb') as f:
            pickle.dump(depth_data, f)
        
        np.save(self.gold_dir / "depths.npy", true_depths)
    
    # =========================================================================
    # PERFORMANCE DATASETS
    # =========================================================================
    
    def _generate_standard_model(self):
        """Generate standard model for performance testing."""
        model = {
            'size': 64,
            'density_model': np.random.randn(64, 64, 64) * 100 + 2700,  # kg/m³
            'grid_spacing': 1.0,  # km
        }
        
        with open(self.data_dir / "standard_model.pkl", 'wb') as f:
            pickle.dump(model, f)
    
    def _generate_inversion_test(self):
        """Generate inversion test case."""
        data = {
            'size': 50,
            'observations': np.random.randn(2500) * 10,  # mGal
            'max_iter': 20,
            'tolerance': 1e-6
        }
        
        with open(self.data_dir / "inversion_test.pkl", 'wb') as f:
            pickle.dump(data, f)
    
    def _generate_ml_test_input(self):
        """Generate ML inference test input."""
        test_input = {
            'batch_size': 32,
            'shape': (64, 64),
            'n_channels': 1
        }
        
        with open(self.data_dir / "ml_test_input.pkl", 'wb') as f:
            pickle.dump(test_input, f)
    
    # =========================================================================
    # DATA LOADING METHODS
    # =========================================================================
    
    def load_point_source(self) -> np.ndarray:
        """Load point source test data."""
        return np.load(self.data_dir / "point_source.npy")
    
    def load_frequency_test(self) -> Dict:
        """Load frequency response test data."""
        with open(self.data_dir / "frequency_test.pkl", 'rb') as f:
            return pickle.load(f)
    
    def load_twin_anomalies(self) -> np.ndarray:
        """Load twin anomaly test data."""
        return np.load(self.data_dir / "twin_anomalies.npy")
    
    def load_overlapping_anomalies(self) -> np.ndarray:
        """Load overlapping anomalies test data."""
        return np.load(self.data_dir / "overlapping_anomalies.npy")
    
    def load_localization_test(self) -> np.ndarray:
        """Load localization test data."""
        return np.load(self.data_dir / "localization_test.npy")
    
    def load_boundary_test(self) -> np.ndarray:
        """Load boundary detection test data."""
        return np.load(self.data_dir / "boundary_test.npy")
    
    def load_multi_target_test(self) -> np.ndarray:
        """Load multi-target test data."""
        return np.load(self.data_dir / "multi_target_test.npy")
    
    def load_depth_test(self) -> Dict:
        """Load depth estimation test data."""
        with open(self.data_dir / "depth_test.pkl", 'rb') as f:
            return pickle.load(f)
    
    def load_standard_model(self) -> Dict:
        """Load standard performance test model."""
        with open(self.data_dir / "standard_model.pkl", 'rb') as f:
            return pickle.load(f)
    
    def load_inversion_test(self) -> Dict:
        """Load inversion test data."""
        with open(self.data_dir / "inversion_test.pkl", 'rb') as f:
            return pickle.load(f)
    
    def load_ml_test_input(self) -> Dict:
        """Load ML test input."""
        with open(self.data_dir / "ml_test_input.pkl", 'rb') as f:
            return pickle.load(f)
    
    def load_gold_output(self, name: str) -> np.ndarray:
        """
        Load gold standard output.
        
        Parameters
        ----------
        name : str
            Name of gold output: 'psf', 'centroids', 'boundaries', etc.
            
        Returns
        -------
        gold : np.ndarray or list
            Gold standard output
        """
        gold_file = self.gold_dir / f"{name}.npy"
        pkl_file = self.gold_dir / f"{name}.pkl"
        
        if gold_file.exists():
            return np.load(gold_file)
        elif pkl_file.exists():
            with open(pkl_file, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Gold output '{name}' not found")
    
    def get_dataset_info(self) -> Dict:
        """Get information about all available datasets."""
        info = {
            'spatial_resolution': [
                'point_source',
                'frequency_test',
                'twin_anomalies',
                'overlapping_anomalies'
            ],
            'localization': [
                'localization_test',
                'boundary_test',
                'multi_target_test',
                'depth_test'
            ],
            'performance': [
                'standard_model',
                'inversion_test',
                'ml_test_input'
            ]
        }
        
        return info
    
    def verify_datasets(self) -> bool:
        """Verify all datasets exist and are valid."""
        print("Verifying regression datasets...")
        
        required_files = [
            "point_source.npy",
            "frequency_test.pkl",
            "twin_anomalies.npy",
            "overlapping_anomalies.npy",
            "localization_test.npy",
            "boundary_test.npy",
            "multi_target_test.npy",
            "depth_test.pkl",
            "standard_model.pkl",
            "inversion_test.pkl",
            "ml_test_input.pkl"
        ]
        
        all_exist = True
        for fname in required_files:
            fpath = self.data_dir / fname
            if fpath.exists():
                print(f"  ✓ {fname}")
            else:
                print(f"  ✗ {fname} - MISSING")
                all_exist = False
        
        return all_exist


def create_sample_datasets():
    """Standalone function to create sample datasets."""
    datasets = RegressionDatasets()
    datasets._generate_all_datasets()
    
    if datasets.verify_datasets():
        print("\n✅ All regression datasets created successfully!")
        return True
    else:
        print("\n❌ Some datasets are missing")
        return False


if __name__ == '__main__':
    create_sample_datasets()
