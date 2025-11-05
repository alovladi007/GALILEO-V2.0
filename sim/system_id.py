"""
System Identification and Parameter Estimation

Estimate physical parameters from measurement residuals:
- Drag coefficient (CD)
- Solar radiation pressure coefficient (CR)
- Empirical accelerations
- Residual model fitting and validation
"""

import numpy as np
from scipy import optimize, stats
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import warnings


@dataclass
class SystemIDResult:
    """Results from system identification."""
    parameters: np.ndarray  # Estimated parameters
    covariance: np.ndarray  # Parameter covariance matrix
    residuals: np.ndarray  # Post-fit residuals
    cost: float  # Final cost function value
    success: bool  # Optimization success flag
    message: str  # Optimization message
    iterations: int  # Number of iterations
    
    def parameter_uncertainties(self) -> np.ndarray:
        """Get 1-sigma parameter uncertainties."""
        return np.sqrt(np.diag(self.covariance))
    
    def correlation_matrix(self) -> np.ndarray:
        """Get parameter correlation matrix."""
        std = self.parameter_uncertainties()
        return self.covariance / np.outer(std, std)


class DragCoefficientEstimator:
    """
    Estimate atmospheric drag coefficient from orbit residuals.
    
    Models drag acceleration as:
        a_drag = -0.5 * CD * (A/m) * ρ * v_rel * |v_rel|
    
    where CD is the drag coefficient to be estimated.
    """
    
    def __init__(self, area_to_mass: float, initial_cd: float = 2.2):
        """
        Initialize drag coefficient estimator.
        
        Parameters
        ----------
        area_to_mass : float
            Cross-sectional area to mass ratio (m²/kg)
        initial_cd : float
            Initial guess for drag coefficient (typically 2.0-2.5)
        """
        self.area_to_mass = area_to_mass
        self.initial_cd = initial_cd
    
    def estimate(self, times: np.ndarray, residuals: np.ndarray,
                 density: np.ndarray, velocity: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> SystemIDResult:
        """
        Estimate drag coefficient from residuals.
        
        Parameters
        ----------
        times : np.ndarray
            Time epochs
        residuals : np.ndarray
            Position/velocity residuals (N x 3)
        density : np.ndarray
            Atmospheric density at each epoch (kg/m³)
        velocity : np.ndarray
            Relative velocity magnitude (m/s)
        weights : np.ndarray, optional
            Observation weights
        
        Returns
        -------
        SystemIDResult
            Estimation results
        """
        if weights is None:
            weights = np.ones(len(times))
        
        def cost_function(cd):
            """Cost function for drag coefficient estimation."""
            # Compute drag acceleration magnitude
            drag_accel = 0.5 * cd[0] * self.area_to_mass * density * velocity**2
            
            # Project onto velocity direction (assuming residuals are velocity)
            predicted = -drag_accel[:, np.newaxis] * (velocity[:, np.newaxis] / 
                                                       np.linalg.norm(velocity[:, np.newaxis], axis=1, keepdims=True))
            
            # Weighted residuals
            weighted_res = weights[:, np.newaxis] * (residuals - predicted)
            
            return np.sum(weighted_res**2)
        
        # Optimize
        result = optimize.minimize(
            cost_function,
            x0=[self.initial_cd],
            method='L-BFGS-B',
            bounds=[(0.5, 5.0)]  # Physical bounds on CD
        )
        
        # Compute covariance using Hessian approximation
        cd_opt = result.x[0]
        
        # Numerical Hessian
        eps = 1e-6
        h11 = (cost_function([cd_opt + eps]) - 2*cost_function([cd_opt]) + 
               cost_function([cd_opt - eps])) / eps**2
        
        # Covariance
        cov = np.array([[2.0 / max(h11, 1e-10)]])
        
        # Post-fit residuals
        drag_accel = 0.5 * cd_opt * self.area_to_mass * density * velocity**2
        predicted = -drag_accel[:, np.newaxis] * (velocity[:, np.newaxis] / 
                                                   np.linalg.norm(velocity[:, np.newaxis], axis=1, keepdims=True))
        post_fit_res = residuals - predicted
        
        return SystemIDResult(
            parameters=result.x,
            covariance=cov,
            residuals=post_fit_res,
            cost=result.fun,
            success=result.success,
            message=result.message,
            iterations=result.nit if hasattr(result, 'nit') else 0
        )


class SolarPressureEstimator:
    """
    Estimate solar radiation pressure coefficient.
    
    Models SRP acceleration as:
        a_srp = -CR * (A/m) * (P_sun / c) * (1 AU / r)² * ŝ
    
    where CR is the radiation pressure coefficient to be estimated.
    """
    
    def __init__(self, area_to_mass: float, initial_cr: float = 1.5):
        """
        Initialize SRP coefficient estimator.
        
        Parameters
        ----------
        area_to_mass : float
            Effective area to mass ratio (m²/kg)
        initial_cr : float
            Initial guess for CR (typically 1.0-2.0)
        """
        self.area_to_mass = area_to_mass
        self.initial_cr = initial_cr
        
        # Solar radiation pressure at 1 AU
        self.P_sun = 4.56e-6  # N/m²
        self.c = 299792458.0  # m/s
        self.AU = 1.49597870700e11  # m
    
    def estimate(self, times: np.ndarray, residuals: np.ndarray,
                 sun_vectors: np.ndarray, sun_distances: np.ndarray,
                 shadow_factors: np.ndarray,
                 weights: Optional[np.ndarray] = None) -> SystemIDResult:
        """
        Estimate SRP coefficient from residuals.
        
        Parameters
        ----------
        times : np.ndarray
            Time epochs
        residuals : np.ndarray
            Position/velocity residuals (N x 3)
        sun_vectors : np.ndarray
            Unit vectors to sun (N x 3)
        sun_distances : np.ndarray
            Distances to sun (m)
        shadow_factors : np.ndarray
            Shadow factors (0=shadow, 1=sunlight, 0-1=penumbra)
        weights : np.ndarray, optional
            Observation weights
        
        Returns
        -------
        SystemIDResult
            Estimation results
        """
        if weights is None:
            weights = np.ones(len(times))
        
        def cost_function(cr):
            """Cost function for SRP coefficient estimation."""
            # SRP acceleration magnitude
            r_factor = (self.AU / sun_distances)**2
            srp_mag = cr[0] * self.area_to_mass * (self.P_sun / self.c) * r_factor
            
            # Apply shadow
            srp_mag *= shadow_factors
            
            # SRP acceleration vector (away from sun)
            srp_accel = srp_mag[:, np.newaxis] * (-sun_vectors)
            
            # Weighted residuals
            weighted_res = weights[:, np.newaxis] * (residuals - srp_accel)
            
            return np.sum(weighted_res**2)
        
        # Optimize
        result = optimize.minimize(
            cost_function,
            x0=[self.initial_cr],
            method='L-BFGS-B',
            bounds=[(0.5, 3.0)]  # Physical bounds on CR
        )
        
        # Compute covariance
        cr_opt = result.x[0]
        eps = 1e-6
        h11 = (cost_function([cr_opt + eps]) - 2*cost_function([cr_opt]) + 
               cost_function([cr_opt - eps])) / eps**2
        cov = np.array([[2.0 / max(h11, 1e-10)]])
        
        # Post-fit residuals
        r_factor = (self.AU / sun_distances)**2
        srp_mag = cr_opt * self.area_to_mass * (self.P_sun / self.c) * r_factor
        srp_mag *= shadow_factors
        srp_accel = srp_mag[:, np.newaxis] * (-sun_vectors)
        post_fit_res = residuals - srp_accel
        
        return SystemIDResult(
            parameters=result.x,
            covariance=cov,
            residuals=post_fit_res,
            cost=result.fun,
            success=result.success,
            message=result.message,
            iterations=result.nit if hasattr(result, 'nit') else 0
        )


class EmpiricalAccelerationModel:
    """
    Fit empirical acceleration model to unmodeled forces.
    
    Uses periodic and polynomial terms to model systematic errors.
    """
    
    def __init__(self, n_harmonics: int = 2, polynomial_degree: int = 1):
        """
        Initialize empirical acceleration model.
        
        Parameters
        ----------
        n_harmonics : int
            Number of harmonic terms (periods of orbital period)
        polynomial_degree : int
            Degree of polynomial trend
        """
        self.n_harmonics = n_harmonics
        self.polynomial_degree = polynomial_degree
    
    def build_design_matrix(self, times: np.ndarray, 
                           orbital_period: float) -> np.ndarray:
        """
        Build design matrix for empirical model.
        
        Parameters
        ----------
        times : np.ndarray
            Time epochs (seconds from reference)
        orbital_period : float
            Orbital period (seconds)
        
        Returns
        -------
        np.ndarray
            Design matrix (N x M) where M is number of parameters
        """
        n = len(times)
        t_norm = times / orbital_period  # Normalize by orbital period
        
        # Number of parameters per component (3 for x, y, z)
        n_params = (2 * self.n_harmonics + self.polynomial_degree + 1)
        
        A = np.zeros((n * 3, n_params * 3))
        
        for axis in range(3):
            col = 0
            
            # Polynomial terms
            for deg in range(self.polynomial_degree + 1):
                A[axis*n:(axis+1)*n, axis*n_params + col] = t_norm**deg
                col += 1
            
            # Harmonic terms
            for k in range(1, self.n_harmonics + 1):
                omega = 2 * np.pi * k
                A[axis*n:(axis+1)*n, axis*n_params + col] = np.sin(omega * t_norm)
                A[axis*n:(axis+1)*n, axis*n_params + col + 1] = np.cos(omega * t_norm)
                col += 2
        
        return A
    
    def fit(self, times: np.ndarray, residuals: np.ndarray,
            orbital_period: float,
            weights: Optional[np.ndarray] = None) -> SystemIDResult:
        """
        Fit empirical acceleration model to residuals.
        
        Parameters
        ----------
        times : np.ndarray
            Time epochs
        residuals : np.ndarray
            Acceleration residuals (N x 3)
        orbital_period : float
            Orbital period
        weights : np.ndarray, optional
            Observation weights
        
        Returns
        -------
        SystemIDResult
            Fit results
        """
        if weights is None:
            weights = np.ones(len(times))
        
        # Build design matrix
        A = self.build_design_matrix(times, orbital_period)
        
        # Weight matrix
        W = np.diag(np.tile(weights, 3))
        
        # Weighted least squares
        y = residuals.flatten()
        
        # Normal equations: (A^T W A) x = A^T W y
        AtWA = A.T @ W @ A
        AtWy = A.T @ W @ y
        
        # Solve (with regularization if needed)
        try:
            params = np.linalg.solve(AtWA, AtWy)
        except np.linalg.LinAlgError:
            # Add regularization
            reg = 1e-8 * np.eye(AtWA.shape[0])
            params = np.linalg.solve(AtWA + reg, AtWy)
        
        # Covariance
        residual_var = np.sum(W @ (y - A @ params)**2) / (len(y) - len(params))
        
        try:
            cov = residual_var * np.linalg.inv(AtWA)
        except np.linalg.LinAlgError:
            cov = np.full_like(AtWA, np.nan)
        
        # Post-fit residuals
        predicted = (A @ params).reshape(-1, 3)
        post_fit = residuals - predicted
        
        cost = np.sum(W @ (y - A @ params)**2)
        
        return SystemIDResult(
            parameters=params,
            covariance=cov,
            residuals=post_fit,
            cost=cost,
            success=True,
            message="Empirical model fitted",
            iterations=1
        )
    
    def evaluate(self, times: np.ndarray, orbital_period: float,
                 parameters: np.ndarray) -> np.ndarray:
        """
        Evaluate empirical model at given times.
        
        Parameters
        ----------
        times : np.ndarray
            Time epochs
        orbital_period : float
            Orbital period
        parameters : np.ndarray
            Model parameters
        
        Returns
        -------
        np.ndarray
            Predicted accelerations (N x 3)
        """
        A = self.build_design_matrix(times, orbital_period)
        predicted = (A @ parameters).reshape(-1, 3)
        
        return predicted


class ResidualAnalyzer:
    """
    Analyze filter residuals for model validation.
    
    Provides tools to assess:
    - Whiteness of residuals
    - Systematic biases
    - Time-correlated errors
    - Outliers
    """
    
    @staticmethod
    def compute_statistics(residuals: np.ndarray) -> Dict[str, float]:
        """
        Compute basic residual statistics.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residual time series (N x d)
        
        Returns
        -------
        dict
            Statistical measures
        """
        residuals = np.atleast_2d(residuals)
        
        return {
            'mean': np.mean(residuals, axis=0),
            'std': np.std(residuals, axis=0),
            'rms': np.sqrt(np.mean(residuals**2, axis=0)),
            'min': np.min(residuals, axis=0),
            'max': np.max(residuals, axis=0),
            'median': np.median(residuals, axis=0),
            'mad': stats.median_abs_deviation(residuals, axis=0)
        }
    
    @staticmethod
    def detect_outliers(residuals: np.ndarray, 
                       threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers using modified Z-score.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residual time series
        threshold : float
            MAD threshold for outlier detection
        
        Returns
        -------
        np.ndarray
            Boolean mask of outliers
        """
        residuals = np.atleast_2d(residuals)
        
        median = np.median(residuals, axis=0)
        mad = stats.median_abs_deviation(residuals, axis=0)
        
        # Modified Z-score
        modified_z = 0.6745 * (residuals - median) / (mad + 1e-10)
        
        # Outliers in any dimension
        outliers = np.any(np.abs(modified_z) > threshold, axis=1)
        
        return outliers
    
    @staticmethod
    def compute_autocorrelation(residuals: np.ndarray, 
                               max_lag: int = 50) -> np.ndarray:
        """
        Compute autocorrelation function.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residual time series (1D)
        max_lag : int
            Maximum lag to compute
        
        Returns
        -------
        np.ndarray
            Autocorrelation values
        """
        residuals = residuals - np.mean(residuals)
        
        acf = np.correlate(residuals, residuals, mode='full')
        acf = acf[len(acf)//2:]
        acf = acf[:max_lag+1] / acf[0]
        
        return acf
    
    @staticmethod
    def normalized_innovation_squared(residuals: np.ndarray,
                                     covariances: np.ndarray) -> np.ndarray:
        """
        Compute normalized innovation squared (NIS).
        
        For properly tuned filter, NIS should be chi-squared distributed.
        
        Parameters
        ----------
        residuals : np.ndarray
            Innovation sequence (N x d)
        covariances : np.ndarray
            Innovation covariance matrices (N x d x d)
        
        Returns
        -------
        np.ndarray
            NIS values
        """
        n = len(residuals)
        nis = np.zeros(n)
        
        for i in range(n):
            try:
                S_inv = np.linalg.inv(covariances[i])
                nis[i] = residuals[i] @ S_inv @ residuals[i]
            except np.linalg.LinAlgError:
                nis[i] = np.nan
        
        return nis
    
    @classmethod
    def comprehensive_analysis(cls, residuals: np.ndarray) -> Dict[str, any]:
        """
        Run comprehensive residual analysis.
        
        Parameters
        ----------
        residuals : np.ndarray
            Residual time series
        
        Returns
        -------
        dict
            Analysis results
        """
        results = {
            'statistics': cls.compute_statistics(residuals),
            'outliers': cls.detect_outliers(residuals),
            'autocorrelation': cls.compute_autocorrelation(residuals.flatten())
        }
        
        results['n_outliers'] = np.sum(results['outliers'])
        results['outlier_fraction'] = results['n_outliers'] / len(residuals)
        
        return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    print("=" * 60)
    print("SYSTEM IDENTIFICATION EXAMPLES")
    print("=" * 60)
    
    # Simulate drag coefficient estimation
    print("\n1. DRAG COEFFICIENT ESTIMATION")
    print("-" * 60)
    
    n = 100
    times = np.linspace(0, 5400, n)  # 1.5 hours
    true_cd = 2.3
    area_to_mass = 0.01  # m²/kg
    
    # Simulate conditions
    density = 1e-12 * np.exp(-times / 3600)  # Decaying density
    velocity = 7500 + 50 * np.sin(2 * np.pi * times / 5400)
    
    # True drag acceleration
    drag_true = 0.5 * true_cd * area_to_mass * density * velocity**2
    drag_accel = drag_true[:, np.newaxis] * np.array([[-1, 0, 0]])
    
    # Add noise
    residuals = drag_accel + np.random.randn(n, 3) * 1e-8
    
    # Estimate
    estimator = DragCoefficientEstimator(area_to_mass, initial_cd=2.0)
    result = estimator.estimate(times, residuals, density, velocity)
    
    print(f"True CD: {true_cd}")
    print(f"Estimated CD: {result.parameters[0]:.4f} ± {result.parameter_uncertainties()[0]:.4f}")
    print(f"Optimization success: {result.success}")
    print(f"RMS residual: {np.sqrt(np.mean(result.residuals**2)):.2e} m/s²")
    
    # SRP coefficient estimation
    print("\n2. SOLAR RADIATION PRESSURE ESTIMATION")
    print("-" * 60)
    
    true_cr = 1.8
    sun_vectors = np.random.randn(n, 3)
    sun_vectors /= np.linalg.norm(sun_vectors, axis=1, keepdims=True)
    sun_distances = np.full(n, 1.496e11)  # 1 AU
    shadow_factors = np.ones(n)
    
    # True SRP
    srp_estimator = SolarPressureEstimator(area_to_mass, initial_cr=1.5)
    P_sun = 4.56e-6
    c = 299792458.0
    srp_mag = true_cr * area_to_mass * (P_sun / c)
    srp_accel = srp_mag * (-sun_vectors)
    
    # Add noise
    residuals_srp = srp_accel + np.random.randn(n, 3) * 1e-9
    
    result_srp = srp_estimator.estimate(times, residuals_srp, sun_vectors,
                                       sun_distances, shadow_factors)
    
    print(f"True CR: {true_cr}")
    print(f"Estimated CR: {result_srp.parameters[0]:.4f} ± {result_srp.parameter_uncertainties()[0]:.4f}")
    print(f"Optimization success: {result_srp.success}")
    
    # Empirical acceleration model
    print("\n3. EMPIRICAL ACCELERATION MODEL")
    print("-" * 60)
    
    orbital_period = 5400.0  # seconds
    emp_model = EmpiricalAccelerationModel(n_harmonics=2, polynomial_degree=1)
    
    # Simulate systematic acceleration
    t_norm = times / orbital_period
    true_accel = np.zeros((n, 3))
    true_accel[:, 0] = 1e-8 * (0.5 + 0.3 * np.sin(2*np.pi*t_norm))
    true_accel[:, 1] = 1e-8 * (0.2 * np.cos(2*np.pi*t_norm))
    
    residuals_emp = true_accel + np.random.randn(n, 3) * 5e-10
    
    result_emp = emp_model.fit(times, residuals_emp, orbital_period)
    
    print(f"Number of parameters: {len(result_emp.parameters)}")
    print(f"Fit success: {result_emp.success}")
    print(f"Pre-fit RMS: {np.sqrt(np.mean(residuals_emp**2)):.2e} m/s²")
    print(f"Post-fit RMS: {np.sqrt(np.mean(result_emp.residuals**2)):.2e} m/s²")
    
    # Residual analysis
    print("\n4. RESIDUAL ANALYSIS")
    print("-" * 60)
    
    analyzer = ResidualAnalyzer()
    analysis = analyzer.comprehensive_analysis(result_emp.residuals)
    
    print(f"\nRMS: {analysis['statistics']['rms']}")
    print(f"Outliers detected: {analysis['n_outliers']} ({analysis['outlier_fraction']*100:.1f}%)")
    print(f"Max autocorrelation (lag>0): {np.max(np.abs(analysis['autocorrelation'][1:])):.3f}")
    
    print("\n✓ System identification examples completed successfully")
