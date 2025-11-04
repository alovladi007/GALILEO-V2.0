"""
Geophysical inversion algorithms for mass distribution recovery.

Implements Tikhonov regularization, iterative methods, and Bayesian inversion
for recovering Earth's mass distribution from gravimetric measurements.
"""

from typing import Callable, Optional, Tuple
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from dataclasses import dataclass


@dataclass
class InversionConfig:
    """Configuration for inversion problem."""
    
    regularization_param: float = 1e-3
    max_iterations: int = 100
    tolerance: float = 1e-6
    regularization_type: str = "tikhonov"  # tikhonov, total_variation, entropy


class TikhonovInversion:
    """
    Tikhonov regularized inversion for ill-posed gravity inverse problem.
    
    Solves: min ||Gm - d||² + λ²||Lm||²
    where G is forward operator, m is model, d is data, L is regularization operator
    """
    
    def __init__(self, config: InversionConfig) -> None:
        self.config = config
        
    @staticmethod
    @jit
    def forward_operator(
        model: jnp.ndarray,
        observation_points: jnp.ndarray,
        source_points: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Gravitational forward operator: maps mass distribution to observations.
        
        Args:
            model: Mass distribution values
            observation_points: Sensor positions [N, 3]
            source_points: Model grid points [M, 3]
            
        Returns:
            Predicted observations [N]
        """
        # Gravitational kernel
        def kernel(obs: jnp.ndarray, src: jnp.ndarray) -> float:
            r = jnp.linalg.norm(obs - src)
            return 1.0 / (r + 1e-10)  # Avoid division by zero
        
        # Vectorized kernel computation
        G = vmap(lambda obs: vmap(lambda src: kernel(obs, src))(source_points))(
            observation_points
        )
        
        return G @ model
    
    def solve(
        self,
        data: jnp.ndarray,
        observation_points: jnp.ndarray,
        source_points: jnp.ndarray,
        initial_model: Optional[jnp.ndarray] = None,
    ) -> Tuple[jnp.ndarray, dict]:
        """
        Solve regularized inverse problem.
        
        Args:
            data: Observed gravity measurements
            observation_points: Sensor positions
            source_points: Model discretization points
            initial_model: Initial model guess
            
        Returns:
            Recovered model and convergence info
        """
        n_params = len(source_points)
        
        if initial_model is None:
            model = jnp.zeros(n_params)
        else:
            model = initial_model
            
        # Build normal equations: (G'G + λ²L'L)m = G'd
        # This is a placeholder structure
        
        history = {
            "residuals": [],
            "regularization": [],
            "iterations": 0,
        }
        
        return model, history
    
    @staticmethod
    @jit
    def data_misfit(
        model: jnp.ndarray,
        data: jnp.ndarray,
        forward_op: Callable,
    ) -> float:
        """Calculate L2 data misfit."""
        predicted = forward_op(model)
        return jnp.sum((predicted - data) ** 2)
    
    @staticmethod
    @jit
    def regularization_term(
        model: jnp.ndarray,
        reg_matrix: jnp.ndarray,
    ) -> float:
        """Calculate regularization term."""
        return jnp.sum((reg_matrix @ model) ** 2)


class BayesianInversion:
    """
    Bayesian inversion with Gaussian priors and likelihoods.
    
    Computes posterior distribution: p(m|d) ∝ p(d|m) p(m)
    """
    
    def __init__(self, config: InversionConfig) -> None:
        self.config = config
        
    @staticmethod
    @jit
    def log_likelihood(
        model: jnp.ndarray,
        data: jnp.ndarray,
        forward_op: Callable,
        noise_variance: float,
    ) -> float:
        """
        Compute log-likelihood assuming Gaussian noise.
        
        Args:
            model: Model parameters
            data: Observations
            forward_op: Forward operator
            noise_variance: Data noise variance
            
        Returns:
            Log-likelihood value
        """
        predicted = forward_op(model)
        residual = data - predicted
        return -0.5 * jnp.sum(residual ** 2) / noise_variance
    
    @staticmethod
    @jit
    def log_prior(
        model: jnp.ndarray,
        prior_mean: jnp.ndarray,
        prior_cov_inv: jnp.ndarray,
    ) -> float:
        """
        Compute log-prior (Gaussian).
        
        Args:
            model: Model parameters
            prior_mean: Prior mean
            prior_cov_inv: Inverse of prior covariance
            
        Returns:
            Log-prior value
        """
        diff = model - prior_mean
        return -0.5 * diff.T @ prior_cov_inv @ diff
    
    def maximum_a_posteriori(
        self,
        data: jnp.ndarray,
        forward_op: Callable,
        prior_mean: jnp.ndarray,
        prior_cov_inv: jnp.ndarray,
        noise_variance: float,
    ) -> jnp.ndarray:
        """
        Find MAP estimate using gradient-based optimization.
        
        Args:
            data: Observations
            forward_op: Forward operator
            prior_mean: Prior mean
            prior_cov_inv: Inverse prior covariance
            noise_variance: Data noise variance
            
        Returns:
            MAP estimate
        """
        def objective(model: jnp.ndarray) -> float:
            return -(
                self.log_likelihood(model, data, forward_op, noise_variance)
                + self.log_prior(model, prior_mean, prior_cov_inv)
            )
        
        # Use JAX optimizer (placeholder)
        initial_model = prior_mean
        return initial_model  # Would run optimization


def resolution_matrix(
    forward_matrix: np.ndarray,
    regularization_matrix: np.ndarray,
    lambda_param: float,
) -> np.ndarray:
    """
    Compute model resolution matrix.
    
    R = (G'G + λ²L'L)^(-1) G'G
    
    Args:
        forward_matrix: Forward operator matrix G
        regularization_matrix: Regularization operator L
        lambda_param: Regularization parameter
        
    Returns:
        Resolution matrix R
    """
    G = forward_matrix
    L = regularization_matrix
    
    # Normal equations matrix
    A = G.T @ G + lambda_param**2 * L.T @ L
    
    # Resolution matrix
    R = np.linalg.solve(A, G.T @ G)
    
    return R
