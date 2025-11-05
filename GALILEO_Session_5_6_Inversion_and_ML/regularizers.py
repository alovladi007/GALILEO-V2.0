"""
Regularization Methods for Geophysical Inversion
=================================================

Implements various regularization techniques:
- Total Variation (TV) for edge-preserving smoothness
- Sparsity-promoting regularizers (L1, weighted L1)
- Geologic/structural priors
"""

import numpy as np
from scipy.sparse import diags, vstack, hstack
from scipy.optimize import minimize
from typing import Tuple, Optional, Callable


class RegularizerBase:
    """Base class for regularization operators."""
    
    def __init__(self, n_model: int):
        self.n_model = n_model
    
    def matrix(self) -> np.ndarray:
        """Return regularization matrix L."""
        raise NotImplementedError
    
    def penalty(self, m: np.ndarray) -> float:
        """Compute penalty value."""
        raise NotImplementedError
    
    def gradient(self, m: np.ndarray) -> np.ndarray:
        """Compute gradient of penalty."""
        raise NotImplementedError


class SmoothnessRegularizer(RegularizerBase):
    """
    First or second derivative smoothness regularizer.
    Penalizes roughness in the model.
    """
    
    def __init__(self, n_model: int, order: int = 2, alpha: float = 1.0):
        """
        Parameters
        ----------
        n_model : int
            Number of model parameters
        order : int
            Derivative order (1 or 2)
        alpha : float
            Anisotropy parameter for 2D grids
        """
        super().__init__(n_model)
        self.order = order
        self.alpha = alpha
    
    def matrix(self) -> np.ndarray:
        """Construct finite difference matrix."""
        if self.order == 1:
            # First derivative
            diag = np.ones(self.n_model - 1)
            L = diags([diag, -diag], [0, 1], shape=(self.n_model - 1, self.n_model))
            return L.toarray()
        
        elif self.order == 2:
            # Second derivative
            diag_m1 = np.ones(self.n_model - 2)
            diag_0 = -2 * np.ones(self.n_model - 2)
            diag_p1 = np.ones(self.n_model - 2)
            L = diags([diag_m1, diag_0, diag_p1], [0, 1, 2], 
                     shape=(self.n_model - 2, self.n_model))
            return L.toarray()
        
        else:
            raise ValueError("Only orders 1 and 2 supported")
    
    def matrix_2d(self, nx: int, ny: int) -> np.ndarray:
        """
        Construct 2D finite difference matrix for gridded models.
        
        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        
        Returns
        -------
        L : ndarray
            Combined x and y derivative matrices
        """
        # X-direction derivatives
        Ix = np.eye(ny)
        if self.order == 1:
            Dx = diags([np.ones(nx-1), -np.ones(nx-1)], [0, 1], shape=(nx-1, nx))
        else:
            Dx = diags([np.ones(nx-2), -2*np.ones(nx-2), np.ones(nx-2)], 
                      [0, 1, 2], shape=(nx-2, nx))
        
        Lx = np.kron(Dx.toarray(), Ix)
        
        # Y-direction derivatives
        Iy = np.eye(nx)
        if self.order == 1:
            Dy = diags([np.ones(ny-1), -np.ones(ny-1)], [0, 1], shape=(ny-1, ny))
        else:
            Dy = diags([np.ones(ny-2), -2*np.ones(ny-2), np.ones(ny-2)], 
                      [0, 1, 2], shape=(ny-2, ny))
        
        Ly = np.kron(Iy, Dy.toarray())
        
        # Combine with anisotropy
        L = np.vstack([Lx, self.alpha * Ly])
        
        return L
    
    def penalty(self, m: np.ndarray) -> float:
        L = self.matrix()
        return 0.5 * np.linalg.norm(L @ m)**2
    
    def gradient(self, m: np.ndarray) -> np.ndarray:
        L = self.matrix()
        return L.T @ (L @ m)


class TotalVariationRegularizer(RegularizerBase):
    """
    Total Variation (TV) regularizer for edge-preserving inversion.
    
    Minimizes TV(m) = sum_i sqrt((∇m_i)^2 + ε^2)
    where ε is a small parameter for numerical stability.
    """
    
    def __init__(self, n_model: int, epsilon: float = 1e-6, beta: float = 1.0):
        """
        Parameters
        ----------
        n_model : int
            Number of model parameters
        epsilon : float
            Smoothing parameter for numerical stability
        beta : float
            Exponent for p-norm TV (beta=1 is standard TV, beta=2 is smooth)
        """
        super().__init__(n_model)
        self.epsilon = epsilon
        self.beta = beta
    
    def penalty(self, m: np.ndarray) -> float:
        """Compute TV penalty."""
        # Simple 1D TV
        grad_m = np.diff(m)
        tv = np.sum(np.sqrt(grad_m**2 + self.epsilon**2))
        return tv
    
    def penalty_2d(self, m: np.ndarray, nx: int, ny: int) -> float:
        """
        Compute 2D TV penalty for gridded model.
        
        Parameters
        ----------
        m : ndarray
            Model vector (flattened)
        nx, ny : int
            Grid dimensions
        """
        m_grid = m.reshape((nx, ny))
        
        # Gradient magnitudes
        grad_x = np.diff(m_grid, axis=0)
        grad_y = np.diff(m_grid, axis=1)
        
        # Pad to same size
        grad_x = np.pad(grad_x, ((0, 1), (0, 0)), mode='constant')
        grad_y = np.pad(grad_y, ((0, 0), (0, 1)), mode='constant')
        
        # TV norm
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + self.epsilon**2)
        tv = np.sum(grad_mag**self.beta)
        
        return tv
    
    def gradient(self, m: np.ndarray) -> np.ndarray:
        """Compute gradient of TV penalty (1D)."""
        grad_m = np.diff(m)
        weights = 1.0 / np.sqrt(grad_m**2 + self.epsilon**2)
        
        # Finite difference approximation of div(∇m/|∇m|)
        grad_tv = np.zeros_like(m)
        grad_tv[:-1] -= weights * grad_m
        grad_tv[1:] += weights * grad_m
        
        return grad_tv
    
    def gradient_2d(self, m: np.ndarray, nx: int, ny: int) -> np.ndarray:
        """Compute gradient of 2D TV penalty."""
        m_grid = m.reshape((nx, ny))
        
        grad_x = np.diff(m_grid, axis=0)
        grad_y = np.diff(m_grid, axis=1)
        
        grad_x = np.pad(grad_x, ((0, 1), (0, 0)), mode='constant')
        grad_y = np.pad(grad_y, ((0, 0), (0, 1)), mode='constant')
        
        grad_mag = np.sqrt(grad_x**2 + grad_y**2 + self.epsilon**2)
        
        # Weighted gradients
        wx = self.beta * grad_x / grad_mag**(2 - self.beta)
        wy = self.beta * grad_y / grad_mag**(2 - self.beta)
        
        # Divergence
        div = np.zeros((nx, ny))
        div[:-1, :] -= wx[:-1, :]
        div[1:, :] += wx[:-1, :]
        div[:, :-1] -= wy[:, :-1]
        div[:, 1:] += wy[:, :-1]
        
        return div.ravel()


class SparsityRegularizer(RegularizerBase):
    """
    L1 sparsity-promoting regularizer.
    
    Promotes sparse solutions by minimizing ||m||_1 or ||Wm||_1
    where W is a weighting matrix (e.g., wavelet transform).
    """
    
    def __init__(self, n_model: int, weights: Optional[np.ndarray] = None,
                 epsilon: float = 1e-8):
        """
        Parameters
        ----------
        n_model : int
            Number of model parameters
        weights : ndarray, optional
            Weighting matrix W (default: identity)
        epsilon : float
            Smoothing parameter for numerical stability
        """
        super().__init__(n_model)
        self.weights = weights if weights is not None else np.eye(n_model)
        self.epsilon = epsilon
    
    def penalty(self, m: np.ndarray) -> float:
        """Compute L1 penalty."""
        Wm = self.weights @ m if self.weights.ndim == 2 else self.weights * m
        return np.sum(np.abs(Wm))
    
    def penalty_smooth(self, m: np.ndarray) -> float:
        """Smooth approximation: sum(sqrt(m_i^2 + ε^2))."""
        Wm = self.weights @ m if self.weights.ndim == 2 else self.weights * m
        return np.sum(np.sqrt(Wm**2 + self.epsilon**2))
    
    def gradient(self, m: np.ndarray) -> np.ndarray:
        """Gradient using smooth approximation."""
        Wm = self.weights @ m if self.weights.ndim == 2 else self.weights * m
        grad_smooth = Wm / np.sqrt(Wm**2 + self.epsilon**2)
        
        if self.weights.ndim == 2:
            return self.weights.T @ grad_smooth
        else:
            return self.weights * grad_smooth


class GeologicPriorRegularizer(RegularizerBase):
    """
    Incorporate geologic/structural prior information.
    
    Penalizes deviation from reference model with spatially varying weights.
    Useful for incorporating geological constraints, known structures, etc.
    """
    
    def __init__(self, n_model: int, m_ref: np.ndarray, 
                 weights: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        n_model : int
            Number of model parameters
        m_ref : ndarray
            Reference model (prior)
        weights : ndarray, optional
            Spatial weights (confidence in prior)
        """
        super().__init__(n_model)
        self.m_ref = m_ref
        self.weights = weights if weights is not None else np.ones(n_model)
    
    def penalty(self, m: np.ndarray) -> float:
        """Weighted squared deviation from reference."""
        diff = m - self.m_ref
        return 0.5 * np.sum(self.weights * diff**2)
    
    def gradient(self, m: np.ndarray) -> np.ndarray:
        """Gradient of penalty."""
        return self.weights * (m - self.m_ref)
    
    def matrix(self) -> np.ndarray:
        """Diagonal weight matrix."""
        return np.diag(np.sqrt(self.weights))


class CrossGradientRegularizer:
    """
    Cross-gradient structural coupling for joint inversion.
    
    Enforces structural similarity between two model types by minimizing
    the cross-gradient: τ = ∇m1 × ∇m2
    """
    
    def __init__(self, nx: int, ny: int):
        """
        Parameters
        ----------
        nx, ny : int
            Grid dimensions
        """
        self.nx = nx
        self.ny = ny
    
    def penalty(self, m1: np.ndarray, m2: np.ndarray) -> float:
        """
        Compute cross-gradient penalty.
        
        Parameters
        ----------
        m1, m2 : ndarray
            Two model vectors to couple
        """
        m1_grid = m1.reshape((self.nx, self.ny))
        m2_grid = m2.reshape((self.nx, self.ny))
        
        # Compute gradients
        grad1_x, grad1_y = np.gradient(m1_grid)
        grad2_x, grad2_y = np.gradient(m2_grid)
        
        # Cross-gradient (in 2D, this is the z-component)
        cross_grad = grad1_x * grad2_y - grad1_y * grad2_x
        
        return 0.5 * np.sum(cross_grad**2)
    
    def gradient(self, m1: np.ndarray, m2: np.ndarray, 
                 which: int = 1) -> np.ndarray:
        """
        Compute gradient of cross-gradient penalty with respect to m1 or m2.
        
        Parameters
        ----------
        m1, m2 : ndarray
            Model vectors
        which : int
            Compute gradient w.r.t. m1 (which=1) or m2 (which=2)
        """
        m1_grid = m1.reshape((self.nx, self.ny))
        m2_grid = m2.reshape((self.nx, self.ny))
        
        grad1_x, grad1_y = np.gradient(m1_grid)
        grad2_x, grad2_y = np.gradient(m2_grid)
        
        cross_grad = grad1_x * grad2_y - grad1_y * grad2_x
        
        if which == 1:
            # Gradient w.r.t. m1
            grad_x = np.gradient(cross_grad * grad2_y, axis=0)
            grad_y = -np.gradient(cross_grad * grad2_x, axis=1)
        else:
            # Gradient w.r.t. m2
            grad_x = -np.gradient(cross_grad * grad1_y, axis=0)
            grad_y = np.gradient(cross_grad * grad1_x, axis=1)
        
        return (grad_x + grad_y).ravel()


class MinimumSupportRegularizer(RegularizerBase):
    """
    Minimum support (compact support) regularizer.
    
    Promotes compact anomalies by minimizing support measure:
    MS(m) = sum_i (m_i / m_max)^2 / (1 + (m_i / m_max)^2)
    """
    
    def __init__(self, n_model: int, m_max: Optional[float] = None):
        """
        Parameters
        ----------
        n_model : int
            Number of model parameters
        m_max : float, optional
            Maximum expected model value (auto-computed if None)
        """
        super().__init__(n_model)
        self.m_max = m_max
    
    def penalty(self, m: np.ndarray) -> float:
        """Compute minimum support penalty."""
        if self.m_max is None:
            self.m_max = np.max(np.abs(m))
        
        m_norm = m / self.m_max
        ms = np.sum(m_norm**2 / (1 + m_norm**2))
        return ms
    
    def gradient(self, m: np.ndarray) -> np.ndarray:
        """Gradient of minimum support penalty."""
        if self.m_max is None:
            self.m_max = np.max(np.abs(m))
        
        m_norm = m / self.m_max
        denom = (1 + m_norm**2)**2
        grad = 2 * m_norm / (self.m_max**2 * denom)
        return grad
