"""
Geophysical Inversion Solvers
==============================

Implements various inversion methods for geophysical problems:
- Linear Tikhonov regularization
- Nonlinear Gauss-Newton optimization
- Bayesian MAP estimation with uncertainty quantification
"""

import numpy as np
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve, LinearOperator
from typing import Callable, Tuple, Optional, Dict
import warnings


class TikhonovSolver:
    """
    Linear Tikhonov regularization solver for linear inverse problems.
    
    Solves: min ||Gm - d||^2 + λ^2 ||Lm||^2
    where G is forward operator, m is model, d is data, L is regularization matrix.
    """
    
    def __init__(self, forward_matrix: np.ndarray, regularization_matrix: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        forward_matrix : ndarray, shape (n_data, n_model)
            Forward operator matrix G
        regularization_matrix : ndarray, optional
            Regularization matrix L (default: identity)
        """
        self.G = forward_matrix
        self.n_data, self.n_model = forward_matrix.shape
        
        if regularization_matrix is None:
            self.L = np.eye(self.n_model)
        else:
            self.L = regularization_matrix
    
    def solve(self, data: np.ndarray, lambda_reg: float, 
              compute_resolution: bool = True) -> Dict:
        """
        Solve Tikhonov inverse problem.
        
        Parameters
        ----------
        data : ndarray, shape (n_data,)
            Observed data vector
        lambda_reg : float
            Regularization parameter λ
        compute_resolution : bool
            Whether to compute resolution matrix
        
        Returns
        -------
        result : dict
            Dictionary containing:
            - 'model': Estimated model
            - 'residual': Data residual norm
            - 'resolution_matrix': Model resolution matrix (if requested)
            - 'predicted_data': Forward modeled data
        """
        # Normal equations: (G^T G + λ^2 L^T L) m = G^T d
        GTG = self.G.T @ self.G
        LTL = self.L.T @ self.L
        GTd = self.G.T @ data
        
        # Solve system
        A = GTG + lambda_reg**2 * LTL
        model = np.linalg.solve(A, GTd)
        
        # Compute residuals
        predicted = self.G @ model
        residual = np.linalg.norm(data - predicted)
        
        result = {
            'model': model,
            'residual': residual,
            'predicted_data': predicted,
            'lambda': lambda_reg
        }
        
        # Resolution matrix: R = (G^T G + λ^2 L^T L)^{-1} G^T G
        if compute_resolution:
            R = np.linalg.solve(A, GTG)
            result['resolution_matrix'] = R
            result['resolution_diagonal'] = np.diag(R)
        
        return result
    
    def l_curve(self, data: np.ndarray, lambdas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute L-curve for regularization parameter selection.
        
        Parameters
        ----------
        data : ndarray
            Observed data
        lambdas : ndarray
            Array of regularization parameters to test
        
        Returns
        -------
        residuals : ndarray
            Data residual norms
        model_norms : ndarray
            Regularized model norms
        """
        residuals = np.zeros_like(lambdas)
        model_norms = np.zeros_like(lambdas)
        
        for i, lam in enumerate(lambdas):
            result = self.solve(data, lam, compute_resolution=False)
            residuals[i] = result['residual']
            model_norms[i] = np.linalg.norm(self.L @ result['model'])
        
        return residuals, model_norms


class GaussNewtonSolver:
    """
    Nonlinear Gauss-Newton solver for nonlinear inverse problems.
    
    Iteratively minimizes: ||F(m) - d||^2 + λ^2 ||L(m - m_ref)||^2
    """
    
    def __init__(self, forward_func: Callable, jacobian_func: Callable,
                 regularization_matrix: Optional[np.ndarray] = None):
        """
        Parameters
        ----------
        forward_func : callable
            Forward modeling function F(m) -> d
        jacobian_func : callable
            Jacobian function J(m) -> dF/dm
        regularization_matrix : ndarray, optional
            Regularization matrix L
        """
        self.forward = forward_func
        self.jacobian = jacobian_func
        self.L = regularization_matrix
    
    def solve(self, data: np.ndarray, m_init: np.ndarray, lambda_reg: float,
              max_iter: int = 20, tol: float = 1e-6, 
              line_search: bool = True) -> Dict:
        """
        Solve nonlinear inverse problem via Gauss-Newton iteration.
        
        Parameters
        ----------
        data : ndarray
            Observed data
        m_init : ndarray
            Initial model estimate
        lambda_reg : float
            Regularization parameter
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        line_search : bool
            Use line search for step length
        
        Returns
        -------
        result : dict
            Optimization results including model, residuals, convergence history
        """
        m = m_init.copy()
        n_model = len(m)
        
        if self.L is None:
            self.L = np.eye(n_model)
        
        # History tracking
        history = {
            'models': [m.copy()],
            'data_misfit': [],
            'model_norm': [],
            'objective': [],
            'step_length': []
        }
        
        for iteration in range(max_iter):
            # Forward model and residual
            d_pred = self.forward(m)
            residual = data - d_pred
            
            # Jacobian
            J = self.jacobian(m)
            
            # Compute objective function
            data_misfit = 0.5 * np.linalg.norm(residual)**2
            model_norm = 0.5 * lambda_reg**2 * np.linalg.norm(self.L @ m)**2
            objective = data_misfit + model_norm
            
            history['data_misfit'].append(data_misfit)
            history['model_norm'].append(model_norm)
            history['objective'].append(objective)
            
            # Check convergence
            if iteration > 0:
                obj_change = abs(objective - history['objective'][-2]) / objective
                if obj_change < tol:
                    print(f"Converged at iteration {iteration}")
                    break
            
            # Gauss-Newton step: solve (J^T J + λ^2 L^T L) δm = J^T r - λ^2 L^T L m
            JTJ = J.T @ J
            LTL = self.L.T @ self.L
            JTr = J.T @ residual
            
            A = JTJ + lambda_reg**2 * LTL
            b = JTr - lambda_reg**2 * LTL @ m
            
            delta_m = np.linalg.solve(A, b)
            
            # Line search
            if line_search:
                alpha = self._line_search(m, delta_m, data, lambda_reg, objective)
            else:
                alpha = 1.0
            
            history['step_length'].append(alpha)
            
            # Update model
            m = m + alpha * delta_m
            history['models'].append(m.copy())
        
        # Final evaluation
        d_pred = self.forward(m)
        final_residual = np.linalg.norm(data - d_pred)
        
        return {
            'model': m,
            'predicted_data': d_pred,
            'residual': final_residual,
            'iterations': iteration + 1,
            'history': history,
            'converged': iteration < max_iter - 1
        }
    
    def _line_search(self, m: np.ndarray, delta_m: np.ndarray, 
                     data: np.ndarray, lambda_reg: float, 
                     current_obj: float) -> float:
        """Backtracking line search for step length."""
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        for _ in range(10):
            m_new = m + alpha * delta_m
            d_pred = self.forward(m_new)
            residual = data - d_pred
            
            data_misfit = 0.5 * np.linalg.norm(residual)**2
            model_norm = 0.5 * lambda_reg**2 * np.linalg.norm(self.L @ m_new)**2
            new_obj = data_misfit + model_norm
            
            # Armijo condition
            if new_obj <= current_obj - c * alpha * np.linalg.norm(delta_m)**2:
                break
            
            alpha *= rho
        
        return alpha


class BayesianMAPSolver:
    """
    Bayesian Maximum A Posteriori (MAP) estimator with uncertainty quantification.
    
    Assumes Gaussian priors and likelihoods:
    - Prior: m ~ N(m_prior, C_m)
    - Likelihood: d|m ~ N(G(m), C_d)
    
    Computes posterior mean and covariance.
    """
    
    def __init__(self, forward_func: Callable, jacobian_func: Callable):
        """
        Parameters
        ----------
        forward_func : callable
            Forward modeling function
        jacobian_func : callable
            Jacobian function
        """
        self.forward = forward_func
        self.jacobian = jacobian_func
    
    def solve(self, data: np.ndarray, m_prior: np.ndarray,
              C_m: np.ndarray, C_d: np.ndarray,
              max_iter: int = 20, tol: float = 1e-6) -> Dict:
        """
        Compute MAP estimate and posterior covariance.
        
        Parameters
        ----------
        data : ndarray
            Observed data
        m_prior : ndarray
            Prior mean model
        C_m : ndarray
            Prior model covariance
        C_d : ndarray
            Data covariance (measurement errors)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        
        Returns
        -------
        result : dict
            MAP estimate, posterior covariance, uncertainties
        """
        # Compute inverse covariances
        C_m_inv = np.linalg.inv(C_m)
        C_d_inv = np.linalg.inv(C_d)
        
        # Initialize with prior
        m = m_prior.copy()
        
        # Gauss-Newton iterations for MAP estimate
        for iteration in range(max_iter):
            d_pred = self.forward(m)
            residual = data - d_pred
            J = self.jacobian(m)
            
            # MAP objective: -log p(m|d) = 0.5*(m-m_prior)^T C_m^{-1} (m-m_prior) 
            #                              + 0.5*(d-F(m))^T C_d^{-1} (d-F(m))
            
            # Gradient
            grad = C_m_inv @ (m - m_prior) - J.T @ C_d_inv @ residual
            
            # Hessian (Gauss-Newton approximation)
            H = C_m_inv + J.T @ C_d_inv @ J
            
            # Newton step
            delta_m = np.linalg.solve(H, -grad)
            
            # Update
            m = m + delta_m
            
            # Check convergence
            if np.linalg.norm(delta_m) / np.linalg.norm(m) < tol:
                break
        
        # Posterior covariance (at MAP point)
        J_map = self.jacobian(m)
        C_post = np.linalg.inv(C_m_inv + J_map.T @ C_d_inv @ J_map)
        
        # Marginal uncertainties
        uncertainties = np.sqrt(np.diag(C_post))
        
        # Model resolution matrix
        R = C_post @ C_m_inv
        
        return {
            'model_map': m,
            'posterior_covariance': C_post,
            'uncertainties': uncertainties,
            'resolution_matrix': R,
            'resolution_diagonal': np.diag(R),
            'iterations': iteration + 1,
            'predicted_data': self.forward(m)
        }
    
    def credible_intervals(self, model_map: np.ndarray, 
                          C_post: np.ndarray, 
                          confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute credible intervals for model parameters.
        
        Parameters
        ----------
        model_map : ndarray
            MAP estimate
        C_post : ndarray
            Posterior covariance
        confidence : float
            Confidence level (default 95%)
        
        Returns
        -------
        lower_bound : ndarray
            Lower credible bounds
        upper_bound : ndarray
            Upper credible bounds
        """
        from scipy.stats import norm
        
        z_score = norm.ppf((1 + confidence) / 2)
        uncertainties = np.sqrt(np.diag(C_post))
        
        lower = model_map - z_score * uncertainties
        upper = model_map + z_score * uncertainties
        
        return lower, upper


class UncertaintyAnalysis:
    """
    Tools for analyzing resolution and uncertainty in inverse problems.
    """
    
    @staticmethod
    def compute_resolution_metrics(R: np.ndarray) -> Dict:
        """
        Compute resolution metrics from resolution matrix.
        
        Parameters
        ----------
        R : ndarray
            Model resolution matrix
        
        Returns
        -------
        metrics : dict
            Various resolution metrics
        """
        n = R.shape[0]
        
        # Diagonal resolution
        r_diag = np.diag(R)
        
        # Spread function (row sums)
        spread = np.sum(np.abs(R), axis=1)
        
        # Model covariance eigenvalues
        eigenvalues = np.linalg.eigvalsh(R)
        
        return {
            'resolution_diagonal': r_diag,
            'mean_resolution': np.mean(r_diag),
            'spread': spread,
            'eigenvalues': eigenvalues,
            'effective_rank': np.sum(eigenvalues) / np.max(eigenvalues)
        }
    
    @staticmethod
    def plot_resolution_map(R: np.ndarray, grid_shape: Tuple[int, int] = None):
        """
        Create resolution map visualization.
        
        Parameters
        ----------
        R : ndarray
            Resolution matrix
        grid_shape : tuple, optional
            Reshape resolution diagonal to 2D grid
        
        Returns
        -------
        resolution_map : ndarray
            Resolution values reshaped for plotting
        """
        r_diag = np.diag(R)
        
        if grid_shape is not None:
            return r_diag.reshape(grid_shape)
        return r_diag
    
    @staticmethod
    def uncertainty_map(uncertainties: np.ndarray, 
                       grid_shape: Tuple[int, int] = None) -> np.ndarray:
        """
        Create uncertainty map from marginal uncertainties.
        
        Parameters
        ----------
        uncertainties : ndarray
            Marginal standard deviations
        grid_shape : tuple, optional
            Reshape to 2D grid
        
        Returns
        -------
        uncertainty_map : ndarray
            Uncertainties reshaped for plotting
        """
        if grid_shape is not None:
            return uncertainties.reshape(grid_shape)
        return uncertainties
