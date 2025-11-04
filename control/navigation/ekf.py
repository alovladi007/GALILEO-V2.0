"""
Extended Kalman Filter (EKF) for satellite navigation.

Implements EKF for nonlinear orbital dynamics and measurements,
using JAX autodiff for automatic Jacobian computation.
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional, Dict, Any
from functools import partial


@jax.jit
def compute_jacobian_autodiff(
    func: Callable,
    x: jnp.ndarray,
    *args
) -> jnp.ndarray:
    """
    Compute Jacobian using JAX automatic differentiation.
    
    Args:
        func: Function f(x, *args) to differentiate
        x: Point at which to compute Jacobian
        *args: Additional arguments to func
        
    Returns:
        J: Jacobian matrix df/dx
    """
    return jax.jacfwd(func, argnums=0)(x, *args)


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation.
    """
    
    def __init__(
        self,
        dynamics_func: Callable,
        measurement_func: Callable,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        dt: float,
        state_dim: int,
        meas_dim: int,
        use_autodiff: bool = True
    ):
        """
        Initialize EKF.
        
        Args:
            dynamics_func: Nonlinear dynamics f(x, u, t)
            measurement_func: Nonlinear measurement h(x, t)
            Q: Process noise covariance (n x n)
            R: Measurement noise covariance (m x m)
            dt: Time step
            state_dim: State dimension
            meas_dim: Measurement dimension
            use_autodiff: Use JAX autodiff for Jacobians
        """
        self.f = dynamics_func
        self.h = measurement_func
        self.Q = Q
        self.R = R
        self.dt = dt
        self.n = state_dim
        self.m = meas_dim
        self.use_autodiff = use_autodiff
        
        # Initialize state and covariance
        self.x = jnp.zeros(state_dim)
        self.P = jnp.eye(state_dim)
        
        # Jacobian functions
        if use_autodiff:
            self.F_func = jax.jit(jax.jacfwd(dynamics_func, argnums=0))
            self.H_func = jax.jit(jax.jacfwd(measurement_func, argnums=0))
        else:
            self.F_func = None
            self.H_func = None
    
    @partial(jax.jit, static_argnums=(0,))
    def predict(
        self,
        x: jnp.ndarray,
        P: jnp.ndarray,
        u: Optional[jnp.ndarray] = None,
        t: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        EKF prediction step.
        
        Args:
            x: Current state estimate
            P: Current error covariance
            u: Control input
            t: Current time
            
        Returns:
            x_pred: Predicted state
            P_pred: Predicted covariance
        """
        # Propagate state
        if u is None:
            u = jnp.zeros(3)  # No control
        
        # Nonlinear dynamics
        x_dot = self.f(x, u, t)
        x_pred = x + x_dot * self.dt  # Euler integration
        
        # Linearize dynamics
        if self.use_autodiff:
            F = self.F_func(x, u, t)
        else:
            # Numerical Jacobian if autodiff not available
            F = self._numerical_jacobian(self.f, x, u, t)
        
        # Discrete-time state transition matrix
        Phi = jnp.eye(self.n) + F * self.dt
        
        # Propagate covariance
        P_pred = Phi @ P @ Phi.T + self.Q
        
        # Ensure symmetry
        P_pred = 0.5 * (P_pred + P_pred.T)
        
        return x_pred, P_pred
    
    @partial(jax.jit, static_argnums=(0,))
    def update(
        self,
        x_pred: jnp.ndarray,
        P_pred: jnp.ndarray,
        y: jnp.ndarray,
        t: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        EKF update (correction) step.
        
        Args:
            x_pred: Predicted state
            P_pred: Predicted covariance
            y: Measurement
            t: Current time
            
        Returns:
            x: Updated state estimate
            P: Updated error covariance
            innovation: Measurement innovation
        """
        # Predicted measurement
        y_pred = self.h(x_pred, t)
        
        # Linearize measurement function
        if self.use_autodiff:
            H = self.H_func(x_pred, t)
        else:
            H = self._numerical_jacobian(lambda x: self.h(x, t), x_pred)
        
        # Innovation (residual)
        innovation = y - y_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + self.R
        
        # Kalman gain
        K = P_pred @ H.T @ jnp.linalg.inv(S)
        
        # Update state
        x = x_pred + K @ innovation
        
        # Update covariance (Joseph form for numerical stability)
        I_KH = jnp.eye(self.n) - K @ H
        P = I_KH @ P_pred @ I_KH.T + K @ self.R @ K.T
        
        # Ensure symmetry
        P = 0.5 * (P + P.T)
        
        return x, P, innovation
    
    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        x: jnp.ndarray,
        P: jnp.ndarray,
        y: jnp.ndarray,
        u: Optional[jnp.ndarray] = None,
        t: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, Any]]:
        """
        Single EKF step (predict + update).
        
        Args:
            x: Current state estimate
            P: Current error covariance
            y: Measurement
            u: Control input
            t: Current time
            
        Returns:
            x: Updated state
            P: Updated covariance
            info: Dictionary with diagnostic information
        """
        # Prediction
        x_pred, P_pred = self.predict(x, P, u, t)
        
        # Update
        x_new, P_new, innovation = self.update(x_pred, P_pred, y, t)
        
        # Compute normalized innovation squared (NIS)
        H = self.H_func(x_pred, t) if self.use_autodiff else \
            self._numerical_jacobian(lambda x: self.h(x, t), x_pred)
        S = H @ P_pred @ H.T + self.R
        nis = innovation.T @ jnp.linalg.inv(S) @ innovation
        
        info = {
            'x_pred': x_pred,
            'P_pred': P_pred,
            'innovation': innovation,
            'nis': nis,
            'P_trace': jnp.trace(P_new)
        }
        
        return x_new, P_new, info
    
    def _numerical_jacobian(
        self,
        func: Callable,
        x: jnp.ndarray,
        *args,
        eps: float = 1e-7
    ) -> jnp.ndarray:
        """
        Compute Jacobian using finite differences.
        
        Args:
            func: Function to differentiate
            x: Point at which to compute Jacobian
            *args: Additional arguments to func
            eps: Finite difference step
            
        Returns:
            J: Jacobian matrix
        """
        n = x.shape[0]
        f0 = func(x, *args)
        m = f0.shape[0] if f0.ndim > 0 else 1
        
        J = jnp.zeros((m, n))
        
        for i in range(n):
            x_plus = x.at[i].add(eps)
            x_minus = x.at[i].add(-eps)
            
            f_plus = func(x_plus, *args)
            f_minus = func(x_minus, *args)
            
            J = J.at[:, i].set((f_plus - f_minus) / (2 * eps))
        
        return J


class OrbitalEKF(ExtendedKalmanFilter):
    """
    EKF specialized for orbital state estimation.
    """
    
    def __init__(
        self,
        mu: float = 398600.4418,
        process_noise_std: float = 1e-6,
        gps_noise_std: float = 10e-3,
        range_noise_std: float = 1e-3,
        dt: float = 1.0
    ):
        """
        Initialize orbital EKF.
        
        Args:
            mu: Gravitational parameter (km³/s²)
            process_noise_std: Process noise (km/s²)
            gps_noise_std: GPS position noise (km)
            range_noise_std: Range measurement noise (km)
            dt: Time step (s)
        """
        self.mu = mu
        
        # Nonlinear orbital dynamics
        def orbital_dynamics(x, u, t):
            """Two-body dynamics with optional control."""
            r = x[:3]
            v = x[3:6]
            
            r_norm = jnp.linalg.norm(r)
            a_grav = -mu / r_norm**3 * r
            
            x_dot = jnp.zeros(6)
            x_dot = x_dot.at[:3].set(v)
            x_dot = x_dot.at[3:].set(a_grav + u)
            
            return x_dot
        
        # GPS measurement (position only)
        def gps_measurement(x, t):
            """GPS position measurement."""
            return x[:3]
        
        # Process noise (acceleration uncertainty)
        Q = jnp.zeros((6, 6))
        Q = Q.at[3:, 3:].set(jnp.eye(3) * process_noise_std**2)
        
        # Measurement noise (GPS)
        R = jnp.eye(3) * gps_noise_std**2
        
        super().__init__(
            dynamics_func=orbital_dynamics,
            measurement_func=gps_measurement,
            Q=Q,
            R=R,
            dt=dt,
            state_dim=6,
            meas_dim=3,
            use_autodiff=True
        )
        
        self.gps_noise_std = gps_noise_std
        self.range_noise_std = range_noise_std
    
    def add_range_measurement(
        self,
        x: jnp.ndarray,
        P: jnp.ndarray,
        range_meas: float,
        station_pos: jnp.ndarray,
        t: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process range measurement from ground station.
        
        Args:
            x: Current state estimate
            P: Current covariance
            range_meas: Measured range (km)
            station_pos: Ground station position (km)
            t: Current time
            
        Returns:
            x: Updated state
            P: Updated covariance
        """
        # Range measurement function
        def range_func(x, t):
            return jnp.linalg.norm(x[:3] - station_pos)
        
        # Predicted range
        range_pred = range_func(x, t)
        
        # Measurement Jacobian
        H_range = jax.jacfwd(range_func, argnums=0)(x, t)
        H_range = H_range.reshape(1, -1)
        
        # Innovation
        innovation = range_meas - range_pred
        
        # Innovation covariance
        S = H_range @ P @ H_range.T + self.range_noise_std**2
        
        # Kalman gain
        K = P @ H_range.T / S.squeeze()
        
        # Update
        x = x + K.squeeze() * innovation
        P = (jnp.eye(6) - jnp.outer(K.squeeze(), H_range)) @ P
        
        return x, P


class RelativeNavigationEKF:
    """
    EKF for relative navigation between two satellites.
    """
    
    def __init__(
        self,
        n: float,
        process_noise_std: float = 1e-8,
        range_noise_std: float = 1e-3,
        rangerate_noise_std: float = 1e-6,
        dt: float = 1.0
    ):
        """
        Initialize relative navigation EKF.
        
        Args:
            n: Mean motion (rad/s)
            process_noise_std: Process noise (km/s²)
            range_noise_std: Range noise (km)
            rangerate_noise_std: Range-rate noise (km/s)
            dt: Time step
        """
        self.n = n
        
        # Hill-Clohessy-Wiltshire dynamics
        def hcw_dynamics(x, u, t):
            """HCW relative dynamics."""
            dx = x[0]
            dy = x[1]
            dz = x[2]
            dvx = x[3]
            dvy = x[4]
            dvz = x[5]
            
            x_dot = jnp.array([
                dvx,
                dvy,
                dvz,
                3*n**2*dx + 2*n*dvy + u[0],
                -2*n*dvx + u[1],
                -n**2*dz + u[2]
            ])
            
            return x_dot
        
        # Range and range-rate measurements
        def range_rangerate_measurement(x, t):
            """Measure range and range-rate."""
            r = x[:3]
            v = x[3:6]
            
            range_val = jnp.linalg.norm(r)
            rangerate_val = jnp.dot(r, v) / range_val
            
            return jnp.array([range_val, rangerate_val])
        
        # Process noise
        Q = jnp.zeros((6, 6))
        Q = Q.at[3:, 3:].set(jnp.eye(3) * process_noise_std**2)
        
        # Measurement noise
        R = jnp.diag(jnp.array([
            range_noise_std**2,
            rangerate_noise_std**2
        ]))
        
        self.ekf = ExtendedKalmanFilter(
            dynamics_func=hcw_dynamics,
            measurement_func=range_rangerate_measurement,
            Q=Q,
            R=R,
            dt=dt,
            state_dim=6,
            meas_dim=2,
            use_autodiff=True
        )
    
    def process_laser_measurement(
        self,
        x: jnp.ndarray,
        P: jnp.ndarray,
        phase_meas: float,
        phase_rate_meas: float,
        wavelength: float = 1064e-9,
        t: float = 0.0
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Process laser interferometry measurement.
        
        Args:
            x: State estimate
            P: Error covariance
            phase_meas: Phase measurement (rad)
            phase_rate_meas: Phase rate (rad/s)
            wavelength: Laser wavelength (m)
            t: Time
            
        Returns:
            Updated state and covariance
        """
        # Convert phase to range
        range_meas = phase_meas * wavelength / (4 * jnp.pi) * 1e-3  # km
        rangerate_meas = phase_rate_meas * wavelength / (4 * jnp.pi) * 1e-3  # km/s
        
        y = jnp.array([range_meas, rangerate_meas])
        
        # EKF update
        x_new, P_new, _ = self.ekf.step(x, P, y, jnp.zeros(3), t)
        
        return x_new, P_new


# Utility functions
@jax.jit
def compute_consistency_nis(
    innovations: jnp.ndarray,
    S_matrices: jnp.ndarray
) -> float:
    """
    Compute Normalized Innovation Squared for consistency check.
    
    Args:
        innovations: Measurement innovations (N x m)
        S_matrices: Innovation covariances (N x m x m)
        
    Returns:
        Average NIS (should be close to measurement dimension)
    """
    N = innovations.shape[0]
    nis_values = jnp.array([
        innovations[i] @ jnp.linalg.inv(S_matrices[i]) @ innovations[i]
        for i in range(N)
    ])
    
    return jnp.mean(nis_values)


@jax.jit
def observability_gramian(
    H_history: jnp.ndarray,
    Phi_history: jnp.ndarray
) -> jnp.ndarray:
    """
    Compute observability Gramian.
    
    Args:
        H_history: Measurement matrices over time (N x m x n)
        Phi_history: State transition matrices (N x n x n)
        
    Returns:
        W_o: Observability Gramian (n x n)
    """
    N, m, n = H_history.shape
    W_o = jnp.zeros((n, n))
    
    for i in range(N):
        # Propagate to current time
        Phi_total = jnp.eye(n)
        for j in range(i):
            Phi_total = Phi_history[j] @ Phi_total
        
        # Add contribution
        W_o += Phi_total.T @ H_history[i].T @ H_history[i] @ Phi_total
    
    return W_o


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Test orbital EKF
    print("Orbital EKF Test")
    print("-" * 40)
    
    # Create EKF
    ekf = OrbitalEKF(
        process_noise_std=1e-7,
        gps_noise_std=0.01,  # 10m GPS accuracy
        dt=1.0
    )
    
    # Initial state (LEO at 500 km)
    r0 = 6378.137 + 500  # km
    v0 = np.sqrt(ekf.mu / r0)  # km/s
    
    x_true = jnp.array([r0, 0, 0, 0, v0, 0])
    x_est = x_true + jnp.array([0.01, 0.01, 0.01, 0.0001, 0.0001, 0.0001])
    P = jnp.diag(jnp.array([0.01**2, 0.01**2, 0.01**2, 
                            1e-8, 1e-8, 1e-8]))
    
    # Simulate for 100 steps
    key = jax.random.PRNGKey(42)
    for i in range(100):
        # Generate noisy GPS measurement
        key, subkey = jax.random.split(key)
        y = x_true[:3] + jax.random.normal(subkey, (3,)) * 0.01
        
        # EKF step
        x_est, P, info = ekf.step(x_est, P, y, jnp.zeros(3), i)
        
        # True dynamics
        x_dot = ekf.f(x_true, jnp.zeros(3), i)
        x_true = x_true + x_dot * ekf.dt
        
        if i % 20 == 0:
            pos_error = jnp.linalg.norm(x_est[:3] - x_true[:3]) * 1000  # m
            vel_error = jnp.linalg.norm(x_est[3:] - x_true[3:]) * 1000  # m/s
            print(f"Step {i:3d}: pos_err={pos_error:6.2f}m, "
                  f"vel_err={vel_error:6.3f}m/s, NIS={info['nis']:.2f}")
    
    print("\n" + "="*40)
    print("Relative Navigation EKF Test")
    print("-" * 40)
    
    # Test relative navigation
    n = 0.001  # rad/s (LEO)
    rel_ekf = RelativeNavigationEKF(
        n=n,
        range_noise_std=0.001,  # 1m range accuracy
        rangerate_noise_std=1e-5,  # 0.01 mm/s range-rate
        dt=1.0
    )
    
    # Initial relative state
    x_rel_true = jnp.array([0.1, 0.0, 0.0, 0.0, 0.0001, 0.0])  # 100m offset
    x_rel_est = x_rel_true * 1.1  # 10% initial error
    P_rel = jnp.diag(jnp.array([0.01**2, 0.01**2, 0.01**2,
                                 1e-8, 1e-8, 1e-8]))
    
    # Simulate
    for i in range(50):
        # Generate measurements
        r = jnp.linalg.norm(x_rel_true[:3])
        rdot = jnp.dot(x_rel_true[:3], x_rel_true[3:]) / r
        
        key, k1, k2 = jax.random.split(key, 3)
        y_rel = jnp.array([
            r + jax.random.normal(k1) * 0.001,
            rdot + jax.random.normal(k2) * 1e-5
        ])
        
        # EKF step
        x_rel_est, P_rel, info = rel_ekf.ekf.step(
            x_rel_est, P_rel, y_rel, jnp.zeros(3), i
        )
        
        # True dynamics
        x_dot = rel_ekf.ekf.f(x_rel_true, jnp.zeros(3), i)
        x_rel_true = x_rel_true + x_dot * rel_ekf.ekf.dt
        
        if i % 10 == 0:
            pos_error = jnp.linalg.norm(x_rel_est[:3] - x_rel_true[:3]) * 1000
            print(f"Step {i:3d}: rel_pos_err={pos_error:6.2f}m, "
                  f"innovation={info['innovation'][0]*1000:.3f}m")
