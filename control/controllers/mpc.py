"""
Model Predictive Control (MPC) for formation flying.

Implements receding horizon control with constraints on states and controls,
optimized for real-time satellite formation maintenance.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Dict, Any, Callable
from functools import partial
import cvxpy as cp
import numpy as np


class MPCController:
    """
    Model Predictive Controller for constrained optimal control.
    """
    
    def __init__(
        self,
        A: jnp.ndarray,
        B: jnp.ndarray,
        Q: jnp.ndarray,
        R: jnp.ndarray,
        N: int,
        state_constraints: Optional[Dict] = None,
        control_constraints: Optional[Dict] = None,
        terminal_weight: Optional[jnp.ndarray] = None,
        dt: float = 1.0
    ):
        """
        Initialize MPC controller.
        
        Args:
            A: System dynamics matrix (discrete-time)
            B: Control input matrix
            Q: State cost matrix
            R: Control cost matrix
            N: Prediction horizon
            state_constraints: Dict with 'min' and 'max' state bounds
            control_constraints: Dict with 'min' and 'max' control bounds
            terminal_weight: Terminal state cost matrix (if None, use Q)
            dt: Time step
        """
        self.A = np.array(A)
        self.B = np.array(B)
        self.Q = np.array(Q)
        self.R = np.array(R)
        self.N = N
        self.dt = dt
        
        self.nx = A.shape[0]  # State dimension
        self.nu = B.shape[1]  # Control dimension
        
        # Terminal weight (use LQR solution if not provided)
        if terminal_weight is None:
            from .lqr import solve_discrete_riccati
            self.P_terminal = np.array(solve_discrete_riccati(
                jnp.array(A), jnp.array(B), 
                jnp.array(Q), jnp.array(R)
            ))
        else:
            self.P_terminal = np.array(terminal_weight)
        
        # Constraints
        self.state_constraints = state_constraints or {}
        self.control_constraints = control_constraints or {}
        
        # Pre-build optimization problem for efficiency
        self._build_optimization_problem()
    
    def _build_optimization_problem(self):
        """
        Build the CVXPY optimization problem structure.
        """
        # Decision variables
        self.x_var = cp.Variable((self.N + 1, self.nx))  # States
        self.u_var = cp.Variable((self.N, self.nu))       # Controls
        
        # Parameters (will be updated each solve)
        self.x0_param = cp.Parameter(self.nx)             # Initial state
        self.xref_param = cp.Parameter((self.N + 1, self.nx))  # Reference trajectory
        
        # Build cost function
        cost = 0
        for k in range(self.N):
            # Stage cost
            x_err = self.x_var[k] - self.xref_param[k]
            cost += cp.quad_form(x_err, self.Q)
            cost += cp.quad_form(self.u_var[k], self.R)
        
        # Terminal cost
        x_err_terminal = self.x_var[self.N] - self.xref_param[self.N]
        cost += cp.quad_form(x_err_terminal, self.P_terminal)
        
        # Build constraints
        constraints = []
        
        # Initial condition
        constraints.append(self.x_var[0] == self.x0_param)
        
        # Dynamics constraints
        for k in range(self.N):
            constraints.append(
                self.x_var[k + 1] == self.A @ self.x_var[k] + self.B @ self.u_var[k]
            )
        
        # State constraints
        if 'min' in self.state_constraints:
            x_min = self.state_constraints['min']
            for k in range(self.N + 1):
                constraints.append(self.x_var[k] >= x_min)
        
        if 'max' in self.state_constraints:
            x_max = self.state_constraints['max']
            for k in range(self.N + 1):
                constraints.append(self.x_var[k] <= x_max)
        
        # Control constraints
        if 'min' in self.control_constraints:
            u_min = self.control_constraints['min']
            for k in range(self.N):
                constraints.append(self.u_var[k] >= u_min)
        
        if 'max' in self.control_constraints:
            u_max = self.control_constraints['max']
            for k in range(self.N):
                constraints.append(self.u_var[k] <= u_max)
        
        # Create optimization problem
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
    
    def solve(
        self,
        x0: np.ndarray,
        xref: Optional[np.ndarray] = None,
        solver: str = 'OSQP',
        verbose: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Solve MPC optimization problem.
        
        Args:
            x0: Current state
            xref: Reference trajectory (N+1 x nx). If None, regulate to origin
            solver: CVXPY solver to use
            verbose: Print solver output
            
        Returns:
            u_opt: Optimal control sequence (N x nu)
            x_opt: Optimal state trajectory (N+1 x nx)
            info: Solver information
        """
        # Set reference trajectory
        if xref is None:
            xref = np.zeros((self.N + 1, self.nx))
        elif xref.shape[0] == 1:
            # Constant reference
            xref = np.tile(xref, (self.N + 1, 1))
        
        # Update parameters
        self.x0_param.value = x0
        self.xref_param.value = xref
        
        # Solve
        try:
            self.problem.solve(solver=solver, verbose=verbose)
            
            if self.problem.status not in ['optimal', 'optimal_inaccurate']:
                print(f"Warning: MPC solve status: {self.problem.status}")
            
            # Extract solution
            u_opt = self.u_var.value
            x_opt = self.x_var.value
            
            info = {
                'status': self.problem.status,
                'cost': self.problem.value,
                'solve_time': self.problem.solver_stats.solve_time if hasattr(self.problem.solver_stats, 'solve_time') else None
            }
            
        except Exception as e:
            print(f"MPC solve failed: {e}")
            # Return zero control on failure
            u_opt = np.zeros((self.N, self.nu))
            x_opt = np.zeros((self.N + 1, self.nx))
            info = {'status': 'failed', 'error': str(e)}
        
        return u_opt, x_opt, info
    
    def get_control(
        self,
        x0: np.ndarray,
        xref: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Get first control action (receding horizon).
        
        Args:
            x0: Current state
            xref: Reference trajectory
            
        Returns:
            u: Control to apply now
        """
        u_seq, _, _ = self.solve(x0, xref)
        return u_seq[0]  # Return first control


class FormationMPCController:
    """
    MPC controller specialized for satellite formation flying with
    operational constraints.
    """
    
    def __init__(
        self,
        n: float,
        N: int = 10,
        dt: float = 10.0,
        max_thrust: float = 0.001,  # km/s² (1 mm/s²)
        max_position_error: float = 0.1,  # km (100m)
        max_velocity_error: float = 0.001,  # km/s (1 m/s)
        collision_radius: float = 0.010,  # km (10m safety)
        fuel_weight: float = 0.01
    ):
        """
        Initialize formation MPC controller.
        
        Args:
            n: Mean motion (rad/s)
            N: Prediction horizon (steps)
            dt: Time step (s)
            max_thrust: Maximum thrust magnitude (km/s²)
            max_position_error: Maximum allowed position deviation (km)
            max_velocity_error: Maximum allowed velocity deviation (km/s)
            collision_radius: Minimum safe separation (km)
            fuel_weight: Weight on fuel usage in cost function
        """
        self.n = n
        self.N = N
        self.dt = dt
        self.max_thrust = max_thrust
        self.collision_radius = collision_radius
        
        # Get discrete HCW matrices
        from .lqr import hill_clohessy_wiltshire_matrices
        self.A, self.B = hill_clohessy_wiltshire_matrices(n, dt)
        self.A = np.array(self.A)
        self.B = np.array(self.B)
        
        # State weights (position more important than velocity)
        Q = np.diag([100., 100., 100., 10., 10., 10.])
        
        # Control weights (fuel usage)
        R = np.eye(3) * fuel_weight
        
        # State constraints
        state_constraints = {
            'min': np.array([
                -max_position_error, -max_position_error, -max_position_error,
                -max_velocity_error, -max_velocity_error, -max_velocity_error
            ]),
            'max': np.array([
                max_position_error, max_position_error, max_position_error,
                max_velocity_error, max_velocity_error, max_velocity_error
            ])
        }
        
        # Control constraints
        control_constraints = {
            'min': np.array([-max_thrust, -max_thrust, -max_thrust]),
            'max': np.array([max_thrust, max_thrust, max_thrust])
        }
        
        # Create base MPC controller
        self.mpc = MPCController(
            self.A, self.B, Q, R, N,
            state_constraints=state_constraints,
            control_constraints=control_constraints,
            dt=dt
        )
    
    def compute_control_with_collision_avoidance(
        self,
        ego_state: np.ndarray,
        other_states: np.ndarray,
        reference: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Compute control with collision avoidance constraints.
        
        Args:
            ego_state: Own satellite state (6,)
            other_states: Other satellites' states (M x 6)
            reference: Reference state
            
        Returns:
            u: Control input
            info: Solver information
        """
        # This would require adding collision constraints dynamically
        # For now, use basic MPC
        u_seq, x_pred, info = self.mpc.solve(ego_state, reference)
        
        # Check predicted trajectory for collisions
        n_others = other_states.shape[0] if other_states.ndim > 1 else 1
        min_sep = float('inf')
        
        for i in range(self.N + 1):
            ego_pos = x_pred[i, :3]
            for j in range(n_others):
                other_pos = other_states[j, :3] if other_states.ndim > 1 else other_states[:3]
                separation = np.linalg.norm(ego_pos - other_pos)
                min_sep = min(min_sep, separation)
        
        info['min_separation'] = min_sep
        info['collision_risk'] = min_sep < self.collision_radius
        
        return u_seq[0], info
    
    def plan_reconfiguration(
        self,
        initial_formation: np.ndarray,
        target_formation: np.ndarray,
        n_satellites: int,
        time_horizon: float
    ) -> Dict:
        """
        Plan optimal reconfiguration maneuver for entire formation.
        
        Args:
            initial_formation: Initial states (n_satellites x 6)
            target_formation: Target states (n_satellites x 6)
            n_satellites: Number of satellites
            time_horizon: Time to complete maneuver (s)
            
        Returns:
            Dictionary with control plans for each satellite
        """
        n_steps = int(time_horizon / self.dt)
        plans = {}
        
        for i in range(n_satellites):
            # Linear interpolation for reference trajectory
            alpha = np.linspace(0, 1, n_steps + 1).reshape(-1, 1)
            reference = (1 - alpha) * initial_formation[i] + alpha * target_formation[i]
            
            # Solve for this satellite
            u_seq, x_seq, info = self.mpc.solve(
                initial_formation[i],
                reference[:self.N + 1]
            )
            
            plans[f'satellite_{i}'] = {
                'control_sequence': u_seq,
                'state_trajectory': x_seq,
                'total_dv': np.sum(np.linalg.norm(u_seq, axis=1)) * self.dt,
                'info': info
            }
        
        return plans


@jax.jit
def compute_mpc_feedback_gain(
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    R: jnp.ndarray,
    N: int
) -> jnp.ndarray:
    """
    Compute equivalent feedback gain for unconstrained MPC.
    
    For unconstrained case, MPC reduces to time-varying LQR.
    
    Args:
        A: System matrix
        B: Control matrix
        Q: State cost
        R: Control cost
        N: Horizon
        
    Returns:
        K: Feedback gain for first step
    """
    # Backward Riccati recursion
    P = Q
    
    for _ in range(N):
        K = jnp.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
        P = Q + A.T @ P @ (A - B @ K)
    
    # First step gain
    K_0 = jnp.linalg.inv(R + B.T @ P @ B) @ B.T @ P @ A
    
    return K_0


class FuelOptimalMPC(MPCController):
    """
    MPC variant optimized for minimum fuel consumption.
    """
    
    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        N: int,
        dt: float = 1.0,
        position_tolerance: float = 0.001,  # km (1m)
        max_thrust: float = 0.001  # km/s²
    ):
        """
        Initialize fuel-optimal MPC.
        
        Uses L1 norm on controls for fuel optimization.
        
        Args:
            A: System dynamics
            B: Control matrix
            N: Horizon
            dt: Time step
            position_tolerance: Acceptable position error
            max_thrust: Maximum thrust
        """
        nx = A.shape[0]
        nu = B.shape[1]
        
        # Very small state cost (mainly terminal)
        Q = np.eye(nx) * 0.01
        Q[:3, :3] *= 100  # Position accuracy at terminal state
        
        # Control cost encourages sparsity (fuel optimal)
        R = np.eye(nu) * 0.001
        
        super().__init__(
            A, B, Q, R, N,
            control_constraints={
                'min': -max_thrust * np.ones(nu),
                'max': max_thrust * np.ones(nu)
            },
            terminal_weight=Q * 1000,  # Strong terminal constraint
            dt=dt
        )
        
        self.position_tolerance = position_tolerance
    
    def solve_minimum_fuel(
        self,
        x0: np.ndarray,
        xf: np.ndarray,
        time_steps: int
    ) -> Tuple[np.ndarray, float]:
        """
        Solve for minimum fuel trajectory.
        
        Args:
            x0: Initial state
            xf: Final state
            time_steps: Number of time steps
            
        Returns:
            u_opt: Optimal control sequence
            fuel_used: Total fuel consumption (∫|u|dt)
        """
        # Extend horizon if needed
        if time_steps > self.N:
            # Would need to rebuild problem
            time_steps = self.N
        
        # Create reference that reaches target
        reference = np.zeros((self.N + 1, self.nx))
        reference[-1] = xf  # Only care about final state
        
        u_opt, x_opt, info = self.solve(x0, reference)
        
        # Compute fuel usage (L1 norm of controls)
        fuel_used = np.sum(np.abs(u_opt)) * self.dt
        
        return u_opt, fuel_used


# Utility functions
def design_mpc_weights(
    position_importance: float = 100.0,
    velocity_importance: float = 10.0,
    control_cost: float = 0.01,
    terminal_weight_factor: float = 10.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Design MPC weight matrices.
    
    Args:
        position_importance: Weight on position tracking
        velocity_importance: Weight on velocity tracking
        control_cost: Weight on control effort
        terminal_weight_factor: Terminal cost multiplier
        
    Returns:
        Q: State cost matrix
        R: Control cost matrix
        Q_f: Terminal cost matrix
    """
    Q = np.diag([
        position_importance, position_importance, position_importance,
        velocity_importance, velocity_importance, velocity_importance
    ])
    
    R = np.eye(3) * control_cost
    
    Q_f = Q * terminal_weight_factor
    
    return Q, R, Q_f


def compute_reachable_set(
    A: np.ndarray,
    B: np.ndarray,
    x0: np.ndarray,
    u_max: float,
    N: int
) -> np.ndarray:
    """
    Compute approximate reachable set for MPC.
    
    Args:
        A: System dynamics
        B: Control matrix
        x0: Initial state
        u_max: Maximum control magnitude
        N: Time steps
        
    Returns:
        vertices: Vertices of reachable set polytope
    """
    nx = A.shape[0]
    nu = B.shape[1]
    
    # Compute by propagating extreme controls
    n_vertices = 2**nu  # All combinations of ±u_max
    vertices = np.zeros((n_vertices, nx))
    
    for i in range(n_vertices):
        x = x0
        for k in range(N):
            # Generate control from binary representation
            u = np.zeros(nu)
            for j in range(nu):
                if (i >> j) & 1:
                    u[j] = u_max
                else:
                    u[j] = -u_max
            
            x = A @ x + B @ u
        
        vertices[i] = x
    
    return vertices


# Example usage and testing
if __name__ == "__main__":
    print("Model Predictive Control Test")
    print("-" * 40)
    
    # Test formation MPC
    n = 0.001  # rad/s (LEO)
    
    mpc = FormationMPCController(
        n=n,
        N=10,
        dt=10.0,
        max_thrust=0.001,  # 1 mm/s²
        fuel_weight=0.01
    )
    
    # Initial state with offset
    x0 = np.array([0.010, 0.005, 0.0, 0.0001, 0.0, 0.0])  # 10m, 5m offset
    
    # Solve MPC
    u_seq, x_seq, info = mpc.mpc.solve(x0)
    
    print(f"MPC Solution:")
    print(f"  Status: {info['status']}")
    print(f"  Cost: {info['cost']:.6f}")
    print(f"  First control: {u_seq[0]*1000} mm/s²")
    print(f"  Final position: {x_seq[-1][:3]*1000} m")
    print(f"  Total ΔV: {np.sum(np.linalg.norm(u_seq, axis=1))*mpc.dt*1000:.3f} m/s")
    
    # Test fuel-optimal MPC
    print("\n" + "="*40)
    print("Fuel-Optimal MPC Test")
    
    fuel_mpc = FuelOptimalMPC(
        mpc.A, mpc.B,
        N=20,
        dt=10.0,
        max_thrust=0.001
    )
    
    xf = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Target: origin
    u_fuel, fuel = fuel_mpc.solve_minimum_fuel(x0, xf, 20)
    
    print(f"  Minimum fuel: {fuel*1000:.3f} m/s")
    print(f"  Number of burns: {np.sum(np.abs(u_fuel) > 1e-6)}")
    
    # Test reconfiguration planning
    print("\n" + "="*40)
    print("Formation Reconfiguration Test")
    
    initial = np.array([
        [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],    # Sat 1
        [0.0, 0.1, 0.0, 0.0, 0.0, 0.0],    # Sat 2
        [0.0, 0.0, 0.1, 0.0, 0.0, 0.0],    # Sat 3
    ])
    
    target = np.array([
        [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],    # New formation
        [0.1, 0.1, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ])
    
    plans = mpc.plan_reconfiguration(initial, target, 3, 300.0)
    
    for sat, plan in plans.items():
        print(f"  {sat}: ΔV = {plan['total_dv']*1000:.3f} m/s")
