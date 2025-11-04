"""
Formation control algorithms for satellite constellations.

This module provides various control strategies for formation flying:
- LQR: Linear Quadratic Regulator for optimal control
- LQG: Linear Quadratic Gaussian combining LQR with Kalman filtering
- MPC: Model Predictive Control with constraints
- Station-keeping: Fuel-optimal maintenance strategies
- Collision avoidance: Safety monitoring and avoidance maneuvers
"""

from .lqr import (
    solve_continuous_riccati,
    solve_discrete_riccati,
    lqr_gain_continuous,
    lqr_gain_discrete,
    hill_clohessy_wiltshire_matrices,
    FormationLQRController,
    compute_controllability_gramian,
    minimum_energy_control,
    design_formation_weights,
)

from .lqg import (
    kalman_filter_continuous,
    kalman_filter_discrete,
    LQGController,
    FormationLQGController,
)

try:
    from .mpc import (
        MPCController,
        FormationMPCController,
        FuelOptimalMPC,
        compute_mpc_feedback_gain,
        design_mpc_weights,
        compute_reachable_set,
    )
except ImportError:
    # MPC requires cvxpy
    pass

from .station_keeping import (
    StationKeepingBox,
    DeadBandController,
    ImpulsiveManeuverPlanner,
    LongTermStationKeeper,
    estimate_annual_fuel,
    optimize_box_size,
)

from .collision_avoidance import (
    CollisionEvent,
    CollisionDetector,
    CollisionAvoidanceManeuver,
    FormationSafetyMonitor,
    compute_separation_matrix,
    predict_collision_risk,
)

__all__ = [
    # LQR functions
    'solve_continuous_riccati',
    'solve_discrete_riccati',
    'lqr_gain_continuous',
    'lqr_gain_discrete',
    'hill_clohessy_wiltshire_matrices',
    'FormationLQRController',
    'compute_controllability_gramian',
    'minimum_energy_control',
    'design_formation_weights',
    # LQG functions
    'kalman_filter_continuous',
    'kalman_filter_discrete',
    'LQGController',
    'FormationLQGController',
    # MPC functions
    'MPCController',
    'FormationMPCController',
    'FuelOptimalMPC',
    'compute_mpc_feedback_gain',
    'design_mpc_weights',
    'compute_reachable_set',
    # Station-keeping
    'StationKeepingBox',
    'DeadBandController',
    'ImpulsiveManeuverPlanner',
    'LongTermStationKeeper',
    'estimate_annual_fuel',
    'optimize_box_size',
    # Collision avoidance
    'CollisionEvent',
    'CollisionDetector',
    'CollisionAvoidanceManeuver',
    'FormationSafetyMonitor',
    'compute_separation_matrix',
    'predict_collision_risk',
]
