"""
Guidance, Navigation, and Control (GNC) Package

This package provides complete GNC capabilities for satellite formation flying:

## Controllers (control.controllers)
- LQR: Linear Quadratic Regulator for optimal control
- LQG: Linear Quadratic Gaussian combining control with estimation
- MPC: Model Predictive Control with constraints
- Station-keeping: Fuel-optimal maintenance strategies
- Collision avoidance: Safety monitoring and maneuver planning

## Navigation (control.navigation)
- EKF: Extended Kalman Filter for nonlinear state estimation
- GPS and laser interferometry measurement processing
- Multi-sensor fusion capabilities

Example:
    >>> from control.controllers import FormationLQRController
    >>> from control.navigation import RelativeNavigationEKF
    >>>
    >>> # Create controller and estimator
    >>> controller = FormationLQRController(n=0.001)  # n = mean motion
    >>> estimator = RelativeNavigationEKF(n=0.001)
    >>>
    >>> # Control loop
    >>> state_est = estimator.update(measurement)
    >>> control = controller.compute_control(state_est)
"""

from . import controllers
from . import navigation

__all__ = ['controllers', 'navigation']
__version__ = '0.3.0'
