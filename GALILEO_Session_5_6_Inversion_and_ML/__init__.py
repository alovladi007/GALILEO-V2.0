"""
Geophysical Inversion Engine
=============================

Comprehensive toolkit for solving geophysical inverse problems.
"""

from .solvers import (
    TikhonovSolver,
    GaussNewtonSolver,
    BayesianMAPSolver,
    UncertaintyAnalysis
)

from .regularizers import (
    SmoothnessRegularizer,
    TotalVariationRegularizer,
    SparsityRegularizer,
    GeologicPriorRegularizer,
    CrossGradientRegularizer,
    MinimumSupportRegularizer
)

__all__ = [
    'TikhonovSolver',
    'GaussNewtonSolver',
    'BayesianMAPSolver',
    'UncertaintyAnalysis',
    'SmoothnessRegularizer',
    'TotalVariationRegularizer',
    'SparsityRegularizer',
    'GeologicPriorRegularizer',
    'CrossGradientRegularizer',
    'MinimumSupportRegularizer'
]
