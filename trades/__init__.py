"""
Mission Trade Studies Package
==============================

Comprehensive trade study framework for space mission design optimization,
featuring multi-objective analysis, Pareto front identification, and
decision support visualization.

Modules:
--------
- baseline_study: Baseline length vs noise vs sensitivity analysis
- orbit_study: Orbit altitude & inclination vs coverage analysis
- optical_study: Optical power & aperture tradeoffs
- pareto_analysis: Multi-objective optimization and Pareto fronts

Usage:
------
>>> from trades.baseline_study import BaselineTradeStudy
>>> study = BaselineTradeStudy()
>>> results = study.run_trade_study()
>>> study.plot_results(results, 'plots/')

Session 12 - Design Trade and Sensitivity Studies
"""

__version__ = "1.0.0"
__author__ = "GALILEO Mission Design Team"

from .baseline_study import BaselineTradeStudy
from .orbit_study import OrbitTradeStudy
from .optical_study import OpticalTradeStudy
from .pareto_analysis import ParetoAnalysis

__all__ = [
    "BaselineTradeStudy",
    "OrbitTradeStudy",
    "OpticalTradeStudy",
    "ParetoAnalysis",
]
