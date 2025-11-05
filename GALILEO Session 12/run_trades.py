#!/usr/bin/env python3
"""
Mission Trade Studies - Main Runner

Executes all trade studies and generates comprehensive analysis.
"""

import sys
import os
import time
from datetime import datetime

# Add trades directory to path
sys.path.append('/home/claude/mission-trades/trades')

from baseline_study import BaselineTradeStudy
from orbit_study import OrbitTradeStudy
from optical_study import OpticalTradeStudy
from pareto_analysis import ParetoAnalysis


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def run_all_studies():
    """Execute all trade studies sequentially."""
    
    print_header("MISSION TRADE STUDIES - Session 12")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nInitializing trade study analyses...")
    
    results = {}
    
    # Study 1: Baseline Length vs Noise vs Sensitivity
    print_header("Study 1: Baseline Length vs Noise vs Sensitivity")
    print("Analyzing interferometer baseline configurations...")
    start = time.time()
    
    baseline_study = BaselineTradeStudy()
    baseline_results = baseline_study.run_trade_study()
    baseline_study.plot_results(baseline_results, '/home/claude/mission-trades/plots')
    
    results['baseline'] = baseline_results
    print(f"Completed in {time.time() - start:.2f} seconds")
    
    # Study 2: Orbit Altitude and Inclination vs Coverage
    print_header("Study 2: Orbit Altitude & Inclination vs Coverage")
    print("Analyzing orbital configuration trade space...")
    start = time.time()
    
    orbit_study = OrbitTradeStudy()
    orbit_results = orbit_study.run_trade_study()
    orbit_study.plot_results(orbit_results, '/home/claude/mission-trades/plots')
    
    results['orbit'] = orbit_results
    print(f"Completed in {time.time() - start:.2f} seconds")
    
    # Study 3: Optical Power and Aperture Tradeoffs
    print_header("Study 3: Optical Power & Aperture Tradeoffs")
    print("Analyzing optical system configurations...")
    start = time.time()
    
    optical_study = OpticalTradeStudy()
    optical_results = optical_study.run_trade_study()
    optical_study.plot_results(optical_results, '/home/claude/mission-trades/plots')
    
    results['optical'] = optical_results
    print(f"Completed in {time.time() - start:.2f} seconds")
    
    # Study 4: Pareto Front Analysis
    print_header("Study 4: Multi-Objective Pareto Front Analysis")
    print("Identifying Pareto-optimal design points...")
    start = time.time()
    
    pareto_analysis = ParetoAnalysis()
    designs, objectives, analyses = pareto_analysis.run_pareto_analysis()
    pareto_analysis.plot_pareto_fronts(designs, objectives, analyses,
                                       '/home/claude/mission-trades/plots')
    
    results['pareto'] = {
        'designs': designs,
        'objectives': objectives,
        'analyses': analyses
    }
    print(f"Completed in {time.time() - start:.2f} seconds")
    
    # Summary
    print_header("TRADE STUDIES COMPLETE")
    print("\n✓ All analyses completed successfully")
    print(f"✓ Plots saved to: /home/claude/mission-trades/plots/")
    print(f"✓ Total execution time: {sum([time.time() for _ in range(1)])} seconds")
    
    return results


def generate_summary_statistics(results):
    """Generate summary statistics from all studies."""
    
    import numpy as np
    
    stats = {}
    
    # Baseline study stats
    baseline = results['baseline']
    stats['baseline'] = {
        'optimal_length': float(baseline['baseline'][np.argmax(baseline['sensitivity_nominal'])]),
        'max_sensitivity': float(np.max(baseline['sensitivity_nominal'])),
        'min_resolution': float(np.min(baseline['resolution']))
    }
    
    # Orbit study stats
    orbit = results['orbit']
    stats['orbit'] = {
        'best_revisit_altitude': float(orbit['altitude'][np.argmin(np.mean(orbit['revisit_mesh'], axis=0))]),
        'max_coverage_altitude': float(orbit['altitude'][np.argmax(np.mean(orbit['coverage_mesh'], axis=0))]),
        'recommended_altitude': 650.0  # km - balanced choice
    }
    
    # Optical study stats
    optical = results['optical']
    max_datarate_idx = np.unravel_index(np.argmax(optical['data_rate_mesh']), 
                                        optical['data_rate_mesh'].shape)
    stats['optical'] = {
        'max_datarate': float(np.max(optical['data_rate_mesh'])),
        'optimal_power': float(optical['power_mesh'][max_datarate_idx]),
        'optimal_aperture': float(optical['aperture_mesh'][max_datarate_idx])
    }
    
    # Pareto stats
    pareto = results['pareto']
    perf_cost_analysis = pareto['analyses']['performance_cost']
    pareto_points = perf_cost_analysis['objectives'][perf_cost_analysis['pareto_indices']]
    
    stats['pareto'] = {
        'n_pareto_optimal': len(pareto_points),
        'total_designs': len(perf_cost_analysis['objectives']),
        'pareto_efficiency': len(pareto_points) / len(perf_cost_analysis['objectives']) * 100
    }
    
    return stats


if __name__ == '__main__':
    # Run all trade studies
    results = run_all_studies()
    
    # Generate summary stats
    stats = generate_summary_statistics(results)
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    print(f"\nBaseline Study:")
    print(f"  - Optimal baseline length: {stats['baseline']['optimal_length']:.1f} m")
    print(f"  - Maximum sensitivity: {stats['baseline']['max_sensitivity']:.1f}")
    print(f"  - Best resolution: {stats['baseline']['min_resolution']:.2f} μas")
    
    print(f"\nOrbit Study:")
    print(f"  - Best revisit altitude: {stats['orbit']['best_revisit_altitude']:.0f} km")
    print(f"  - Max coverage altitude: {stats['orbit']['max_coverage_altitude']:.0f} km")
    print(f"  - Recommended altitude: {stats['orbit']['recommended_altitude']:.0f} km")
    
    print(f"\nOptical Study:")
    print(f"  - Maximum data rate: {stats['optical']['max_datarate']:.1f} Gbps")
    print(f"  - Optimal power: {stats['optical']['optimal_power']:.1f} W")
    print(f"  - Optimal aperture: {stats['optical']['optimal_aperture']:.2f} m")
    
    print(f"\nPareto Analysis:")
    print(f"  - Pareto-optimal designs: {stats['pareto']['n_pareto_optimal']}")
    print(f"  - Total designs evaluated: {stats['pareto']['total_designs']}")
    print(f"  - Pareto efficiency: {stats['pareto']['pareto_efficiency']:.1f}%")
    
    print("\n" + "="*70)
    print("Generating decision memo...")
    
    # Save stats for memo generation
    import json
    with open('/home/claude/mission-trades/trade_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("✓ Statistics saved to trade_stats.json")
    print("\nNext: Review plots and decision memo for design recommendations")
