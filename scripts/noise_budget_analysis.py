"""
Noise Budget Analysis for Inter-Satellite Ranging.

This script generates detailed noise budget tables and plots showing
the contribution of each noise source to ranging measurement uncertainty.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from sensing.model import MeasurementModel, NoiseParameters, OpticalLink


def generate_noise_budget_table(
    baselines: List[float] = None,
    integration_times: List[float] = None
) -> None:
    """
    Generate comprehensive noise budget table.
    
    Args:
        baselines: List of inter-satellite distances (m)
        integration_times: List of integration times (s)
    """
    if baselines is None:
        baselines = [10e3, 50e3, 100e3, 200e3, 500e3]  # 10 km to 500 km
    
    if integration_times is None:
        integration_times = [0.1, 1.0, 10.0]  # 0.1 s to 10 s
    
    print("="*80)
    print("INTER-SATELLITE RANGING NOISE BUDGET")
    print("GeoSense Platform - Session 1")
    print("="*80)
    print()
    
    for tau in integration_times:
        print(f"\n{'─'*80}")
        print(f"Integration Time: {tau} s")
        print(f"{'─'*80}\n")
        
        # Table header
        header = f"{'Baseline':<12} | {'Shot':<10} | {'Freq':<10} | {'Clock':<10} | {'Point':<10} | {'Total':<10}"
        print(header)
        print("─" * len(header))
        
        for baseline in baselines:
            # Create measurement model
            noise_params = NoiseParameters(
                photon_rate=1e9,  # 1 billion photons/s
                quantum_efficiency=0.8,
                frequency_noise_psd=1e-24,
                phase_noise_floor=1e-6,
                pointing_jitter_rms=1e-6,  # 1 μrad
                temperature=300.0,
                bandwidth=1e6
            )
            
            link = OpticalLink(
                wavelength=1064e-9,  # Nd:YAG
                power_transmitted=1.0,
                aperture_diameter=0.3,
                range=baseline
            )
            
            model = MeasurementModel(
                noise_params=noise_params,
                link=link,
                integration_time=tau
            )
            
            # Get noise budget
            budget = model.noise_budget()
            
            # Print row
            print(f"{baseline/1e3:>6.0f} km    | "
                  f"{budget['shot_noise']*1e9:>8.2f} nm | "
                  f"{budget['frequency_noise']*1e9:>8.2f} nm | "
                  f"{budget['clock_noise']*1e9:>8.2f} nm | "
                  f"{budget['pointing_noise']*1e6:>8.2f} μm | "
                  f"{budget['total_rss']*1e6:>8.2f} μm")
    
    print("\n" + "="*80)
    print("Notes:")
    print("  - Shot noise: Quantum (photon counting) noise")
    print("  - Freq: Laser frequency/phase instability")
    print("  - Clock: Clock Allan deviation contribution")
    print("  - Point: Pointing jitter (angular uncertainty)")
    print("  - Total: Root-sum-square of all contributions")
    print("="*80)


def generate_parametric_plots(output_dir: str = ".") -> None:
    """
    Generate plots showing noise vs. key parameters.
    
    Args:
        output_dir: Directory to save plots
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Noise vs. Baseline
    fig, ax = plt.subplots(figsize=(10, 6))
    
    baselines = np.logspace(4, 5.7, 50)  # 10 km to 500 km
    noise_components = {
        'Shot noise': [],
        'Frequency noise': [],
        'Clock noise': [],
        'Pointing jitter': [],
        'Total RSS': []
    }
    
    for baseline in baselines:
        noise_params = NoiseParameters(
            photon_rate=1e9,
            quantum_efficiency=0.8,
            pointing_jitter_rms=1e-6
        )
        link = OpticalLink(range=baseline)
        model = MeasurementModel(noise_params=noise_params, link=link)
        budget = model.noise_budget()
        
        noise_components['Shot noise'].append(budget['shot_noise'] * 1e6)
        noise_components['Frequency noise'].append(budget['frequency_noise'] * 1e6)
        noise_components['Clock noise'].append(budget['clock_noise'] * 1e6)
        noise_components['Pointing jitter'].append(budget['pointing_noise'] * 1e6)
        noise_components['Total RSS'].append(budget['total_rss'] * 1e6)
    
    for label, values in noise_components.items():
        style = '--' if label == 'Total RSS' else '-'
        width = 2 if label == 'Total RSS' else 1
        ax.loglog(baselines/1e3, values, style, linewidth=width, label=label)
    
    ax.set_xlabel('Baseline (km)', fontsize=12)
    ax.set_ylabel('Ranging Noise (μm)', fontsize=12)
    ax.set_title('Ranging Noise vs. Inter-Satellite Baseline', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noise_vs_baseline.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Noise vs. Integration Time
    fig, ax = plt.subplots(figsize=(10, 6))
    
    integration_times = np.logspace(-1, 2, 50)  # 0.1 s to 100 s
    baseline = 100e3  # Fixed 100 km baseline
    
    noise_vs_tau = {
        'Shot noise': [],
        'Frequency noise': [],
        'Clock noise': [],
        'Total RSS': []
    }
    
    for tau in integration_times:
        noise_params = NoiseParameters(photon_rate=1e9, quantum_efficiency=0.8)
        link = OpticalLink(range=baseline)
        model = MeasurementModel(
            noise_params=noise_params,
            link=link,
            integration_time=tau
        )
        budget = model.noise_budget()
        
        noise_vs_tau['Shot noise'].append(budget['shot_noise'] * 1e9)
        noise_vs_tau['Frequency noise'].append(budget['frequency_noise'] * 1e9)
        noise_vs_tau['Clock noise'].append(budget['clock_noise'] * 1e9)
        noise_vs_tau['Total RSS'].append(budget['total_rss'] * 1e9)
    
    for label, values in noise_vs_tau.items():
        style = '--' if label == 'Total RSS' else '-'
        width = 2 if label == 'Total RSS' else 1
        ax.loglog(integration_times, values, style, linewidth=width, label=label)
    
    # Add τ^(-1/2) reference line
    ref_line = noise_vs_tau['Shot noise'][10] * np.sqrt(integration_times[10] / integration_times)
    ax.loglog(integration_times, ref_line, ':', color='gray', alpha=0.5, label='τ^(-1/2) scaling')
    
    ax.set_xlabel('Integration Time (s)', fontsize=12)
    ax.set_ylabel('Ranging Noise (nm)', fontsize=12)
    ax.set_title('Ranging Noise vs. Integration Time (100 km baseline)', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noise_vs_integration_time.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Noise Breakdown Pie Chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    configurations = [
        ('Short Baseline (10 km)', 10e3),
        ('Medium Baseline (100 km)', 100e3),
        ('Long Baseline (500 km)', 500e3)
    ]
    
    for ax, (title, baseline) in zip(axes, configurations):
        noise_params = NoiseParameters(
            photon_rate=1e9,
            quantum_efficiency=0.8,
            pointing_jitter_rms=1e-6
        )
        link = OpticalLink(range=baseline)
        model = MeasurementModel(noise_params=noise_params, link=link)
        budget = model.noise_budget()
        
        # Calculate contributions (squared for RSS)
        contributions = {
            'Shot': budget['shot_noise']**2,
            'Frequency': budget['frequency_noise']**2,
            'Clock': budget['clock_noise']**2,
            'Pointing': budget['pointing_noise']**2
        }
        
        # Filter out negligible contributions
        total_power = sum(contributions.values())
        filtered = {k: v for k, v in contributions.items() if v/total_power > 0.01}
        
        labels = list(filtered.keys())
        sizes = list(filtered.values())
        
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        ax.pie(sizes, labels=labels, colors=colors[:len(labels)], autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.suptitle('Noise Budget Breakdown by Baseline', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/noise_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plots saved to {output_dir}/")


def generate_allan_deviation_plot(output_dir: str = ".") -> None:
    """Generate example Allan deviation plot."""
    from sensing.model import allan_deviation, NoiseGenerator
    import jax
    
    # Generate example time series with realistic clock noise
    key = jax.random.PRNGKey(42)
    n_samples = 100000
    dt = 0.1  # 100 ms sampling
    
    # White frequency noise component
    key1, key2 = jax.random.split(key)
    white_noise = NoiseGenerator.white_noise(key1, n_samples, std=2e-13)
    
    # Random walk component
    walk = NoiseGenerator.random_walk(key2, n_samples, diffusion_coeff=1e-28, dt=dt)
    
    # Combined clock signal (fractional frequency)
    clock_signal = np.array(white_noise) + np.array(walk)
    
    # Compute Allan deviation
    taus, adevs = allan_deviation(clock_signal, dt, max_tau=n_samples//10)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.loglog(taus, adevs, 'o-', linewidth=2, markersize=4, label='Computed σ_y(τ)')
    
    # Fit and plot components
    log_tau = np.log10(taus[5:30])
    log_adev = np.log10(adevs[5:30])
    
    # White frequency noise line (slope -1/2)
    white_ref = adevs[10] * (taus / taus[10])**(-0.5)
    ax.loglog(taus, white_ref, '--', alpha=0.5, label='τ^(-1/2) (white freq)')
    
    # Random walk line (slope +1/2)
    walk_ref = adevs[-10] * (taus / taus[-10])**(0.5)
    ax.loglog(taus, walk_ref, '--', alpha=0.5, label='τ^(+1/2) (random walk)')
    
    ax.set_xlabel('Averaging Time τ (s)', fontsize=12)
    ax.set_ylabel('Allan Deviation σ_y(τ)', fontsize=12)
    ax.set_title('Clock Stability - Allan Deviation', fontsize=14, fontweight='bold')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/allan_deviation.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Allan deviation plot saved to {output_dir}/")


if __name__ == "__main__":
    # Generate tables
    print("\n" + "="*80)
    print("GENERATING NOISE BUDGET ANALYSIS")
    print("="*80 + "\n")
    
    generate_noise_budget_table()
    
    # Generate plots
    print("\n\nGenerating plots...")
    generate_parametric_plots(output_dir="docs/figures")
    generate_allan_deviation_plot(output_dir="docs/figures")
    
    print("\n" + "="*80)
    print("✅ NOISE BUDGET ANALYSIS COMPLETE")
    print("="*80)
    print("\nOutputs:")
    print("  - Noise budget tables (printed above)")
    print("  - docs/figures/noise_vs_baseline.png")
    print("  - docs/figures/noise_vs_integration_time.png")
    print("  - docs/figures/noise_breakdown.png")
    print("  - docs/figures/allan_deviation.png")
    print("="*80)
