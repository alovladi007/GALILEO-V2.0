"""
Generate Example Data and Plots for Documentation

Creates sample synthetic data and generates visualization plots.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
sys.path.append('/home/claude')

from sim.synthetic import (
    SimulationConfig, SatelliteConfig, SubsurfaceModel,
    ForwardModel, SyntheticDataGenerator
)

# Set style for nice plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_example_data():
    """Generate example synthetic dataset"""
    print("Generating example synthetic data...")
    
    # Configure simulation
    sim_config = SimulationConfig(
        grid_size=(80, 80, 40),
        grid_spacing=10.0,
        time_steps=50,
        time_interval=1.0,
        seed=42,
        noise_level=0.1,
        atmospheric_noise=0.05,
        thermal_noise=0.03
    )
    
    sat_config = SatelliteConfig(
        num_satellites=2,
        orbital_height=500e3,
        baseline_nominal=200.0,
        baseline_variation=10.0
    )
    
    # Generate data
    generator = SyntheticDataGenerator(sim_config, sat_config)
    results = generator.generate("./data")
    
    print(f"✓ Generated data with ID: {results['simulation_id']}")
    return results, sim_config, sat_config


def create_visualization_plots(results, sim_config):
    """Create visualization plots for documentation"""
    print("Creating visualization plots...")
    
    # Create output directory
    plot_dir = Path("./docs/images")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    phase_data = np.load(results['phase_path'])
    telemetry = pd.read_parquet(results['telemetry_path'])
    
    with open(results['card_path'], 'r') as f:
        dataset_card = json.load(f)
    
    # 1. Density Field Cross-Section
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    density = phase_data['density_field']
    
    # XY slice at mid-depth
    im1 = axes[0].imshow(density[:, :, density.shape[2]//2].T, 
                         cmap='RdBu_r', aspect='auto')
    axes[0].set_title('Density Field (XY plane, mid-depth)')
    axes[0].set_xlabel('X [grid points]')
    axes[0].set_ylabel('Y [grid points]')
    plt.colorbar(im1, ax=axes[0], label='Density [kg/m³]')
    
    # XZ slice at center
    im2 = axes[1].imshow(density[:, density.shape[1]//2, :].T,
                         cmap='RdBu_r', aspect='auto', origin='lower')
    axes[1].set_title('Density Field (XZ plane, center)')
    axes[1].set_xlabel('X [grid points]')
    axes[1].set_ylabel('Depth [grid points]')
    plt.colorbar(im2, ax=axes[1], label='Density [kg/m³]')
    
    # YZ slice at center
    im3 = axes[2].imshow(density[density.shape[0]//2, :, :].T,
                         cmap='RdBu_r', aspect='auto', origin='lower')
    axes[2].set_title('Density Field (YZ plane, center)')
    axes[2].set_xlabel('Y [grid points]')
    axes[2].set_ylabel('Depth [grid points]')
    plt.colorbar(im3, ax=axes[2], label='Density [kg/m³]')
    
    plt.suptitle('Subsurface Density Model with Procedural Anomalies')
    plt.tight_layout()
    plt.savefig(plot_dir / 'density_field.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Gravity Field
    fig, ax = plt.subplots(figsize=(8, 6))
    
    g_field = phase_data['gravity_field']
    im = ax.imshow(g_field.T * 1e9, cmap='seismic', aspect='auto')  # Convert to nanoGal
    ax.set_title('Gravity Field at Satellite Altitude')
    ax.set_xlabel('X [grid points]')
    ax.set_ylabel('Y [grid points]')
    plt.colorbar(im, ax=ax, label='Gravity anomaly [nGal]')
    
    # Add contours
    contours = ax.contour(g_field.T * 1e9, levels=10, colors='black', 
                         alpha=0.3, linewidths=0.5)
    ax.clabel(contours, inline=True, fontsize=8, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'gravity_field.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Phase Time Series
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    phases = phase_data['phases']
    noisy_phases = phase_data['noisy_phases']
    
    # Clean phases at t=0
    im1 = axes[0, 0].imshow(phases[0].T, cmap='twilight', 
                            vmin=-np.pi, vmax=np.pi, aspect='auto')
    axes[0, 0].set_title('Clean Phase (t=0)')
    axes[0, 0].set_xlabel('X [grid points]')
    axes[0, 0].set_ylabel('Y [grid points]')
    plt.colorbar(im1, ax=axes[0, 0], label='Phase [rad]')
    
    # Noisy phases at t=0
    im2 = axes[0, 1].imshow(noisy_phases[0].T, cmap='twilight',
                            vmin=-np.pi, vmax=np.pi, aspect='auto')
    axes[0, 1].set_title('Noisy Phase (t=0)')
    axes[0, 1].set_xlabel('X [grid points]')
    axes[0, 1].set_ylabel('Y [grid points]')
    plt.colorbar(im2, ax=axes[0, 1], label='Phase [rad]')
    
    # Phase difference (t=25 - t=0)
    if phases.shape[0] > 25:
        phase_diff = phases[25] - phases[0]
        im3 = axes[1, 0].imshow(phase_diff.T, cmap='RdBu_r', aspect='auto')
        axes[1, 0].set_title('Phase Change (t=25 - t=0)')
        axes[1, 0].set_xlabel('X [grid points]')
        axes[1, 0].set_ylabel('Y [grid points]')
        plt.colorbar(im3, ax=axes[1, 0], label='ΔPhase [rad]')
    
    # Time series at center point
    center_x, center_y = phases.shape[1]//2, phases.shape[2]//2
    axes[1, 1].plot(phases[:, center_x, center_y], label='Clean', linewidth=2)
    axes[1, 1].plot(noisy_phases[:, center_x, center_y], label='Noisy', 
                    alpha=0.7, linewidth=1)
    axes[1, 1].set_title(f'Phase Time Series at Center ({center_x}, {center_y})')
    axes[1, 1].set_xlabel('Time [days]')
    axes[1, 1].set_ylabel('Phase [rad]')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Interferometric Phase Evolution')
    plt.tight_layout()
    plt.savefig(plot_dir / 'phase_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 4. Baseline Dynamics
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    baselines = phase_data['baselines']
    
    # Baseline map at mid-time
    mid_t = baselines.shape[0] // 2
    im1 = axes[0].imshow(baselines[mid_t].T, cmap='viridis', aspect='auto')
    axes[0].set_title(f'Baseline Configuration (t={mid_t})')
    axes[0].set_xlabel('X [grid points]')
    axes[0].set_ylabel('Y [grid points]')
    plt.colorbar(im1, ax=axes[0], label='Baseline [m]')
    
    # Baseline evolution histogram
    axes[1].hist(baselines.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1].axvline(200.0, color='red', linestyle='--', label='Nominal (200m)')
    axes[1].set_title('Baseline Distribution')
    axes[1].set_xlabel('Baseline [m]')
    axes[1].set_ylabel('Frequency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Satellite Baseline Dynamics')
    plt.tight_layout()
    plt.savefig(plot_dir / 'baseline_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 5. Telemetry Statistics
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Phase distribution
    axes[0, 0].hist(telemetry['phase'], bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Phase Distribution')
    axes[0, 0].set_xlabel('Phase [rad]')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Coherence distribution
    axes[0, 1].hist(telemetry['coherence'], bins=30, alpha=0.7, 
                    edgecolor='black', color='green')
    axes[0, 1].set_title('Coherence Distribution')
    axes[0, 1].set_xlabel('Coherence')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(telemetry['coherence'].mean(), color='red', 
                       linestyle='--', label=f'Mean: {telemetry["coherence"].mean():.2f}')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # SNR distribution
    axes[0, 2].hist(telemetry['snr'], bins=30, alpha=0.7, 
                    edgecolor='black', color='orange')
    axes[0, 2].set_title('SNR Distribution')
    axes[0, 2].set_xlabel('SNR [dB]')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Spatial sampling pattern
    sample_t = telemetry[telemetry['timestamp'] == telemetry['timestamp'].unique()[0]]
    axes[1, 0].scatter(sample_t['pixel_x'], sample_t['pixel_y'], 
                      c=sample_t['phase'], cmap='twilight', s=20, alpha=0.6)
    axes[1, 0].set_title('Spatial Sampling Pattern (t=0)')
    axes[1, 0].set_xlabel('X [pixels]')
    axes[1, 0].set_ylabel('Y [pixels]')
    axes[1, 0].set_aspect('equal')
    
    # Quality flag distribution
    quality_counts = telemetry['quality_flag'].value_counts().sort_index()
    axes[1, 1].bar(quality_counts.index, quality_counts.values, 
                   color=['green', 'yellow', 'red'])
    axes[1, 1].set_title('Quality Flag Distribution')
    axes[1, 1].set_xlabel('Quality Flag')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks([0, 1, 2])
    axes[1, 1].set_xticklabels(['Good', 'Warning', 'Bad'])
    axes[1, 1].grid(True, alpha=0.3)
    
    # Time series sample count
    time_counts = telemetry.groupby('timestamp').size()
    axes[1, 2].plot(range(len(time_counts)), time_counts.values, 
                    marker='o', linewidth=2)
    axes[1, 2].set_title('Samples per Time Step')
    axes[1, 2].set_xlabel('Time Step')
    axes[1, 2].set_ylabel('Number of Samples')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Telemetry Data Statistics')
    plt.tight_layout()
    plt.savefig(plot_dir / 'telemetry_statistics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 6. Anomaly Summary
    fig, ax = plt.subplots(figsize=(8, 6))
    
    anomaly_stats = dataset_card['anomalies']['types']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = ax.bar(anomaly_stats.keys(), anomaly_stats.values(), color=colors, 
                  edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=12)
    
    ax.set_title('Procedural Anomaly Distribution')
    ax.set_xlabel('Anomaly Type')
    ax.set_ylabel('Count')
    ax.set_ylim(0, max(anomaly_stats.values()) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add total count
    total = sum(anomaly_stats.values())
    ax.text(0.98, 0.98, f'Total Anomalies: {total}',
            transform=ax.transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'anomaly_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Created {6} visualization plots in ./docs/images/")
    return plot_dir


def create_statistics_summary(results):
    """Create summary statistics table"""
    print("Generating statistics summary...")
    
    # Load data
    telemetry = pd.read_parquet(results['telemetry_path'])
    phase_data = np.load(results['phase_path'])
    
    with open(results['card_path'], 'r') as f:
        dataset_card = json.load(f)
    
    # Compute statistics
    stats = {
        'Dataset': {
            'Simulation ID': results['simulation_id'],
            'Grid Size': str(dataset_card['configuration']['simulation']['grid_size']),
            'Time Steps': dataset_card['configuration']['simulation']['time_steps'],
            'Total Anomalies': dataset_card['anomalies']['count']
        },
        'Telemetry': {
            'Total Records': len(telemetry),
            'Phase Range (rad)': f"[{telemetry['phase'].min():.3f}, {telemetry['phase'].max():.3f}]",
            'Mean Coherence': f"{telemetry['coherence'].mean():.3f}",
            'Mean SNR (dB)': f"{telemetry['snr'].mean():.1f}",
            'Good Quality (%)': f"{100*(telemetry['quality_flag']==0).mean():.1f}"
        },
        'Physical Parameters': {
            'Grid Spacing (m)': dataset_card['configuration']['simulation']['grid_spacing'],
            'Orbital Height (km)': dataset_card['configuration']['satellite']['orbital_height'] / 1000,
            'Baseline (m)': f"{dataset_card['configuration']['satellite']['baseline_nominal']} ± "
                           f"{dataset_card['configuration']['satellite']['baseline_variation']}",
            'Noise Level (rad)': dataset_card['configuration']['simulation']['noise_level']
        }
    }
    
    # Save as formatted text
    stats_path = Path('./docs/statistics_summary.txt')
    stats_path.parent.mkdir(exist_ok=True)
    
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("SYNTHETIC DATA STATISTICS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        
        for section, values in stats.items():
            f.write(f"{section}:\n")
            f.write("-" * 40 + "\n")
            for key, value in values.items():
                f.write(f"  {key:<25} {value}\n")
            f.write("\n")
    
    print(f"✓ Saved statistics summary to {stats_path}")
    return stats


def main():
    """Main execution"""
    print("\n" + "="*60)
    print("SYNTHETIC DATA GENERATOR - EXAMPLE & DOCUMENTATION")
    print("="*60 + "\n")
    
    # Generate example data
    results, sim_config, sat_config = generate_example_data()
    
    # Create visualizations
    plot_dir = create_visualization_plots(results, sim_config)
    
    # Generate statistics
    stats = create_statistics_summary(results)
    
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    print(f"✓ Data files in: ./data/")
    print(f"✓ Plot images in: {plot_dir}")
    print(f"✓ Statistics in: ./docs/statistics_summary.txt")
    print("\nFiles generated:")
    print(f"  - Telemetry: {Path(results['telemetry_path']).name}")
    print(f"  - Phases: {Path(results['phase_path']).name}")
    print(f"  - STAC Collection: {Path(results['collection_path']).name}")
    print(f"  - STAC Item: {Path(results['item_path']).name}")
    print(f"  - Dataset Card: {Path(results['card_path']).name}")
    

if __name__ == "__main__":
    main()
