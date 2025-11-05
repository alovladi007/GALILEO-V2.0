"""
Baseline Length vs Noise vs Sensitivity Trade Study

Analyzes the relationship between interferometer baseline length,
measurement noise, and sensitivity for optical/IR space missions.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class BaselineTradeStudy:
    """Trade study for baseline length vs noise vs sensitivity."""
    
    def __init__(self):
        self.baseline_range = np.linspace(10, 1000, 50)  # meters
        self.wavelength = 10e-6  # 10 microns (mid-IR)
        self.integration_time = 3600  # seconds
        
    def angular_resolution(self, baseline):
        """Calculate angular resolution in microarcseconds."""
        return (self.wavelength / baseline) * 206265e6  # convert to microarcsec
    
    def sensitivity(self, baseline, noise_level):
        """
        Calculate sensitivity (SNR) as a function of baseline and noise.
        
        Sensitivity improves with baseline (more spatial frequency coverage)
        but degrades with higher noise levels.
        """
        # Baseline contribution (longer baseline = better resolution = better sensitivity)
        baseline_factor = np.sqrt(baseline / 10)  # normalized to 10m baseline
        
        # Noise impact (higher noise = lower sensitivity)
        noise_factor = 1 / (1 + noise_level)
        
        # Integration time contribution
        time_factor = np.sqrt(self.integration_time / 3600)
        
        return baseline_factor * noise_factor * time_factor * 100  # arbitrary units
    
    def noise_sources(self, baseline):
        """
        Model various noise sources as a function of baseline.
        
        Longer baselines typically have more noise due to:
        - Increased pathlength errors
        - More challenging alignment
        - Thermal and vibration effects
        """
        # Base thermal noise
        thermal_noise = 0.05
        
        # Pathlength noise (increases with baseline)
        pathlength_noise = 0.02 * np.log10(baseline / 10 + 1)
        
        # Vibration noise (scales with baseline)
        vibration_noise = 0.01 * (baseline / 100)
        
        # Detector noise (constant)
        detector_noise = 0.03
        
        total_noise = np.sqrt(thermal_noise**2 + pathlength_noise**2 + 
                             vibration_noise**2 + detector_noise**2)
        
        return total_noise
    
    def run_trade_study(self):
        """Execute the full trade study and generate results."""
        results = {}
        
        # Calculate angular resolution for all baselines
        resolution = self.angular_resolution(self.baseline_range)
        
        # Calculate noise levels
        noise_levels = np.array([self.noise_sources(b) for b in self.baseline_range])
        
        # Calculate sensitivity for various noise scenarios
        low_noise = noise_levels * 0.5  # optimistic
        nominal_noise = noise_levels
        high_noise = noise_levels * 2.0  # pessimistic
        
        sensitivity_low = np.array([self.sensitivity(b, n) 
                                   for b, n in zip(self.baseline_range, low_noise)])
        sensitivity_nom = np.array([self.sensitivity(b, n) 
                                   for b, n in zip(self.baseline_range, nominal_noise)])
        sensitivity_high = np.array([self.sensitivity(b, n) 
                                    for b, n in zip(self.baseline_range, high_noise)])
        
        results = {
            'baseline': self.baseline_range,
            'resolution': resolution,
            'noise_low': low_noise,
            'noise_nominal': nominal_noise,
            'noise_high': high_noise,
            'sensitivity_low': sensitivity_low,
            'sensitivity_nominal': sensitivity_nom,
            'sensitivity_high': sensitivity_high
        }
        
        return results
    
    def plot_results(self, results, output_dir):
        """Generate comprehensive plots of trade study results."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        
        # Plot 1: Resolution vs Baseline
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(results['baseline'], results['resolution'], 'b-', linewidth=2)
        ax1.set_xlabel('Baseline Length (m)', fontsize=11)
        ax1.set_ylabel('Angular Resolution (μas)', fontsize=11)
        ax1.set_title('Resolution vs Baseline Length', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Noise vs Baseline
        ax2 = plt.subplot(2, 3, 2)
        ax2.plot(results['baseline'], results['noise_low'], 'g--', label='Low Noise', linewidth=2)
        ax2.plot(results['baseline'], results['noise_nominal'], 'b-', label='Nominal', linewidth=2)
        ax2.plot(results['baseline'], results['noise_high'], 'r--', label='High Noise', linewidth=2)
        ax2.set_xlabel('Baseline Length (m)', fontsize=11)
        ax2.set_ylabel('Noise Level (normalized)', fontsize=11)
        ax2.set_title('Noise vs Baseline Length', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Sensitivity vs Baseline
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(results['baseline'], results['sensitivity_low'], 'g--', 
                label='Low Noise', linewidth=2)
        ax3.plot(results['baseline'], results['sensitivity_nominal'], 'b-', 
                label='Nominal', linewidth=2)
        ax3.plot(results['baseline'], results['sensitivity_high'], 'r--', 
                label='High Noise', linewidth=2)
        ax3.set_xlabel('Baseline Length (m)', fontsize=11)
        ax3.set_ylabel('Sensitivity (SNR, arb. units)', fontsize=11)
        ax3.set_title('Sensitivity vs Baseline Length', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: 3D Surface Plot - Baseline vs Noise vs Sensitivity
        ax4 = fig.add_subplot(2, 3, 4, projection='3d')
        baseline_mesh, noise_mesh = np.meshgrid(results['baseline'][:30], 
                                                np.linspace(0.05, 0.3, 30))
        sensitivity_mesh = np.array([[self.sensitivity(b, n) 
                                     for b in baseline_mesh[0]] 
                                    for n in noise_mesh[:, 0]])
        
        surf = ax4.plot_surface(baseline_mesh, noise_mesh, sensitivity_mesh,
                               cmap=cm.viridis, alpha=0.8)
        ax4.set_xlabel('Baseline (m)', fontsize=10)
        ax4.set_ylabel('Noise Level', fontsize=10)
        ax4.set_zlabel('Sensitivity', fontsize=10)
        ax4.set_title('3D Trade Space', fontsize=12, fontweight='bold')
        plt.colorbar(surf, ax=ax4, shrink=0.5)
        
        # Plot 5: Sensitivity-Resolution Tradeoff
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(results['resolution'], results['sensitivity_nominal'], 'b-', 
                linewidth=2, marker='o', markersize=3)
        ax5.set_xlabel('Angular Resolution (μas)', fontsize=11)
        ax5.set_ylabel('Sensitivity (SNR)', fontsize=11)
        ax5.set_title('Sensitivity-Resolution Tradeoff', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.invert_xaxis()  # Better resolution is smaller
        ax5.set_xscale('log')
        
        # Plot 6: Performance Metrics
        ax6 = plt.subplot(2, 3, 6)
        
        # Find optimal baselines for different criteria
        idx_max_sens = np.argmax(results['sensitivity_nominal'])
        idx_balance = np.argmin(np.abs(results['baseline'] - 100))  # 100m reference
        
        baselines_of_interest = [
            results['baseline'][idx_balance],
            results['baseline'][idx_max_sens]
        ]
        sensitivities = [
            results['sensitivity_nominal'][idx_balance],
            results['sensitivity_nominal'][idx_max_sens]
        ]
        resolutions = [
            results['resolution'][idx_balance],
            results['resolution'][idx_max_sens]
        ]
        
        x_pos = np.arange(len(['Balanced (100m)', 'Max Sensitivity']))
        ax6.bar(x_pos, sensitivities, alpha=0.7, color=['blue', 'green'])
        ax6.set_ylabel('Sensitivity (SNR)', fontsize=11)
        ax6.set_title('Design Point Comparison', fontsize=12, fontweight='bold')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(['Balanced\n(100m)', f'Max Sens.\n({baselines_of_interest[1]:.0f}m)'])
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add resolution as secondary axis
        ax6_twin = ax6.twinx()
        ax6_twin.plot(x_pos, resolutions, 'ro-', linewidth=2, markersize=8, label='Resolution')
        ax6_twin.set_ylabel('Resolution (μas)', fontsize=11, color='r')
        ax6_twin.tick_params(axis='y', labelcolor='r')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/baseline_trade_study.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved baseline trade study plots to {output_dir}/baseline_trade_study.png")
        
        return fig


def main():
    """Run the baseline trade study."""
    study = BaselineTradeStudy()
    results = study.run_trade_study()
    study.plot_results(results, '/home/claude/mission-trades/plots')
    return results


if __name__ == '__main__':
    main()
