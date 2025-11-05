"""
Optical Power and Aperture Tradeoffs Trade Study

Analyzes the relationship between laser power, telescope aperture,
and system performance for optical communication and sensing.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class OpticalTradeStudy:
    """Trade study for optical power and aperture tradeoffs."""
    
    def __init__(self):
        self.power_range = np.linspace(1, 100, 50)  # Watts
        self.aperture_range = np.linspace(0.1, 2.0, 50)  # meters
        self.wavelength = 1550e-9  # 1550 nm (near-IR communication)
        self.distance = 40000e3  # 40,000 km (GEO distance)
        self.atm_transmittance = 0.7  # Atmospheric transmission factor
        
    def beam_divergence(self, aperture):
        """
        Calculate beam divergence angle (radians).
        
        Smaller aperture = larger divergence = more spreading
        """
        return 2.44 * self.wavelength / aperture
    
    def link_budget(self, power, aperture_tx, aperture_rx=1.0):
        """
        Calculate optical link budget (received power).
        
        Args:
            power: Transmit power (W)
            aperture_tx: Transmit aperture (m)
            aperture_rx: Receive aperture (m)
        """
        # Beam divergence
        theta = self.beam_divergence(aperture_tx)
        
        # Beam spot size at receiver
        spot_radius = self.distance * np.tan(theta / 2)
        spot_area = np.pi * spot_radius**2
        
        # Geometric spreading loss
        spreading_loss = (aperture_tx**2) / (4 * spot_area)
        
        # Atmospheric loss
        atm_loss = self.atm_transmittance
        
        # Receiver collection area
        rx_area = np.pi * (aperture_rx / 2)**2
        collection_efficiency = rx_area / spot_area if spot_area > rx_area else 1.0
        
        # Received power
        P_rx = power * spreading_loss * atm_loss * collection_efficiency
        
        return P_rx
    
    def data_rate(self, received_power, noise_power=1e-12):
        """
        Estimate achievable data rate using Shannon capacity.
        
        Args:
            received_power: Received optical power (W)
            noise_power: Noise power (W)
        """
        # SNR
        snr = received_power / noise_power
        
        # Shannon capacity (simplified)
        # Bandwidth assumed to be 10 GHz for optical communication
        bandwidth = 10e9  # Hz
        
        if snr > 0:
            capacity = bandwidth * np.log2(1 + snr)
            return capacity / 1e9  # Return in Gbps
        else:
            return 0
    
    def pointing_accuracy_requirement(self, aperture):
        """
        Calculate required pointing accuracy based on beam divergence.
        
        Narrower beams require tighter pointing control.
        """
        theta = self.beam_divergence(aperture)
        # Pointing accuracy should be ~10% of beam divergence
        pointing_req = theta * 0.1
        return np.degrees(pointing_req) * 3600  # Convert to arcseconds
    
    def system_mass(self, aperture, power):
        """
        Estimate total system mass based on aperture and power.
        
        Larger apertures and higher power require more mass.
        """
        # Aperture mass (telescope + structure)
        # Scales approximately as aperture^2.5
        aperture_mass = 50 * (aperture / 0.5)**2.5  # kg
        
        # Power system mass (solar panels + batteries)
        # Assume 0.5 kg/W for space-qualified power systems
        power_mass = power * 0.5  # kg
        
        # Laser/transmitter mass
        # Scales with power
        laser_mass = 10 + power * 0.3  # kg
        
        # Total system mass
        total_mass = aperture_mass + power_mass + laser_mass + 20  # +20 kg for structure
        
        return total_mass
    
    def thermal_management(self, power):
        """
        Estimate thermal management requirements.
        
        Higher power = more waste heat to reject.
        """
        # Assume 30% efficiency, rest is waste heat
        waste_heat = power * 0.7  # Watts
        
        # Radiator area required (W/m^2 radiator effectiveness)
        radiator_effectiveness = 150  # W/m^2 at typical space temps
        radiator_area = waste_heat / radiator_effectiveness
        
        return radiator_area
    
    def system_cost(self, aperture, power):
        """
        Estimate relative system cost.
        
        Costs scale with aperture size and power requirements.
        """
        # Base cost
        base_cost = 5  # M$
        
        # Aperture cost (scales exponentially)
        aperture_cost = 10 * (aperture / 0.5)**2  # M$
        
        # Power system cost
        power_cost = power * 0.05  # M$ (solar panels, batteries)
        
        # Laser cost (high power lasers are expensive)
        laser_cost = 5 + power * 0.1  # M$
        
        total_cost = base_cost + aperture_cost + power_cost + laser_cost
        
        return total_cost
    
    def run_trade_study(self):
        """Execute the full optical trade study."""
        results = {}
        
        # Create 2D meshgrids
        power_mesh, aperture_mesh = np.meshgrid(self.power_range, self.aperture_range)
        
        # Calculate performance metrics
        received_power_mesh = np.zeros_like(power_mesh)
        data_rate_mesh = np.zeros_like(power_mesh)
        mass_mesh = np.zeros_like(power_mesh)
        cost_mesh = np.zeros_like(power_mesh)
        
        for i in range(power_mesh.shape[0]):
            for j in range(power_mesh.shape[1]):
                pwr = power_mesh[i, j]
                apt = aperture_mesh[i, j]
                
                rx_pwr = self.link_budget(pwr, apt)
                received_power_mesh[i, j] = rx_pwr
                data_rate_mesh[i, j] = self.data_rate(rx_pwr)
                mass_mesh[i, j] = self.system_mass(apt, pwr)
                cost_mesh[i, j] = self.system_cost(apt, pwr)
        
        # 1D analyses
        pointing_reqs = [self.pointing_accuracy_requirement(apt) 
                        for apt in self.aperture_range]
        thermal_loads = [self.thermal_management(pwr) for pwr in self.power_range]
        
        results = {
            'power': self.power_range,
            'aperture': self.aperture_range,
            'power_mesh': power_mesh,
            'aperture_mesh': aperture_mesh,
            'received_power_mesh': received_power_mesh,
            'data_rate_mesh': data_rate_mesh,
            'mass_mesh': mass_mesh,
            'cost_mesh': cost_mesh,
            'pointing_requirements': np.array(pointing_reqs),
            'thermal_loads': np.array(thermal_loads)
        }
        
        return results
    
    def plot_results(self, results, output_dir):
        """Generate comprehensive optical trade study plots."""
        
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Received Power Contour
        ax1 = plt.subplot(3, 3, 1)
        contour1 = ax1.contourf(results['power_mesh'], results['aperture_mesh'],
                                np.log10(results['received_power_mesh'] * 1e12),  # in pW
                                levels=20, cmap='plasma')
        ax1.set_xlabel('Transmit Power (W)', fontsize=11)
        ax1.set_ylabel('Aperture Diameter (m)', fontsize=11)
        ax1.set_title('Received Power (log scale)', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(contour1, ax=ax1)
        cbar1.set_label('log₁₀(Received Power, pW)', fontsize=10)
        
        # Plot 2: Data Rate Contour
        ax2 = plt.subplot(3, 3, 2)
        contour2 = ax2.contourf(results['power_mesh'], results['aperture_mesh'],
                                results['data_rate_mesh'], levels=20, cmap='viridis')
        ax2.set_xlabel('Transmit Power (W)', fontsize=11)
        ax2.set_ylabel('Aperture Diameter (m)', fontsize=11)
        ax2.set_title('Data Rate Capability', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(contour2, ax=ax2)
        cbar2.set_label('Data Rate (Gbps)', fontsize=10)
        
        # Add contour lines
        contour_lines = ax2.contour(results['power_mesh'], results['aperture_mesh'],
                                    results['data_rate_mesh'], levels=[1, 10, 50, 100],
                                    colors='white', linewidths=1.5)
        ax2.clabel(contour_lines, inline=True, fontsize=9, fmt='%d Gbps')
        
        # Plot 3: System Mass Contour
        ax3 = plt.subplot(3, 3, 3)
        contour3 = ax3.contourf(results['power_mesh'], results['aperture_mesh'],
                                results['mass_mesh'], levels=20, cmap='YlOrRd')
        ax3.set_xlabel('Transmit Power (W)', fontsize=11)
        ax3.set_ylabel('Aperture Diameter (m)', fontsize=11)
        ax3.set_title('System Mass', fontsize=12, fontweight='bold')
        cbar3 = plt.colorbar(contour3, ax=ax3)
        cbar3.set_label('Mass (kg)', fontsize=10)
        
        # Plot 4: Cost Contour
        ax4 = plt.subplot(3, 3, 4)
        contour4 = ax4.contourf(results['power_mesh'], results['aperture_mesh'],
                                results['cost_mesh'], levels=20, cmap='RdYlGn_r')
        ax4.set_xlabel('Transmit Power (W)', fontsize=11)
        ax4.set_ylabel('Aperture Diameter (m)', fontsize=11)
        ax4.set_title('System Cost', fontsize=12, fontweight='bold')
        cbar4 = plt.colorbar(contour4, ax=ax4)
        cbar4.set_label('Cost (M$)', fontsize=10)
        
        # Plot 5: 3D Surface - Data Rate
        ax5 = fig.add_subplot(3, 3, 5, projection='3d')
        surf1 = ax5.plot_surface(results['power_mesh'], results['aperture_mesh'],
                                results['data_rate_mesh'], cmap='coolwarm', alpha=0.8)
        ax5.set_xlabel('Power (W)', fontsize=10)
        ax5.set_ylabel('Aperture (m)', fontsize=10)
        ax5.set_zlabel('Data Rate (Gbps)', fontsize=10)
        ax5.set_title('3D Performance Surface', fontsize=12, fontweight='bold')
        plt.colorbar(surf1, ax=ax5, shrink=0.5)
        
        # Plot 6: Pointing Requirements vs Aperture
        ax6 = plt.subplot(3, 3, 6)
        ax6.semilogy(results['aperture'], results['pointing_requirements'], 'b-', linewidth=2)
        ax6.set_xlabel('Aperture Diameter (m)', fontsize=11)
        ax6.set_ylabel('Pointing Accuracy (arcsec)', fontsize=11)
        ax6.set_title('Pointing Requirements', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=10, color='red', linestyle='--', label='10" threshold')
        ax6.axhline(y=1, color='orange', linestyle='--', label='1" threshold')
        ax6.legend()
        
        # Plot 7: Thermal Management vs Power
        ax7 = plt.subplot(3, 3, 7)
        ax7.plot(results['power'], results['thermal_loads'], 'r-', linewidth=2)
        ax7.set_xlabel('Transmit Power (W)', fontsize=11)
        ax7.set_ylabel('Radiator Area (m²)', fontsize=11)
        ax7.set_title('Thermal Management Requirements', fontsize=12, fontweight='bold')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Performance vs Cost Trade
        ax8 = plt.subplot(3, 3, 8)
        
        # Sample different configurations
        configs = [
            (10, 0.3, 'Small/Low'),
            (25, 0.5, 'Medium'),
            (50, 1.0, 'Large'),
            (100, 1.5, 'XLarge')
        ]
        
        config_names = []
        config_costs = []
        config_datarates = []
        config_masses = []
        
        for pwr, apt, name in configs:
            config_names.append(name)
            rx_pwr = self.link_budget(pwr, apt)
            config_datarates.append(self.data_rate(rx_pwr))
            config_costs.append(self.system_cost(apt, pwr))
            config_masses.append(self.system_mass(apt, pwr))
        
        # Normalize for plotting
        norm_datarates = np.array(config_datarates) / max(config_datarates)
        
        x_pos = np.arange(len(config_names))
        bars = ax8.bar(x_pos, norm_datarates, alpha=0.7, 
                      color=['green', 'blue', 'orange', 'red'])
        ax8.set_ylabel('Normalized Data Rate', fontsize=11)
        ax8.set_title('Configuration Comparison', fontsize=12, fontweight='bold')
        ax8.set_xticks(x_pos)
        ax8.set_xticklabels(config_names)
        ax8.grid(True, alpha=0.3, axis='y')
        
        # Add cost as text
        for i, (cost, mass) in enumerate(zip(config_costs, config_masses)):
            ax8.text(i, norm_datarates[i] + 0.05, 
                    f'${cost:.0f}M\n{mass:.0f}kg',
                    ha='center', fontsize=8)
        
        # Plot 9: Multi-objective Trade Space
        ax9 = plt.subplot(3, 3, 9)
        
        # Flatten arrays for scatter plot
        power_flat = results['power_mesh'].flatten()
        aperture_flat = results['aperture_mesh'].flatten()
        datarate_flat = results['data_rate_mesh'].flatten()
        cost_flat = results['cost_mesh'].flatten()
        
        # Sample points to avoid overcrowding
        sample_idx = np.random.choice(len(power_flat), 500, replace=False)
        
        scatter = ax9.scatter(cost_flat[sample_idx], datarate_flat[sample_idx],
                             c=aperture_flat[sample_idx], s=power_flat[sample_idx]*2,
                             cmap='coolwarm', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax9.set_xlabel('System Cost (M$)', fontsize=11)
        ax9.set_ylabel('Data Rate (Gbps)', fontsize=11)
        ax9.set_title('Cost-Performance Trade Space\n(color=aperture, size=power)',
                     fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax9)
        cbar.set_label('Aperture (m)', fontsize=10)
        
        # Highlight Pareto-efficient region
        ax9.annotate('High Efficiency\nRegion', xy=(60, 80), fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/optical_trade_study.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved optical trade study plots to {output_dir}/optical_trade_study.png")
        
        return fig


def main():
    """Run the optical trade study."""
    study = OpticalTradeStudy()
    results = study.run_trade_study()
    study.plot_results(results, '/home/claude/mission-trades/plots')
    return results


if __name__ == '__main__':
    main()
