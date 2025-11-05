"""
Orbit Altitude and Inclination vs Coverage Trade Study

Analyzes Earth observation coverage as a function of orbital parameters.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


class OrbitTradeStudy:
    """Trade study for orbit altitude and inclination vs coverage."""
    
    def __init__(self):
        self.earth_radius = 6371  # km
        self.altitude_range = np.linspace(400, 1500, 50)  # km (LEO to MEO)
        self.inclination_range = np.linspace(0, 98, 50)  # degrees
        
    def swath_width(self, altitude, max_viewing_angle=55):
        """
        Calculate ground swath width for a given altitude.
        
        Args:
            altitude: Orbital altitude in km
            max_viewing_angle: Maximum off-nadir viewing angle in degrees
        """
        theta = np.radians(max_viewing_angle)
        R = self.earth_radius
        h = altitude
        
        # Swath half-angle from geometry
        rho = np.arcsin((R / (R + h)) * np.sin(theta))
        swath_half = R * (theta - rho)
        
        return 2 * swath_half  # Full swath width in km
    
    def revisit_time(self, altitude, inclination):
        """
        Estimate revisit time based on altitude and inclination.
        
        Higher altitude = longer orbital period = fewer passes per day
        Higher inclination = better coverage at high latitudes
        """
        # Orbital period (simplified Kepler's third law)
        mu = 398600  # Earth's gravitational parameter (km^3/s^2)
        a = self.earth_radius + altitude  # semi-major axis
        period = 2 * np.pi * np.sqrt(a**3 / mu) / 3600  # hours
        
        # Number of orbits per day
        orbits_per_day = 24 / period
        
        # Coverage factor based on inclination
        # Sun-synchronous orbits (~98°) provide excellent global coverage
        if inclination >= 90:
            coverage_factor = 1.0  # Polar orbits cover everything
        elif inclination >= 45:
            coverage_factor = 0.5 + 0.5 * ((inclination - 45) / 53)
        else:
            coverage_factor = 0.3 + 0.2 * (inclination / 45)  # Equatorial coverage
        
        # Effective revisit time (days)
        # Lower is better
        revisit = (1.0 / (orbits_per_day * coverage_factor)) * 2
        
        return revisit
    
    def coverage_area(self, altitude, inclination):
        """
        Calculate accessible ground coverage area.
        
        Returns area in million km^2 per orbit.
        """
        swath = self.swath_width(altitude)
        
        # Orbital period
        mu = 398600
        a = self.earth_radius + altitude
        period = 2 * np.pi * np.sqrt(a**3 / mu)  # seconds
        
        # Ground track length per orbit (approximate)
        ground_velocity = 2 * np.pi * self.earth_radius * 1000 / period  # m/s
        track_length = ground_velocity * period / 1000  # km
        
        # Coverage area per orbit
        coverage = (swath * track_length) / 1e6  # million km^2
        
        # Inclination factor (higher inclination = more unique coverage)
        inc_factor = 0.3 + 0.7 * (min(inclination, 90) / 90)
        
        return coverage * inc_factor
    
    def power_requirement(self, altitude):
        """
        Estimate power requirements as a function of altitude.
        
        Higher altitude requires more power for communication and operations.
        """
        # Base power (watts)
        base_power = 500
        
        # Altitude-dependent power (communication link budget)
        altitude_power = 200 * (altitude / 800)  # scales with altitude
        
        # Total power
        return base_power + altitude_power
    
    def mission_lifetime(self, altitude):
        """
        Estimate mission lifetime based on altitude (atmospheric drag).
        
        Lower altitudes have shorter lifetimes due to atmospheric drag.
        """
        # Simplified exponential model
        if altitude < 300:
            return 0.5  # Very short lifetime
        elif altitude < 500:
            return 2 + 8 * (altitude - 300) / 200
        else:
            return 10 + 5 * np.log10((altitude - 500) / 100 + 1)
    
    def run_trade_study(self):
        """Execute the full orbit trade study."""
        results = {}
        
        # 1D analyses
        swath_widths = [self.swath_width(alt) for alt in self.altitude_range]
        power_reqs = [self.power_requirement(alt) for alt in self.altitude_range]
        lifetimes = [self.mission_lifetime(alt) for alt in self.altitude_range]
        
        # 2D analyses - create meshgrid
        alt_mesh, inc_mesh = np.meshgrid(self.altitude_range, self.inclination_range)
        
        revisit_mesh = np.zeros_like(alt_mesh)
        coverage_mesh = np.zeros_like(alt_mesh)
        
        for i in range(alt_mesh.shape[0]):
            for j in range(alt_mesh.shape[1]):
                alt = alt_mesh[i, j]
                inc = inc_mesh[i, j]
                revisit_mesh[i, j] = self.revisit_time(alt, inc)
                coverage_mesh[i, j] = self.coverage_area(alt, inc)
        
        results = {
            'altitude': self.altitude_range,
            'inclination': self.inclination_range,
            'swath_width': np.array(swath_widths),
            'power_requirement': np.array(power_reqs),
            'lifetime': np.array(lifetimes),
            'altitude_mesh': alt_mesh,
            'inclination_mesh': inc_mesh,
            'revisit_mesh': revisit_mesh,
            'coverage_mesh': coverage_mesh
        }
        
        return results
    
    def plot_results(self, results, output_dir):
        """Generate comprehensive orbit trade study plots."""
        
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Swath Width vs Altitude
        ax1 = plt.subplot(3, 3, 1)
        ax1.plot(results['altitude'], results['swath_width'], 'b-', linewidth=2)
        ax1.set_xlabel('Altitude (km)', fontsize=11)
        ax1.set_ylabel('Swath Width (km)', fontsize=11)
        ax1.set_title('Ground Swath vs Altitude', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power vs Altitude
        ax2 = plt.subplot(3, 3, 2)
        ax2.plot(results['altitude'], results['power_requirement'], 'r-', linewidth=2)
        ax2.set_xlabel('Altitude (km)', fontsize=11)
        ax2.set_ylabel('Power Requirement (W)', fontsize=11)
        ax2.set_title('Power Requirements vs Altitude', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Mission Lifetime vs Altitude
        ax3 = plt.subplot(3, 3, 3)
        ax3.plot(results['altitude'], results['lifetime'], 'g-', linewidth=2)
        ax3.set_xlabel('Altitude (km)', fontsize=11)
        ax3.set_ylabel('Mission Lifetime (years)', fontsize=11)
        ax3.set_title('Expected Lifetime vs Altitude', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=5, color='orange', linestyle='--', label='5-year requirement')
        ax3.legend()
        
        # Plot 4: 2D Revisit Time Contour
        ax4 = plt.subplot(3, 3, 4)
        contour1 = ax4.contourf(results['altitude_mesh'], results['inclination_mesh'],
                                results['revisit_mesh'], levels=20, cmap='RdYlGn_r')
        ax4.set_xlabel('Altitude (km)', fontsize=11)
        ax4.set_ylabel('Inclination (degrees)', fontsize=11)
        ax4.set_title('Revisit Time (days)', fontsize=12, fontweight='bold')
        cbar1 = plt.colorbar(contour1, ax=ax4)
        cbar1.set_label('Days', fontsize=10)
        
        # Add contour lines
        contour_lines1 = ax4.contour(results['altitude_mesh'], results['inclination_mesh'],
                                     results['revisit_mesh'], levels=10, colors='black', 
                                     linewidths=0.5, alpha=0.4)
        ax4.clabel(contour_lines1, inline=True, fontsize=8)
        
        # Plot 5: 2D Coverage Area Contour
        ax5 = plt.subplot(3, 3, 5)
        contour2 = ax5.contourf(results['altitude_mesh'], results['inclination_mesh'],
                                results['coverage_mesh'], levels=20, cmap='viridis')
        ax5.set_xlabel('Altitude (km)', fontsize=11)
        ax5.set_ylabel('Inclination (degrees)', fontsize=11)
        ax5.set_title('Coverage Area per Orbit', fontsize=12, fontweight='bold')
        cbar2 = plt.colorbar(contour2, ax=ax5)
        cbar2.set_label('Million km²', fontsize=10)
        
        # Add contour lines
        contour_lines2 = ax5.contour(results['altitude_mesh'], results['inclination_mesh'],
                                     results['coverage_mesh'], levels=10, colors='white',
                                     linewidths=0.5, alpha=0.6)
        ax5.clabel(contour_lines2, inline=True, fontsize=8)
        
        # Plot 6: 3D Surface - Revisit Time
        ax6 = fig.add_subplot(3, 3, 6, projection='3d')
        surf1 = ax6.plot_surface(results['altitude_mesh'], results['inclination_mesh'],
                                results['revisit_mesh'], cmap='coolwarm', alpha=0.8)
        ax6.set_xlabel('Altitude (km)', fontsize=10)
        ax6.set_ylabel('Inclination (°)', fontsize=10)
        ax6.set_zlabel('Revisit (days)', fontsize=10)
        ax6.set_title('3D Revisit Time Surface', fontsize=12, fontweight='bold')
        plt.colorbar(surf1, ax=ax6, shrink=0.5)
        
        # Plot 7: Orbit Selection Recommendations
        ax7 = plt.subplot(3, 3, 7)
        
        # Define key orbit types
        orbits = {
            'LEO Low\n(400 km, 51°)': (400, 51),
            'LEO Mid\n(600 km, 98°)': (600, 98),
            'LEO High\n(800 km, 98°)': (800, 98),
            'MEO\n(1200 km, 60°)': (1200, 60)
        }
        
        orbit_names = list(orbits.keys())
        revisit_times = []
        coverage_areas = []
        
        for name, (alt, inc) in orbits.items():
            revisit_times.append(self.revisit_time(alt, inc))
            coverage_areas.append(self.coverage_area(alt, inc))
        
        x_pos = np.arange(len(orbit_names))
        bars = ax7.bar(x_pos, revisit_times, alpha=0.7, color=['blue', 'green', 'orange', 'red'])
        ax7.set_ylabel('Revisit Time (days)', fontsize=11)
        ax7.set_title('Orbit Configuration Comparison', fontsize=12, fontweight='bold')
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(orbit_names, fontsize=9)
        ax7.grid(True, alpha=0.3, axis='y')
        
        # Plot 8: Coverage vs Power Trade
        ax8 = plt.subplot(3, 3, 8)
        
        # Calculate average coverage for each altitude
        avg_coverage = np.mean(results['coverage_mesh'], axis=0)
        
        scatter = ax8.scatter(results['power_requirement'], avg_coverage,
                             c=results['altitude'], cmap='plasma', s=100, alpha=0.6)
        ax8.set_xlabel('Power Requirement (W)', fontsize=11)
        ax8.set_ylabel('Avg Coverage (million km²)', fontsize=11)
        ax8.set_title('Coverage vs Power Trade', fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax8)
        cbar.set_label('Altitude (km)', fontsize=10)
        
        # Plot 9: Mission Design Space
        ax9 = plt.subplot(3, 3, 9)
        
        # Create scatter plot showing trade space
        # Color by revisit time, size by lifetime
        min_revisit = np.min(results['revisit_mesh'], axis=0)
        lifetimes_norm = (results['lifetime'] - results['lifetime'].min()) / \
                        (results['lifetime'].max() - results['lifetime'].min())
        
        scatter2 = ax9.scatter(results['altitude'], results['swath_width'],
                              c=min_revisit, s=lifetimes_norm*300 + 50,
                              cmap='RdYlGn_r', alpha=0.6, edgecolors='black', linewidth=0.5)
        ax9.set_xlabel('Altitude (km)', fontsize=11)
        ax9.set_ylabel('Swath Width (km)', fontsize=11)
        ax9.set_title('Mission Design Space\n(size = lifetime, color = revisit)', 
                     fontsize=12, fontweight='bold')
        ax9.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter2, ax=ax9)
        cbar.set_label('Min Revisit Time (days)', fontsize=10)
        
        # Annotate optimal regions
        ax9.annotate('Good Balance', xy=(650, 2800), fontsize=9,
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/orbit_trade_study.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved orbit trade study plots to {output_dir}/orbit_trade_study.png")
        
        return fig


def main():
    """Run the orbit trade study."""
    study = OrbitTradeStudy()
    results = study.run_trade_study()
    study.plot_results(results, '/home/claude/mission-trades/plots')
    return results


if __name__ == '__main__':
    main()
