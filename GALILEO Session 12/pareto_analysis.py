"""
Pareto Front Analysis and Multi-Objective Optimization

Identifies Pareto-optimal solutions across all trade studies.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial import ConvexHull


class ParetoAnalysis:
    """Multi-objective Pareto front analysis for mission design."""
    
    def __init__(self):
        self.design_points = []
        
    def is_dominated(self, point, other_points, objectives_to_maximize):
        """
        Check if a point is dominated by any other point.
        
        A point is dominated if there exists another point that is better
        in at least one objective and no worse in all other objectives.
        
        Args:
            point: The point to check (array of objective values)
            other_points: List of other points
            objectives_to_maximize: Boolean array indicating which objectives to maximize
        """
        for other in other_points:
            if np.array_equal(point, other):
                continue
                
            # Check if 'other' dominates 'point'
            better_in_one = False
            worse_in_any = False
            
            for i, maximize in enumerate(objectives_to_maximize):
                if maximize:
                    if other[i] > point[i]:
                        better_in_one = True
                    elif other[i] < point[i]:
                        worse_in_any = True
                else:
                    if other[i] < point[i]:
                        better_in_one = True
                    elif other[i] > point[i]:
                        worse_in_any = True
            
            if better_in_one and not worse_in_any:
                return True
        
        return False
    
    def find_pareto_front(self, objectives, objectives_to_maximize):
        """
        Find the Pareto front from a set of design points.
        
        Args:
            objectives: Array of shape (n_points, n_objectives)
            objectives_to_maximize: Boolean array indicating direction for each objective
        
        Returns:
            pareto_indices: Indices of points on the Pareto front
        """
        n_points = objectives.shape[0]
        pareto_indices = []
        
        for i in range(n_points):
            if not self.is_dominated(objectives[i], objectives, objectives_to_maximize):
                pareto_indices.append(i)
        
        return np.array(pareto_indices)
    
    def normalize_objectives(self, objectives):
        """Normalize objectives to [0, 1] range for comparison."""
        normalized = np.zeros_like(objectives)
        for i in range(objectives.shape[1]):
            min_val = np.min(objectives[:, i])
            max_val = np.max(objectives[:, i])
            if max_val > min_val:
                normalized[:, i] = (objectives[:, i] - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.5
        return normalized
    
    def hypervolume_indicator(self, pareto_front, reference_point):
        """
        Calculate hypervolume indicator for Pareto front quality.
        
        Higher is better.
        """
        # Simplified 2D hypervolume calculation
        if pareto_front.shape[1] != 2:
            return None
        
        # Sort by first objective
        sorted_front = pareto_front[pareto_front[:, 0].argsort()]
        
        hypervolume = 0
        for i in range(len(sorted_front)):
            if i == 0:
                width = sorted_front[i, 0] - reference_point[0]
            else:
                width = sorted_front[i, 0] - sorted_front[i-1, 0]
            
            height = reference_point[1] - sorted_front[i, 1]
            hypervolume += width * height
        
        return hypervolume
    
    def generate_design_space(self, n_samples=1000):
        """
        Generate sample design space for integrated mission analysis.
        
        Design variables:
        - Baseline length (m)
        - Orbit altitude (km)
        - Optical power (W)
        - Aperture diameter (m)
        
        Objectives:
        - Maximize: Angular resolution, coverage, data rate, lifetime
        - Minimize: Cost, mass, power consumption, revisit time
        """
        np.random.seed(42)
        
        designs = {
            'baseline': np.random.uniform(50, 500, n_samples),
            'altitude': np.random.uniform(500, 1200, n_samples),
            'power': np.random.uniform(10, 80, n_samples),
            'aperture': np.random.uniform(0.3, 1.5, n_samples)
        }
        
        # Calculate objectives for each design
        objectives = {}
        
        # Angular resolution (maximize, so we'll use 1/resolution)
        wavelength = 10e-6
        objectives['resolution_inv'] = designs['baseline'] / wavelength * 1e-6
        
        # Coverage area (maximize)
        earth_radius = 6371
        swath_width = 2 * earth_radius * np.arcsin(
            np.sin(np.radians(55)) / (1 + designs['altitude'] / earth_radius)
        )
        objectives['coverage'] = swath_width * 2000  # Simplified
        
        # Data rate (maximize)
        objectives['data_rate'] = designs['power'] * designs['aperture']**2 * 2
        
        # Mission lifetime (maximize)
        objectives['lifetime'] = 2 + 8 * np.log10((designs['altitude'] - 400) / 100 + 1)
        
        # Cost (minimize)
        objectives['cost'] = (10 * (designs['aperture'])**2 + 
                            designs['power'] * 0.1 + 
                            designs['baseline'] * 0.05 +
                            20)
        
        # Mass (minimize)
        objectives['mass'] = (50 * designs['aperture']**2.5 + 
                            designs['power'] * 0.5 + 
                            designs['baseline'] * 0.3 +
                            50)
        
        # Power consumption (minimize)
        objectives['power_consumption'] = (200 * (designs['altitude'] / 800) + 
                                          designs['power'] * 1.2 + 
                                          500)
        
        # Revisit time (minimize)
        mu = 398600
        period = 2 * np.pi * np.sqrt((earth_radius + designs['altitude'])**3 / mu) / 3600
        objectives['revisit_time'] = 24 / period * 0.5
        
        return designs, objectives
    
    def run_pareto_analysis(self):
        """Run comprehensive Pareto analysis on mission design space."""
        
        # Generate design space
        designs, objectives = self.generate_design_space(n_samples=1000)
        
        # Define key trade-offs for Pareto analysis
        analyses = {}
        
        # 1. Performance vs Cost
        obj_perf_cost = np.column_stack([
            objectives['data_rate'],  # maximize
            objectives['cost']  # minimize
        ])
        pareto_perf_cost = self.find_pareto_front(obj_perf_cost, [True, False])
        analyses['performance_cost'] = {
            'objectives': obj_perf_cost,
            'pareto_indices': pareto_perf_cost,
            'labels': ['Data Rate (Gbps)', 'Cost (M$)']
        }
        
        # 2. Coverage vs Revisit Time
        obj_cov_revisit = np.column_stack([
            objectives['coverage'],  # maximize
            objectives['revisit_time']  # minimize
        ])
        pareto_cov_revisit = self.find_pareto_front(obj_cov_revisit, [True, False])
        analyses['coverage_revisit'] = {
            'objectives': obj_cov_revisit,
            'pareto_indices': pareto_cov_revisit,
            'labels': ['Coverage Area', 'Revisit Time (days)']
        }
        
        # 3. Resolution vs Mass
        obj_res_mass = np.column_stack([
            objectives['resolution_inv'],  # maximize (1/resolution)
            objectives['mass']  # minimize
        ])
        pareto_res_mass = self.find_pareto_front(obj_res_mass, [True, False])
        analyses['resolution_mass'] = {
            'objectives': obj_res_mass,
            'pareto_indices': pareto_res_mass,
            'labels': ['Resolution (1/μas)', 'Mass (kg)']
        }
        
        # 4. Lifetime vs Power
        obj_life_power = np.column_stack([
            objectives['lifetime'],  # maximize
            objectives['power_consumption']  # minimize
        ])
        pareto_life_power = self.find_pareto_front(obj_life_power, [True, False])
        analyses['lifetime_power'] = {
            'objectives': obj_life_power,
            'pareto_indices': pareto_life_power,
            'labels': ['Lifetime (years)', 'Power (W)']
        }
        
        # 5. Multi-objective: Performance + Cost + Mass (3D)
        obj_multi = np.column_stack([
            objectives['data_rate'],  # maximize
            objectives['cost'],  # minimize
            objectives['mass']  # minimize
        ])
        pareto_multi = self.find_pareto_front(obj_multi, [True, False, False])
        analyses['multi_objective'] = {
            'objectives': obj_multi,
            'pareto_indices': pareto_multi,
            'labels': ['Data Rate (Gbps)', 'Cost (M$)', 'Mass (kg)']
        }
        
        return designs, objectives, analyses
    
    def plot_pareto_fronts(self, designs, objectives, analyses, output_dir):
        """Generate comprehensive Pareto front visualizations."""
        
        fig = plt.figure(figsize=(18, 12))
        
        # Plot 1: Performance vs Cost
        ax1 = plt.subplot(2, 3, 1)
        data = analyses['performance_cost']
        obj = data['objectives']
        pareto_idx = data['pareto_indices']
        
        ax1.scatter(obj[:, 1], obj[:, 0], c='lightblue', s=20, alpha=0.5, label='All Designs')
        ax1.scatter(obj[pareto_idx, 1], obj[pareto_idx, 0], 
                   c='red', s=60, marker='s', label='Pareto Front', edgecolors='black')
        
        # Connect Pareto points
        pareto_sorted = pareto_idx[np.argsort(obj[pareto_idx, 1])]
        ax1.plot(obj[pareto_sorted, 1], obj[pareto_sorted, 0], 'r--', linewidth=2, alpha=0.7)
        
        ax1.set_xlabel(data['labels'][1], fontsize=11)
        ax1.set_ylabel(data['labels'][0], fontsize=11)
        ax1.set_title('Performance vs Cost Trade-off', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coverage vs Revisit
        ax2 = plt.subplot(2, 3, 2)
        data = analyses['coverage_revisit']
        obj = data['objectives']
        pareto_idx = data['pareto_indices']
        
        ax2.scatter(obj[:, 1], obj[:, 0], c='lightgreen', s=20, alpha=0.5, label='All Designs')
        ax2.scatter(obj[pareto_idx, 1], obj[pareto_idx, 0],
                   c='darkgreen', s=60, marker='s', label='Pareto Front', edgecolors='black')
        
        pareto_sorted = pareto_idx[np.argsort(obj[pareto_idx, 1])]
        ax2.plot(obj[pareto_sorted, 1], obj[pareto_sorted, 0], 'g--', linewidth=2, alpha=0.7)
        
        ax2.set_xlabel(data['labels'][1], fontsize=11)
        ax2.set_ylabel(data['labels'][0], fontsize=11)
        ax2.set_title('Coverage vs Revisit Time', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Resolution vs Mass
        ax3 = plt.subplot(2, 3, 3)
        data = analyses['resolution_mass']
        obj = data['objectives']
        pareto_idx = data['pareto_indices']
        
        ax3.scatter(obj[:, 1], obj[:, 0], c='lightyellow', s=20, alpha=0.5, label='All Designs')
        ax3.scatter(obj[pareto_idx, 1], obj[pareto_idx, 0],
                   c='orange', s=60, marker='s', label='Pareto Front', edgecolors='black')
        
        pareto_sorted = pareto_idx[np.argsort(obj[pareto_idx, 1])]
        ax3.plot(obj[pareto_sorted, 1], obj[pareto_sorted, 0], 'orange', 
                linestyle='--', linewidth=2, alpha=0.7)
        
        ax3.set_xlabel(data['labels'][1], fontsize=11)
        ax3.set_ylabel(data['labels'][0], fontsize=11)
        ax3.set_title('Resolution vs Mass Trade-off', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Lifetime vs Power
        ax4 = plt.subplot(2, 3, 4)
        data = analyses['lifetime_power']
        obj = data['objectives']
        pareto_idx = data['pareto_indices']
        
        ax4.scatter(obj[:, 1], obj[:, 0], c='lightcoral', s=20, alpha=0.5, label='All Designs')
        ax4.scatter(obj[pareto_idx, 1], obj[pareto_idx, 0],
                   c='darkred', s=60, marker='s', label='Pareto Front', edgecolors='black')
        
        pareto_sorted = pareto_idx[np.argsort(obj[pareto_idx, 1])]
        ax4.plot(obj[pareto_sorted, 1], obj[pareto_sorted, 0], 'r--', linewidth=2, alpha=0.7)
        
        ax4.set_xlabel(data['labels'][1], fontsize=11)
        ax4.set_ylabel(data['labels'][0], fontsize=11)
        ax4.set_title('Lifetime vs Power Consumption', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: 3D Pareto Front
        ax5 = fig.add_subplot(2, 3, 5, projection='3d')
        data = analyses['multi_objective']
        obj = data['objectives']
        pareto_idx = data['pareto_indices']
        
        ax5.scatter(obj[:, 1], obj[:, 2], obj[:, 0], 
                   c='lightblue', s=10, alpha=0.3, label='All Designs')
        ax5.scatter(obj[pareto_idx, 1], obj[pareto_idx, 2], obj[pareto_idx, 0],
                   c='red', s=80, marker='s', label='Pareto Front', 
                   edgecolors='black', linewidth=1)
        
        ax5.set_xlabel(data['labels'][1], fontsize=10)
        ax5.set_ylabel(data['labels'][2], fontsize=10)
        ax5.set_zlabel(data['labels'][0], fontsize=10)
        ax5.set_title('3D Multi-Objective Pareto Front', fontsize=12, fontweight='bold')
        ax5.legend()
        
        # Plot 6: Design Space Exploration
        ax6 = plt.subplot(2, 3, 6)
        
        # Color by position on Pareto front
        is_pareto = np.zeros(len(obj), dtype=bool)
        is_pareto[pareto_idx] = True
        
        colors = ['red' if p else 'blue' for p in is_pareto]
        sizes = [60 if p else 20 for p in is_pareto]
        alphas = [0.8 if p else 0.3 for p in is_pareto]
        
        for i, (c, s, a) in enumerate(zip(colors, sizes, alphas)):
            ax6.scatter(obj[i, 1], obj[i, 0], c=c, s=s, alpha=a)
        
        # Add annotations for key points
        best_performance_idx = pareto_idx[np.argmax(obj[pareto_idx, 0])]
        lowest_cost_idx = pareto_idx[np.argmin(obj[pareto_idx, 1])]
        
        ax6.annotate('Best Performance', 
                    xy=(obj[best_performance_idx, 1], obj[best_performance_idx, 0]),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        ax6.annotate('Lowest Cost',
                    xy=(obj[lowest_cost_idx, 1], obj[lowest_cost_idx, 0]),
                    xytext=(10, -20), textcoords='offset points',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='black'))
        
        ax6.set_xlabel('Cost (M$)', fontsize=11)
        ax6.set_ylabel('Data Rate (Gbps)', fontsize=11)
        ax6.set_title('Design Point Recommendations', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                   markersize=8, label='Pareto Optimal'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
                   markersize=6, alpha=0.5, label='Dominated')
        ]
        ax6.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pareto_fronts.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved Pareto front analysis to {output_dir}/pareto_fronts.png")
        
        return fig


def main():
    """Run Pareto front analysis."""
    analysis = ParetoAnalysis()
    designs, objectives, analyses = analysis.run_pareto_analysis()
    analysis.plot_pareto_fronts(designs, objectives, analyses, 
                                '/home/claude/mission-trades/plots')
    return designs, objectives, analyses


if __name__ == '__main__':
    main()
