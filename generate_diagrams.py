"""
Generate C4 Model architecture diagrams for GeoSense Platform.

Creates:
1. Context diagram - System in environment
2. Container diagram - High-level technical components
3. Component diagram - Internal component structure
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def create_context_diagram():
    """Create system context diagram showing GeoSense in its environment."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'GeoSense Platform - System Context', 
            ha='center', fontsize=16, fontweight='bold')
    
    # Central system
    system_box = FancyBboxPatch((4.5, 4), 5, 2, 
                                boxstyle="round,pad=0.1", 
                                edgecolor='#2E86AB', facecolor='#A4C3D2',
                                linewidth=3)
    ax.add_patch(system_box)
    ax.text(7, 5.2, 'GeoSense Platform', ha='center', fontsize=14, fontweight='bold')
    ax.text(7, 4.7, 'Space-based Gravimetric', ha='center', fontsize=10)
    ax.text(7, 4.4, 'Sensing & Analysis', ha='center', fontsize=10)
    
    # External actors and systems
    actors = [
        # (x, y, width, height, name, description, color)
        (0.5, 7, 2.5, 1.5, 'Scientists', 'Earth Science\nResearchers', '#E8B4B8'),
        (11, 7, 2.5, 1.5, 'Operators', 'Mission Control\nEngineers', '#E8B4B8'),
        (0.5, 2, 2.5, 1.5, 'Data Archives', 'EGM2008, GRACE,\nGNSS Ephemeris', '#C9ADA7'),
        (11, 2, 2.5, 1.5, 'Cloud Services', 'AWS/GCP/Azure\nInfrastructure', '#C9ADA7'),
        (5.25, 0.5, 3.5, 1.2, 'Satellites', 'GNSS Constellation\n& Reference Frames', '#C9ADA7'),
    ]
    
    for x, y, w, h, name, desc, color in actors:
        box = FancyBboxPatch((x, y), w, h, 
                           boxstyle="round,pad=0.08", 
                           edgecolor='black', facecolor=color,
                           linewidth=2)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2 + 0.2, name, 
               ha='center', fontsize=11, fontweight='bold')
        ax.text(x + w/2, y + h/2 - 0.25, desc, 
               ha='center', fontsize=8)
    
    # Relationships
    relationships = [
        # (x1, y1, x2, y2, label, style)
        (2, 7.5, 4.5, 5.5, 'Query data,\nvisualize', 'simple'),
        (11, 5.5, 9.5, 5.5, 'Monitor,\ncontrol', 'simple'),
        (2.5, 2.7, 4.5, 4.5, 'Load gravity\nmodels', 'simple'),
        (11, 3, 9.5, 4.5, 'Deploy to,\nscale on', 'simple'),
        (7, 1.7, 7, 4, 'Receive GNSS\nobservations', 'simple'),
    ]
    
    for x1, y1, x2, y2, label, style in relationships:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=20,
                              linewidth=2, color='#555555')
        ax.add_patch(arrow)
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x, mid_y, label, ha='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig


def create_container_diagram():
    """Create container diagram showing major technical components."""
    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(8, 11.5, 'GeoSense Platform - Container Diagram', 
            ha='center', fontsize=16, fontweight='bold')
    
    # System boundary
    boundary = patches.Rectangle((0.3, 0.3), 15.4, 10.5, 
                                linewidth=2, edgecolor='#2E86AB', 
                                facecolor='none', linestyle='--')
    ax.add_patch(boundary)
    ax.text(8, 10.5, 'GeoSense Platform', ha='center', 
           fontsize=12, fontweight='bold', color='#2E86AB')
    
    # Containers
    containers = [
        # UI Layer
        (1, 8.5, 3, 1.5, 'Web UI', 'Next.js + CesiumJS\nReact + TypeScript', '#A4C3D2'),
        
        # Application Layer
        (1, 6.5, 2, 1.3, 'Simulation\nEngine', 'Python + JAX\nOrbit & Gravity', '#FFD6A5'),
        (3.5, 6.5, 2, 1.3, 'Sensing\nPipeline', 'Python + NumPy\nData Processing', '#FFD6A5'),
        (6, 6.5, 2, 1.3, 'Inversion\nEngine', 'Python + JAX\nTikhonov/Bayes', '#FFD6A5'),
        (8.5, 6.5, 2, 1.3, 'ML Pipeline', 'Flax + PyTorch\nNeural Networks', '#FFD6A5'),
        
        # Control Layer
        (11, 8.5, 4, 1.5, 'Control Systems (Rust)', 
         'Orbit Dynamics • Attitude • Power\nnalgebra + tokio', '#CAFFBF'),
        
        # API Layer
        (11, 6.5, 4, 1.3, 'REST API', 'FastAPI + Uvicorn\nPython 3.11', '#9BF6FF'),
        
        # Worker Layer
        (1, 4.5, 4, 1.2, 'Async Workers', 'Celery + Python\nBackground Tasks', '#FFC6FF'),
        (5.5, 4.5, 4, 1.2, 'Task Scheduler', 'Celery Beat\nMission Planning', '#FFC6FF'),
        
        # Data Layer
        (1, 2.5, 2.5, 1.3, 'PostgreSQL', 'Metadata &\nResults DB', '#FDFFB6'),
        (4, 2.5, 2.5, 1.3, 'TimescaleDB', 'Time Series\nTelemetry', '#FDFFB6'),
        (7, 2.5, 2.5, 1.3, 'Redis', 'Cache & Queue\nIn-Memory DB', '#FDFFB6'),
        (10, 2.5, 2.5, 1.3, 'Object Store', 'S3/MinIO\nBinary Data', '#FDFFB6'),
        
        # Monitoring
        (13, 4.5, 2.5, 1.2, 'Monitoring', 'Grafana + Prometheus\nJaeger Tracing', '#EECBFF'),
        (13, 2.5, 2.5, 1.3, 'Logging', 'ELK Stack\nCentralized Logs', '#EECBFF'),
    ]
    
    for x, y, w, h, name, desc, color in containers:
        box = FancyBboxPatch((x, y), w, h, 
                           boxstyle="round,pad=0.08", 
                           edgecolor='black', facecolor=color,
                           linewidth=2)
        ax.add_patch(box)
        lines = name.split('\n')
        y_offset = 0.15 * len(lines)
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h/2 + y_offset - i*0.3, line, 
                   ha='center', fontsize=10, fontweight='bold')
        ax.text(x + w/2, y + h/2 - 0.35, desc, 
               ha='center', fontsize=7, style='italic')
    
    # Key relationships (simplified)
    arrows = [
        (2.5, 8.5, 2.5, 7.8, 'views'),
        (4.5, 8.5, 13, 8.5, 'API calls'),
        (13, 7.8, 13, 6.5, 'serves'),
        (7, 6.5, 7, 5.7, 'submits'),
        (3, 4.5, 3, 3.8, 'stores'),
        (7.5, 4.5, 8, 3.8, 'writes'),
    ]
    
    for x1, y1, x2, y2, label in arrows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=15,
                              linewidth=1.5, color='#555555')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig


def create_component_diagram():
    """Create component diagram for inversion engine."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'GeoSense Platform - Inversion Engine Components', 
            ha='center', fontsize=16, fontweight='bold')
    
    # System boundary
    boundary = patches.Rectangle((0.5, 0.5), 13, 8.5, 
                                linewidth=2, edgecolor='#2E86AB', 
                                facecolor='#F0F0F0', linestyle='-', alpha=0.3)
    ax.add_patch(boundary)
    ax.text(7, 8.7, 'Inversion Engine', ha='center', 
           fontsize=12, fontweight='bold', color='#2E86AB')
    
    # Components
    components = [
        # Input layer
        (1, 7, 2, 0.8, 'Data Loader', 'Load measurements', '#A4C3D2'),
        (3.5, 7, 2, 0.8, 'Preprocessor', 'Filter & calibrate', '#A4C3D2'),
        
        # Core algorithms
        (1, 5, 2, 1, 'Forward\nOperator', 'G: m → d\nGravity kernel', '#FFD6A5'),
        (3.5, 5, 2, 1, 'Tikhonov\nSolver', 'L2 regularization\nNormal equations', '#FFD6A5'),
        (6, 5, 2, 1, 'Bayesian\nEstimator', 'MAP/MCMC\nUncertainty', '#FFD6A5'),
        (8.5, 5, 2, 1, 'Constraint\nManager', 'Bounds, priors\nRegularization', '#FFD6A5'),
        
        # Support components
        (11, 7, 2, 0.8, 'Resolution\nAnalysis', 'Compute R matrix', '#CAFFBF'),
        (11, 5.5, 2, 0.8, 'Validation', 'Cross-validation\nMetrics', '#CAFFBF'),
        
        # Solver backend
        (1, 2.5, 3, 1.2, 'JAX Backend', 'Auto-diff, JIT compile\nGPU acceleration', '#FFC6FF'),
        (4.5, 2.5, 3, 1.2, 'SciPy Backend', 'Sparse solvers\nLSQR, CG, GMRES', '#FFC6FF'),
        
        # Output
        (8, 2.5, 5, 1.2, 'Result Manager', 
         'Model storage • Metadata • Visualization prep', '#FDFFB6'),
        
        # External
        (1, 1, 2, 0.7, 'Config', 'YAML settings', '#EECBFF'),
        (11, 1, 2, 0.7, 'Cache', 'Kernel matrix', '#EECBFF'),
    ]
    
    for x, y, w, h, name, desc, color in components:
        box = FancyBboxPatch((x, y), w, h, 
                           boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor=color,
                           linewidth=1.5)
        ax.add_patch(box)
        lines = name.split('\n')
        y_offset = 0.1 * len(lines)
        for i, line in enumerate(lines):
            ax.text(x + w/2, y + h/2 + y_offset - i*0.2, line, 
                   ha='center', fontsize=9, fontweight='bold')
        ax.text(x + w/2, y + h/2 - 0.25, desc, 
               ha='center', fontsize=6, style='italic')
    
    # Data flow
    flows = [
        (2, 7, 4.5, 7, 'raw data'),
        (4.5, 6.2, 4.5, 5.5, 'clean data'),
        (2, 5, 3.5, 5.5, 'model'),
        (3, 5, 4.5, 5.5, 'uses'),
        (5, 5, 6, 5.5, 'alternative'),
        (7, 5, 8.5, 5.5, 'applies'),
        (9.5, 5, 11, 5.9, 'validates'),
        (2.5, 3.7, 2.5, 5, 'compute'),
        (6, 3.7, 6, 5, 'solve'),
        (10, 3.7, 4.5, 5.5, 'store'),
    ]
    
    for x1, y1, x2, y2, label in flows:
        arrow = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=12,
                              linewidth=1.2, color='#555555')
        ax.add_patch(arrow)
    
    plt.tight_layout()
    return fig


def save_all_diagrams():
    """Generate and save all architecture diagrams."""
    # Create output directory
    import os
    os.makedirs('/home/claude/geosense-platform/docs/architecture', exist_ok=True)
    
    # Generate diagrams
    print("Generating context diagram...")
    fig1 = create_context_diagram()
    fig1.savefig('/home/claude/geosense-platform/docs/architecture/01_context_diagram.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig1)
    
    print("Generating container diagram...")
    fig2 = create_container_diagram()
    fig2.savefig('/home/claude/geosense-platform/docs/architecture/02_container_diagram.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig2)
    
    print("Generating component diagram...")
    fig3 = create_component_diagram()
    fig3.savefig('/home/claude/geosense-platform/docs/architecture/03_component_diagram.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig3)
    
    print("✅ All diagrams generated successfully!")
    print(f"   Saved to: /home/claude/geosense-platform/docs/architecture/")


if __name__ == '__main__':
    save_all_diagrams()
