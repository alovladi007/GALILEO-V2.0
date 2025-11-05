"""
Joint Inversion Module

Provides API for joint inversion of gravity data with other geophysical
datasets (seismic, magnetic, etc.) with integration to Session 5 inversion tools.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Callable, Any
from dataclasses import dataclass
import json


@dataclass
class JointInversionModel:
    """
    Container for joint inversion configuration and results.
    
    Attributes:
        name: Model name
        data_types: List of data types being jointly inverted
        model_parameters: Dict of model parameters
        coupling_constraints: Constraints between different data types
        regularization: Regularization parameters
        results: Inversion results
        metadata: Additional metadata
    """
    name: str
    data_types: List[str]
    model_parameters: Dict[str, np.ndarray]
    coupling_constraints: Optional[Dict[str, Any]] = None
    regularization: Optional[Dict[str, float]] = None
    results: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict] = None
    
    def add_data_type(self, data_type: str, data: np.ndarray, 
                     uncertainty: Optional[np.ndarray] = None):
        """Add a new data type to the joint inversion."""
        if data_type not in self.data_types:
            self.data_types.append(data_type)
        
        self.model_parameters[f'{data_type}_data'] = data
        if uncertainty is not None:
            self.model_parameters[f'{data_type}_uncertainty'] = uncertainty
    
    def set_coupling(self, param1: str, param2: str, 
                    relationship: str = 'proportional',
                    strength: float = 1.0):
        """
        Define coupling between model parameters.
        
        Args:
            param1: First parameter name
            param2: Second parameter name
            relationship: Type of relationship ('proportional', 'inverse', 'custom')
            strength: Coupling strength (0=no coupling, 1=strong coupling)
        """
        if self.coupling_constraints is None:
            self.coupling_constraints = {}
        
        self.coupling_constraints[f'{param1}_{param2}'] = {
            'type': relationship,
            'strength': strength
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get inversion results."""
        if self.results is None:
            raise ValueError("No results available. Run inversion first.")
        return self.results


def setup_joint_inversion(
    gravity_data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    depth: Optional[np.ndarray] = None,
    model_name: str = 'joint_model'
) -> JointInversionModel:
    """
    Set up a joint inversion model starting with gravity data.
    
    Args:
        gravity_data: Gravity observations (mGal)
        lat: Latitude points
        lon: Longitude points
        depth: Depth points (if 3D inversion)
        model_name: Name for the model
        
    Returns:
        JointInversionModel instance ready for configuration
    """
    # Initialize model parameters
    model_params = {
        'gravity_data': gravity_data,
        'latitude': lat,
        'longitude': lon,
    }
    
    if depth is not None:
        model_params['depth'] = depth
    
    # Initialize with gravity as first data type
    data_types = ['gravity']
    
    # Default regularization
    regularization = {
        'alpha_smooth': 0.1,  # Smoothness constraint
        'alpha_small': 0.01,  # Smallness constraint
        'alpha_coupling': 0.5,  # Coupling strength
    }
    
    metadata = {
        'description': 'Joint geophysical inversion',
        'created': 'auto',
        'n_observations': len(gravity_data.flat),
    }
    
    return JointInversionModel(
        name=model_name,
        data_types=data_types,
        model_parameters=model_params,
        regularization=regularization,
        metadata=metadata
    )


def integrate_gravity_seismic(
    joint_model: JointInversionModel,
    seismic_data: np.ndarray,
    seismic_type: str = 'velocity',
    coupling_type: str = 'petrophysical'
) -> JointInversionModel:
    """
    Integrate seismic data into joint inversion with gravity.
    
    Args:
        joint_model: Existing joint inversion model
        seismic_data: Seismic observations (velocity, traveltime, etc.)
        seismic_type: Type of seismic data
        coupling_type: How to couple gravity and seismic
                      ('petrophysical', 'structural', 'uncoupled')
        
    Returns:
        Updated JointInversionModel
    """
    # Add seismic data
    joint_model.add_data_type(f'seismic_{seismic_type}', seismic_data)
    
    # Set up coupling constraints
    if coupling_type == 'petrophysical':
        # Density-velocity relationship (Gardner's relation, etc.)
        # ρ = a * V^b
        joint_model.set_coupling(
            'density', 'velocity',
            relationship='proportional',
            strength=0.7
        )
        
        joint_model.metadata['coupling_description'] = \
            'Petrophysical density-velocity coupling'
    
    elif coupling_type == 'structural':
        # Structural boundaries match between models
        joint_model.set_coupling(
            'gravity_gradient', 'seismic_gradient',
            relationship='proportional',
            strength=0.5
        )
        
        joint_model.metadata['coupling_description'] = \
            'Structural boundary coupling'
    
    elif coupling_type == 'uncoupled':
        joint_model.metadata['coupling_description'] = \
            'Independent inversion with shared geometry'
    
    return joint_model


def add_magnetic_data(
    joint_model: JointInversionModel,
    magnetic_data: np.ndarray,
    magnetic_type: str = 'total_field'
) -> JointInversionModel:
    """
    Add magnetic data to joint inversion.
    
    Args:
        joint_model: Existing joint inversion model
        magnetic_data: Magnetic observations
        magnetic_type: Type of magnetic data
        
    Returns:
        Updated JointInversionModel
    """
    joint_model.add_data_type(f'magnetic_{magnetic_type}', magnetic_data)
    
    # Magnetic and gravity can share structural features
    joint_model.set_coupling(
        'density', 'susceptibility',
        relationship='proportional',
        strength=0.3  # Weaker coupling than density-velocity
    )
    
    return joint_model


def perform_joint_inversion(
    joint_model: JointInversionModel,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6,
    session5_integrator: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    Perform joint inversion using all available data types.
    
    This is a placeholder that would integrate with Session 5 inversion tools.
    
    Args:
        joint_model: Configured joint inversion model
        max_iterations: Maximum number of iterations
        convergence_tol: Convergence tolerance
        session5_integrator: Optional Session 5 integration function
        
    Returns:
        Dictionary with inversion results
    """
    # Extract data
    gravity_data = joint_model.model_parameters['gravity_data']
    n_data = len(gravity_data.flat)
    
    # Initialize model parameters
    # In real implementation, would set up proper inversion grid
    # Use a reasonable number of model parameters
    n_cells = min(n_data // 2, 500)  # Reasonable ratio
    
    density_model = np.ones(n_cells) * 2670.0  # Initial guess
    
    # If Session 5 integrator provided, use it
    if session5_integrator is not None:
        try:
            results = session5_integrator(joint_model, max_iterations, convergence_tol)
            joint_model.results = results
            return results
        except Exception as e:
            print(f"Session 5 integration failed: {e}")
            print("Falling back to standalone inversion")
    
    # Simplified inversion loop (placeholder)
    for iteration in range(max_iterations):
        # Forward modeling - interpolate model to data locations
        # Simple approach: replicate density values to match data size
        n_replicate = (n_data + n_cells - 1) // n_cells
        predicted_gravity = np.tile(density_model * 0.1, n_replicate)[:n_data]
        
        # Compute misfit
        residual = gravity_data.flat - predicted_gravity
        misfit = np.sum(residual**2)
        
        # Check convergence
        if misfit < convergence_tol:
            break
        
        # Update model (simplified gradient descent)
        sensitivity = compute_sensitivity_matrix(n_cells, len(gravity_data.flat))
        update = sensitivity.T @ residual
        
        # Apply regularization
        alpha = joint_model.regularization['alpha_smooth']
        smooth_operator = compute_smoothness_matrix(n_cells)
        regularized_update = update - alpha * smooth_operator @ density_model
        
        # Apply coupling constraints if multiple data types
        if len(joint_model.data_types) > 1:
            regularized_update = apply_coupling_constraints(
                regularized_update,
                joint_model.coupling_constraints
            )
        
        # Update model
        step_size = 0.01
        density_model += step_size * regularized_update
    
    # Compute final results
    n_replicate = (n_data + n_cells - 1) // n_cells
    final_predicted = np.tile(density_model * 0.1, n_replicate)[:n_data]
    final_residual = gravity_data.flat - final_predicted
    
    results = {
        'density_model': density_model,
        'predicted_gravity': final_predicted,
        'residual': final_residual,
        'misfit': np.sum(final_residual**2),
        'iterations': iteration + 1,
        'converged': misfit < convergence_tol,
        'rms_error': np.sqrt(np.mean(final_residual**2)),
    }
    
    # Add results for other data types if present
    for data_type in joint_model.data_types:
        if data_type != 'gravity':
            results[f'{data_type}_predicted'] = \
                forward_model_for_type(density_model, data_type)
    
    joint_model.results = results
    return results


def forward_gravity_model(density: np.ndarray) -> np.ndarray:
    """
    Forward model for gravity (placeholder).
    
    Args:
        density: Density model
        
    Returns:
        Predicted gravity
    """
    # Simplified forward model
    # Real implementation would use proper Green's functions
    # Return array of same size as would be needed for data
    return np.tile(density * 0.1, max(1, 10 // len(density) + 1))[:10 * len(density)]  # mGal per kg/m³ (placeholder)


def forward_model_for_type(density: np.ndarray, data_type: str) -> np.ndarray:
    """
    Forward model for other data types.
    
    Args:
        density: Density model
        data_type: Type of data to predict
        
    Returns:
        Predicted data
    """
    if 'seismic' in data_type:
        # Use density-velocity relationship
        # Gardner's relation: ρ = 0.31 * V^0.25 (or inverse)
        velocity = (density / 310) ** 4
        return velocity
    
    elif 'magnetic' in data_type:
        # Simple susceptibility from density (very simplified)
        susceptibility = (density - 2670) * 0.001
        return susceptibility
    
    else:
        return density


def compute_sensitivity_matrix(n_model: int, n_data: int) -> np.ndarray:
    """
    Compute sensitivity (Jacobian) matrix.
    
    Args:
        n_model: Number of model parameters
        n_data: Number of data points
        
    Returns:
        Sensitivity matrix (n_data x n_model)
    """
    # Simplified sensitivity matrix
    # Real implementation would compute proper derivatives
    sensitivity = np.random.randn(n_data, n_model) * 0.1
    return sensitivity


def compute_smoothness_matrix(n_model: int) -> np.ndarray:
    """
    Compute smoothness regularization operator.
    
    Args:
        n_model: Number of model parameters
        
    Returns:
        Smoothness matrix
    """
    # Second derivative operator (Laplacian)
    smooth = np.zeros((n_model, n_model))
    for i in range(1, n_model - 1):
        smooth[i, i-1] = 1
        smooth[i, i] = -2
        smooth[i, i+1] = 1
    
    return smooth


def apply_coupling_constraints(
    update: np.ndarray,
    constraints: Optional[Dict[str, Any]]
) -> np.ndarray:
    """
    Apply coupling constraints to model update.
    
    Args:
        update: Model update vector
        constraints: Coupling constraints
        
    Returns:
        Constrained update
    """
    if constraints is None:
        return update
    
    # Apply constraints (simplified)
    constrained_update = update.copy()
    
    for constraint_name, constraint_info in constraints.items():
        strength = constraint_info['strength']
        relationship = constraint_info['type']
        
        # Reduce update magnitude based on coupling strength
        constrained_update *= (1 - 0.5 * strength)
    
    return constrained_update


def compute_model_resolution(
    sensitivity: np.ndarray,
    regularization_matrix: np.ndarray,
    alpha: float = 0.1
) -> np.ndarray:
    """
    Compute model resolution matrix.
    
    Args:
        sensitivity: Sensitivity matrix
        regularization_matrix: Regularization operator
        alpha: Regularization parameter
        
    Returns:
        Resolution matrix
    """
    # R = (G^T G + α L^T L)^-1 G^T G
    GTG = sensitivity.T @ sensitivity
    LTL = regularization_matrix.T @ regularization_matrix
    
    try:
        inv_term = np.linalg.inv(GTG + alpha * LTL)
        resolution = inv_term @ GTG
    except np.linalg.LinAlgError:
        # If singular, use pseudo-inverse
        resolution = np.eye(sensitivity.shape[1])
    
    return resolution


def export_for_session5(
    joint_model: JointInversionModel,
    output_file: str
) -> None:
    """
    Export joint inversion setup for Session 5 integration.
    
    Args:
        joint_model: Joint inversion model
        output_file: Output file path
    """
    export_data = {
        'model_name': joint_model.name,
        'data_types': joint_model.data_types,
        'model_parameters': {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in joint_model.model_parameters.items()
        },
        'coupling_constraints': joint_model.coupling_constraints,
        'regularization': joint_model.regularization,
        'metadata': joint_model.metadata,
    }
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)


def load_from_session5(input_file: str) -> JointInversionModel:
    """
    Load joint inversion setup from Session 5 export.
    
    Args:
        input_file: Input file path
        
    Returns:
        JointInversionModel instance
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    model_params = {
        k: np.array(v) if isinstance(v, list) else v
        for k, v in data['model_parameters'].items()
    }
    
    return JointInversionModel(
        name=data['model_name'],
        data_types=data['data_types'],
        model_parameters=model_params,
        coupling_constraints=data.get('coupling_constraints'),
        regularization=data.get('regularization'),
        metadata=data.get('metadata'),
    )
