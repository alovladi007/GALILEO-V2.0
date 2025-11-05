"""
FastAPI application for GeoSense Platform.

Provides REST API endpoints for:
- Orbit propagation
- Formation control simulation
- Laser interferometry analysis
- ML model predictions
- Platform status and documentation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional
import os

# Import GeoSense modules (optional - gracefully degrade if not available)
try:
    import jax.numpy as jnp
    import numpy as np
    from sim.dynamics import (
        two_body_dynamics,
        propagate_orbit_jax,
        orbital_elements_to_cartesian,
        propagate_relative_orbit,
    )
    from sensing.phase_model import compute_phase, range_to_phase
    from sensing.noise import total_phase_noise_std
    from sensing.allan import allan_deviation
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    import_error_message = str(e)

app = FastAPI(
    title="GeoSense Platform API",
    description="AI-enhanced space-based geophysical sensing platform",
    version="0.4.0"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Request/Response Models
# ============================================================================

class OrbitalElements(BaseModel):
    """Orbital elements (a, e, i, Ω, ω, ν) in km and degrees."""
    semi_major_axis: float  # km
    eccentricity: float
    inclination: float  # degrees
    raan: float  # degrees (Ω)
    argument_of_perigee: float  # degrees (ω)
    true_anomaly: float  # degrees (ν)

class PropagationRequest(BaseModel):
    """Request for orbit propagation."""
    orbital_elements: OrbitalElements
    duration: float  # seconds
    time_step: float = 10.0  # seconds

class FormationRequest(BaseModel):
    """Request for formation flying simulation."""
    delta_state: List[float]  # [x, y, z, vx, vy, vz] in km and km/s
    mean_motion: float  # rad/s
    duration: float  # seconds
    time_step: float = 10.0  # seconds

class PhaseRequest(BaseModel):
    """Request for phase measurement calculation."""
    range_km: float
    wavelength: float = 1064e-9  # meters (default Nd:YAG)

class NoiseRequest(BaseModel):
    """Request for noise budget calculation."""
    power: float  # Watts
    range_km: float
    range_rate_km_s: float
    frequency_stability: float = 1e-13

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/")
async def root():
    """Redirect to API documentation."""
    return {
        "message": "GeoSense Platform API",
        "version": "0.4.0",
        "documentation": "/docs",
        "ui": "http://localhost:3000 (Next.js UI - run 'cd ui && npm install && npm run dev')",
        "health": "/health",
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.4.0",
        "modules": {
            "imports_available": IMPORTS_AVAILABLE,
            "jax": True,
            "numpy": True,
        },
        "capabilities": [
            "orbital_dynamics",
            "formation_flying",
            "laser_interferometry",
            "gnc_systems",
            "machine_learning",
            "gravity_inversion"
        ]
    }

@app.post("/api/propagate")
async def propagate_orbit(request: PropagationRequest):
    """
    Propagate an orbit from orbital elements.

    Returns trajectory as list of [x, y, z, vx, vy, vz] states in km and km/s.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Simulation modules not available")

    try:
        # Use simulation service
        from api.services import get_simulation_service
        service = get_simulation_service()

        # Prepare elements dict
        elements = {
            'semi_major_axis': request.orbital_elements.semi_major_axis,
            'eccentricity': request.orbital_elements.eccentricity,
            'inclination': request.orbital_elements.inclination,
            'raan': request.orbital_elements.raan,
            'argument_of_perigee': request.orbital_elements.argument_of_perigee,
            'true_anomaly': request.orbital_elements.true_anomaly,
        }

        # Propagate orbit
        result = service.propagate_from_elements(
            elements,
            duration=request.duration,
            time_step=request.time_step,
            include_perturbations=False
        )

        # Add orbital parameters
        params = service.compute_orbital_parameters(elements)

        return {
            **result,
            "orbital_parameters": params,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Propagation failed: {str(e)}")

@app.post("/api/formation")
async def propagate_formation(request: FormationRequest):
    """
    Simulate formation flying using Hill-Clohessy-Wiltshire equations.

    Returns relative trajectory as list of [x, y, z, vx, vy, vz] states.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Simulation modules not available")

    try:
        # Use simulation service
        from api.services import get_simulation_service
        service = get_simulation_service()

        # For now, use simple formation propagation
        # TODO: Support full chief/deputy simulation from request
        delta_state = request.delta_state

        # Create a basic LEO orbit for chief if not provided
        chief_elements = {
            'semi_major_axis': 6900.0,  # 500 km altitude LEO
            'eccentricity': 0.001,
            'inclination': 98.0,  # Sun-synchronous
            'raan': 0.0,
            'argument_of_perigee': 0.0,
            'true_anomaly': 0.0,
        }

        # Simulate formation
        result = service.simulate_formation(
            chief_elements,
            delta_state,
            duration=request.duration,
            time_step=request.time_step
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Formation simulation failed: {str(e)}")

@app.post("/api/phase")
async def calculate_phase(request: PhaseRequest):
    """Calculate phase measurement from range."""
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sensing modules not available")

    try:
        phase_rad = compute_phase(request.range_km, request.wavelength)
        phase_cycles = range_to_phase(request.range_km, request.wavelength)

        return {
            "range_km": request.range_km,
            "wavelength_m": request.wavelength,
            "phase_rad": float(phase_rad),
            "phase_cycles": float(phase_cycles),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Phase calculation failed: {str(e)}")

@app.post("/api/noise")
async def calculate_noise_budget(request: NoiseRequest):
    """Calculate comprehensive noise budget for laser interferometry."""
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Sensing modules not available")

    try:
        total_noise, breakdown = total_phase_noise_std(
            power=request.power,
            range_km=request.range_km,
            range_rate_km_s=request.range_rate_km_s,
            frequency_stability=request.frequency_stability,
        )

        # Convert JAX arrays to Python floats
        breakdown_dict = {k: float(v) for k, v in breakdown.items()}

        return {
            "total_noise_rad": float(total_noise),
            "breakdown": breakdown_dict,
            "range_km": request.range_km,
            "power_W": request.power,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Noise calculation failed: {str(e)}")

# ============================================================================
# Inversion Endpoints
# ============================================================================

class TikhonovInversionRequest(BaseModel):
    """Request for Tikhonov regularized inversion."""
    forward_matrix: List[List[float]]  # n_data x n_model
    data: List[float]  # n_data
    lambda_reg: float
    regularization_type: str = 'smoothness'  # 'smoothness' or 'identity'
    regularization_order: int = 2

class LCurveRequest(BaseModel):
    """Request for L-curve analysis."""
    forward_matrix: List[List[float]]
    data: List[float]
    lambda_range: Optional[List[float]] = None

class GravityAnomalyRequest(BaseModel):
    """Request for gravity anomaly computation."""
    latitude: List[float]
    longitude: List[float]
    observed_gravity: List[float]  # mGal
    model_name: str = 'EGM96'
    correction_type: str = 'free_air'  # 'free_air', 'bouguer', 'isostatic'
    elevation: Optional[List[float]] = None  # meters

class JointInversionSetupRequest(BaseModel):
    """Request for joint inversion setup."""
    gravity_data: List[float]
    latitude: List[float]
    longitude: List[float]
    depth: Optional[List[float]] = None
    model_name: str = 'joint_model'

@app.post("/api/inversion/tikhonov")
async def solve_tikhonov_inversion(request: TikhonovInversionRequest):
    """
    Solve linear inverse problem using Tikhonov regularization.

    Returns model estimate with resolution diagnostics.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Inversion modules not available")

    try:
        from api.services import get_inversion_service
        import numpy as np

        service = get_inversion_service()

        # Convert to numpy arrays
        forward_matrix = np.array(request.forward_matrix)
        data = np.array(request.data)

        # Solve inversion
        result = service.solve_tikhonov(
            forward_matrix,
            data,
            request.lambda_reg,
            regularization_type=request.regularization_type,
            regularization_order=request.regularization_order
        )

        return result.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tikhonov inversion failed: {str(e)}")

@app.post("/api/inversion/l-curve")
async def compute_l_curve(request: LCurveRequest):
    """
    Compute L-curve for regularization parameter selection.

    Returns optimal lambda and curve data for visualization.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Inversion modules not available")

    try:
        from api.services import get_inversion_service
        import numpy as np

        service = get_inversion_service()

        # Convert to numpy arrays
        forward_matrix = np.array(request.forward_matrix)
        data = np.array(request.data)
        lambda_range = np.array(request.lambda_range) if request.lambda_range else None

        # Compute L-curve
        result = service.compute_l_curve(forward_matrix, data, lambda_range)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"L-curve computation failed: {str(e)}")

@app.get("/api/inversion/gravity-model/{model_name}")
async def load_gravity_model(model_name: str):
    """
    Load reference gravity field model (EGM96 or EGM2008).

    Returns model metadata and specifications.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Inversion modules not available")

    try:
        from api.services import get_inversion_service

        service = get_inversion_service()

        # Load model
        result = service.load_gravity_model(model_name)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gravity model loading failed: {str(e)}")

@app.post("/api/inversion/gravity-anomaly")
async def compute_gravity_anomaly(request: GravityAnomalyRequest):
    """
    Compute gravity anomaly relative to reference model.

    Supports free-air, Bouguer, and isostatic corrections.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Inversion modules not available")

    try:
        from api.services import get_inversion_service
        import numpy as np

        service = get_inversion_service()

        # Convert to numpy arrays
        lat = np.array(request.latitude)
        lon = np.array(request.longitude)
        observed_gravity = np.array(request.observed_gravity)
        elevation = np.array(request.elevation) if request.elevation else None

        # Compute anomaly
        result = service.compute_gravity_anomaly(
            lat, lon, observed_gravity,
            model_name=request.model_name,
            correction_type=request.correction_type,
            elevation=elevation
        )

        return result.to_dict()

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gravity anomaly computation failed: {str(e)}")

@app.post("/api/inversion/joint/setup")
async def setup_joint_inversion(request: JointInversionSetupRequest):
    """
    Set up joint inversion model with gravity data.

    Returns model ID for subsequent operations.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Inversion modules not available")

    try:
        from api.services import get_inversion_service
        import numpy as np

        service = get_inversion_service()

        # Convert to numpy arrays
        gravity_data = np.array(request.gravity_data)
        lat = np.array(request.latitude)
        lon = np.array(request.longitude)
        depth = np.array(request.depth) if request.depth else None

        # Setup joint inversion
        model_id = service.setup_joint_inversion_model(
            gravity_data, lat, lon, depth, request.model_name
        )

        return {
            "model_id": model_id,
            "status": "configured",
            "data_types": ["gravity"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Joint inversion setup failed: {str(e)}")

@app.post("/api/inversion/joint/{model_id}/run")
async def run_joint_inversion(
    model_id: str,
    max_iterations: int = 100,
    convergence_tol: float = 1e-6
):
    """
    Execute joint inversion for configured model.

    Returns inversion results including model, residuals, and convergence info.
    """
    if not IMPORTS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Inversion modules not available")

    try:
        from api.services import get_inversion_service

        service = get_inversion_service()

        # Run inversion
        results = service.run_joint_inversion(
            model_id,
            max_iterations=max_iterations,
            convergence_tol=convergence_tol
        )

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Joint inversion failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
