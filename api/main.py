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
        # Convert degrees to radians for angles
        oe = jnp.array([
            request.orbital_elements.semi_major_axis,
            request.orbital_elements.eccentricity,
            jnp.deg2rad(request.orbital_elements.inclination),
            jnp.deg2rad(request.orbital_elements.raan),
            jnp.deg2rad(request.orbital_elements.argument_of_perigee),
            jnp.deg2rad(request.orbital_elements.true_anomaly),
        ])

        # Convert to Cartesian state
        state0 = orbital_elements_to_cartesian(oe)

        # Propagate
        times, states = propagate_orbit_jax(
            two_body_dynamics,
            state0,
            t_span=(0.0, request.duration),
            dt=request.time_step
        )

        return {
            "times": times.tolist(),
            "states": states.tolist(),
            "num_points": len(times),
            "duration": request.duration,
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
        delta_state = jnp.array(request.delta_state)

        times, rel_states = propagate_relative_orbit(
            delta_state,
            request.mean_motion,
            t_span=(0.0, request.duration),
            dt=request.time_step
        )

        return {
            "times": times.tolist(),
            "states": rel_states.tolist(),
            "num_points": len(times),
            "duration": request.duration,
        }
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
