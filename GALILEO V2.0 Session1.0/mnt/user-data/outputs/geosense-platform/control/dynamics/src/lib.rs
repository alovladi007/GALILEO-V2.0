//! Orbital dynamics and propagation for geophysical sensing satellites
//!
//! This module implements high-fidelity orbit propagation including:
//! - Two-body Keplerian dynamics
//! - J2-J6 Earth oblateness perturbations
//! - Third-body gravitational effects (Sun, Moon)
//! - Atmospheric drag modeling
//! - Solar radiation pressure
//! - Relativistic corrections

use nalgebra::{Vector3, Matrix3};
use chrono::{DateTime, Utc};
use std::f64::consts::PI;

pub mod propagator;
pub mod perturbations;
pub mod ephemeris;

/// Gravitational parameter for Earth (km³/s²)
pub const MU_EARTH: f64 = 398600.4418;

/// Mean Earth radius (km)
pub const R_EARTH: f64 = 6378.137;

/// Orbital state vector in Earth-Centered Inertial (ECI) frame
#[derive(Debug, Clone)]
pub struct OrbitalState {
    /// Position vector (km)
    pub position: Vector3<f64>,
    /// Velocity vector (km/s)
    pub velocity: Vector3<f64>,
    /// Epoch time
    pub epoch: DateTime<Utc>,
}

impl OrbitalState {
    /// Create new orbital state
    pub fn new(position: Vector3<f64>, velocity: Vector3<f64>, epoch: DateTime<Utc>) -> Self {
        Self { position, velocity, epoch }
    }

    /// Calculate orbital energy (km²/s²)
    pub fn energy(&self) -> f64 {
        let v_mag = self.velocity.norm();
        let r_mag = self.position.norm();
        0.5 * v_mag * v_mag - MU_EARTH / r_mag
    }

    /// Calculate angular momentum vector (km²/s)
    pub fn angular_momentum(&self) -> Vector3<f64> {
        self.position.cross(&self.velocity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orbital_state_creation() {
        let pos = Vector3::new(7000.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 7.5, 0.0);
        let epoch = Utc::now();
        
        let state = OrbitalState::new(pos, vel, epoch);
        assert_eq!(state.position.norm(), 7000.0);
    }

    #[test]
    fn test_energy_calculation() {
        let pos = Vector3::new(7000.0, 0.0, 0.0);
        let vel = Vector3::new(0.0, 7.5, 0.0);
        let state = OrbitalState::new(pos, vel, Utc::now());
        
        let energy = state.energy();
        assert!(energy < 0.0, "Bound orbit should have negative energy");
    }
}
