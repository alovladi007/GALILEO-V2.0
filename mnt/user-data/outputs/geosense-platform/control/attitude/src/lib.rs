//! Attitude determination and control system (ADCS)
//!
//! Manages spacecraft orientation for precision gravity measurements

use nalgebra::{UnitQuaternion, Vector3};
use chrono::{DateTime, Utc};

pub mod determination;
pub mod control;
pub mod actuators;

/// Spacecraft attitude state
#[derive(Debug, Clone)]
pub struct AttitudeState {
    /// Orientation quaternion (inertial to body frame)
    pub orientation: UnitQuaternion<f64>,
    /// Angular velocity (rad/s) in body frame
    pub angular_velocity: Vector3<f64>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl AttitudeState {
    pub fn new(
        orientation: UnitQuaternion<f64>,
        angular_velocity: Vector3<f64>,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            orientation,
            angular_velocity,
            timestamp,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attitude_creation() {
        let orientation = UnitQuaternion::identity();
        let angular_vel = Vector3::zeros();
        let state = AttitudeState::new(orientation, angular_vel, Utc::now());
        assert!(state.orientation.is_identity(1e-6));
    }
}
