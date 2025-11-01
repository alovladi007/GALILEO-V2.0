//! Power management and energy budget tracking
//!
//! Monitors battery state, solar array generation, and subsystem power consumption

use chrono::{DateTime, Utc};

pub mod battery;
pub mod solar;
pub mod budget;

/// Power system state
#[derive(Debug, Clone)]
pub struct PowerState {
    /// Battery charge level (0.0 to 1.0)
    pub battery_soc: f64,
    /// Solar array power generation (W)
    pub solar_power: f64,
    /// Total system power consumption (W)
    pub load_power: f64,
    /// Battery voltage (V)
    pub battery_voltage: f64,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

impl PowerState {
    pub fn new(
        battery_soc: f64,
        solar_power: f64,
        load_power: f64,
        battery_voltage: f64,
        timestamp: DateTime<Utc>,
    ) -> Self {
        Self {
            battery_soc: battery_soc.clamp(0.0, 1.0),
            solar_power,
            load_power,
            battery_voltage,
            timestamp,
        }
    }

    /// Net power balance (W)
    pub fn net_power(&self) -> f64 {
        self.solar_power - self.load_power
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_state() {
        let state = PowerState::new(0.85, 120.0, 90.0, 28.5, Utc::now());
        assert_eq!(state.net_power(), 30.0);
        assert!(state.battery_soc >= 0.0 && state.battery_soc <= 1.0);
    }
}
