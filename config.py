"""
GALILEO V2.0 - Unified Configuration Management
================================================

Centralized configuration for all platform components.
"""

import os
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, field
import json

# ============================================================================
# Base Paths
# ============================================================================

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, OUTPUTS_DIR, CHECKPOINTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class APIConfig:
    """API Server Configuration"""
    host: str = "0.0.0.0"
    port: int = 5050
    reload: bool = False  # Disable auto-reload to avoid file watching issues
    workers: int = 1
    log_level: str = "info"
    cors_origins: list = field(default_factory=lambda: ["*"])

@dataclass
class DatabaseConfig:
    """Database Configuration (if needed)"""
    type: str = "sqlite"
    path: str = str(DATA_DIR / "galileo.db")
    echo: bool = False

@dataclass
class SimulationConfig:
    """Simulation Parameters"""
    default_timestep: float = 10.0  # seconds
    max_duration: float = 86400.0  # 1 day
    propagator: str = "rk4"  # rk4, rk45, or dopri5
    mu_earth: float = 398600.4418  # km^3/s^2

@dataclass
class SensingConfig:
    """Sensing System Parameters"""
    default_wavelength: float = 1064e-9  # meters (Nd:YAG)
    default_power: float = 1.0  # Watts
    frequency_stability: float = 1e-13

@dataclass
class ControlConfig:
    """GNC System Parameters"""
    controller_type: str = "lqr"  # lqr, lqg, or mpc
    update_rate: float = 1.0  # Hz
    horizon: int = 20  # MPC horizon

@dataclass
class MLConfig:
    """Machine Learning Parameters"""
    default_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    device: str = "cpu"  # cpu or cuda
    checkpoint_dir: str = str(CHECKPOINTS_DIR)

@dataclass
class GeophysicsConfig:
    """Geophysics Processing Parameters"""
    gravity_model: str = "egm2008"  # egm96, egm2008
    crustal_model: str = "crust1.0"
    resolution: int = 180  # degrees
    regularization: float = 1e-3

@dataclass
class EmulatorConfig:
    """Laboratory Emulator Parameters"""
    baseline_length: float = 1.0  # meters
    wavelength: float = 632.8e-9  # He-Ne laser
    sampling_rate: float = 1000.0  # Hz
    websocket_port: int = 8765
    dashboard_port: int = 8080

@dataclass
class ComplianceConfig:
    """Security & Compliance Parameters"""
    rbac_enabled: bool = True
    audit_logging: bool = True
    secrets_encryption: bool = True
    data_retention_days: int = 2555  # 7 years

@dataclass
class GalileoConfig:
    """Master Configuration"""
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    sensing: SensingConfig = field(default_factory=SensingConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    geophysics: GeophysicsConfig = field(default_factory=GeophysicsConfig)
    emulator: EmulatorConfig = field(default_factory=EmulatorConfig)
    compliance: ComplianceConfig = field(default_factory=ComplianceConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "api": self.api.__dict__,
            "database": self.database.__dict__,
            "simulation": self.simulation.__dict__,
            "sensing": self.sensing.__dict__,
            "control": self.control.__dict__,
            "ml": self.ml.__dict__,
            "geophysics": self.geophysics.__dict__,
            "emulator": self.emulator.__dict__,
            "compliance": self.compliance.__dict__,
        }

    def save(self, path: Path):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'GalileoConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(
            api=APIConfig(**data.get('api', {})),
            database=DatabaseConfig(**data.get('database', {})),
            simulation=SimulationConfig(**data.get('simulation', {})),
            sensing=SensingConfig(**data.get('sensing', {})),
            control=ControlConfig(**data.get('control', {})),
            ml=MLConfig(**data.get('ml', {})),
            geophysics=GeophysicsConfig(**data.get('geophysics', {})),
            emulator=EmulatorConfig(**data.get('emulator', {})),
            compliance=ComplianceConfig(**data.get('compliance', {})),
        )

# ============================================================================
# Global Configuration Instance
# ============================================================================

# Load from file if exists, otherwise use defaults
CONFIG_PATH = BASE_DIR / "galileo_config.json"

if CONFIG_PATH.exists():
    try:
        config = GalileoConfig.load(CONFIG_PATH)
        print(f"✓ Loaded configuration from {CONFIG_PATH}")
    except Exception as e:
        print(f"⚠ Failed to load config: {e}. Using defaults.")
        config = GalileoConfig()
else:
    config = GalileoConfig()
    # Save default configuration
    try:
        config.save(CONFIG_PATH)
        print(f"✓ Created default configuration at {CONFIG_PATH}")
    except Exception as e:
        print(f"⚠ Failed to save default config: {e}")

# ============================================================================
# Environment Variables (override config)
# ============================================================================

# API
if os.getenv("GALILEO_API_PORT"):
    config.api.port = int(os.getenv("GALILEO_API_PORT"))

if os.getenv("GALILEO_API_HOST"):
    config.api.host = os.getenv("GALILEO_API_HOST")

# Database
if os.getenv("GALILEO_DB_PATH"):
    config.database.path = os.getenv("GALILEO_DB_PATH")

# ML
if os.getenv("GALILEO_ML_DEVICE"):
    config.ml.device = os.getenv("GALILEO_ML_DEVICE")

# ============================================================================
# Utility Functions
# ============================================================================

def get_config() -> GalileoConfig:
    """Get global configuration instance."""
    return config

def update_config(updates: Dict[str, Any]):
    """Update configuration values."""
    global config
    for key, value in updates.items():
        if hasattr(config, key):
            setattr(config, key, value)

def print_config():
    """Print current configuration."""
    print("=" * 70)
    print("GALILEO V2.0 Configuration")
    print("=" * 70)
    for key, value in config.to_dict().items():
        print(f"\n{key.upper()}:")
        for k, v in value.items():
            print(f"  {k}: {v}")
    print("=" * 70)

# ============================================================================
# Module Export
# ============================================================================

__all__ = [
    'config',
    'get_config',
    'update_config',
    'print_config',
    'GalileoConfig',
    'APIConfig',
    'DatabaseConfig',
    'SimulationConfig',
    'SensingConfig',
    'ControlConfig',
    'MLConfig',
    'GeophysicsConfig',
    'EmulatorConfig',
    'ComplianceConfig',
    'BASE_DIR',
    'DATA_DIR',
    'OUTPUTS_DIR',
    'CHECKPOINTS_DIR',
    'LOGS_DIR',
]
