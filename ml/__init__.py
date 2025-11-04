"""
Machine Learning Module for GeoSense Platform
Session 3: Intelligent Satellite Operations

This module provides ML-enhanced capabilities for:
- Neural orbit prediction
- Anomaly detection
- Reinforcement learning control
- Multi-agent coordination
- Real-time inference
"""

from ml.models import (
    OrbitPredictor,
    AnomalyDetector,
    FormationOptimizer,
    NeuralStateEstimator,
    MLConfig,
    create_orbit_predictor,
    create_anomaly_detector,
    create_formation_optimizer,
    create_neural_estimator,
)

from ml.reinforcement import (
    PPOAgent,
    SACAgent,
    MultiAgentCoordinator,
    RLState,
    RLConfig,
    create_rl_trainer,
    compute_gae,
    formation_reward,
)

from ml.training import (
    DataGenerator,
    SupervisedTrainer,
    TrainingConfig,
    ModelCheckpoint,
    TransferLearning,
    train_orbit_predictor,
    train_anomaly_detector,
)

from ml.inference import (
    ModelOptimizer,
    InferenceEngine,
    RealtimePredictor,
    InferenceConfig,
    EdgeDeployment,
    MLEnhancedController,
)

__all__ = [
    'OrbitPredictor', 'AnomalyDetector', 'FormationOptimizer', 'NeuralStateEstimator',
    'MLConfig', 'PPOAgent', 'SACAgent', 'MultiAgentCoordinator',
    'RLState', 'RLConfig', 'DataGenerator', 'SupervisedTrainer',
    'TrainingConfig', 'ModelCheckpoint', 'TransferLearning',
    'ModelOptimizer', 'InferenceEngine', 'RealtimePredictor',
    'InferenceConfig', 'EdgeDeployment', 'MLEnhancedController',
]

__version__ = '0.4.0'
__session__ = 'Session 3: Machine Learning Integration'
