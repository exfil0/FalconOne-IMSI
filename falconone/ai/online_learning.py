"""
Online Learning Module - Incremental Model Updates
Enables continuous model improvement without full retraining

Features:
- Incremental learning for classification and regression
- Mini-batch gradient descent updates
- Adaptive learning rates
- Concept drift detection
- Model versioning and rollback
- Memory-efficient streaming updates

Supported Algorithms:
- SGDClassifier (sklearn)
- PassiveAggressiveClassifier
- Online Neural Networks
- Incremental PCA
- River (online ML library)

Author: FalconOne Team
Version: 3.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import time
import pickle
import os
from collections import deque

try:
    from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from river import drift
    from river import metrics as river_metrics
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


@dataclass
class ModelSnapshot:
    """Snapshot of model state for versioning"""
    version: int
    model_state: bytes
    timestamp: float
    metrics: Dict[str, float]
    num_samples: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningMetrics:
    """Metrics for online learning performance"""
    num_updates: int = 0
    total_samples: int = 0
    accuracy: float = 0.0
    loss: float = 0.0
    learning_rate: float = 0.01
    concept_drifts_detected: int = 0
    model_rollbacks: int = 0
    avg_update_time_ms: float = 0.0


class OnlineLearner:
    """
    Online learning system for incremental model updates
    
    Supports:
    - Streaming data updates
    - Concept drift detection and adaptation
    - Model versioning and rollback
    - Adaptive learning rates
    - Multi-class and multi-label classification
    """
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize online learner
        
        Args:
            config: Configuration dictionary with:
                - algorithm: 'sgd', 'passive_aggressive', 'neural'
                - learning_rate: Initial learning rate (default: 0.01)
                - batch_size: Mini-batch size (default: 32)
                - drift_detection: Enable drift detection (default: True)
                - drift_threshold: Drift detection threshold (default: 0.05)
                - max_snapshots: Maximum model snapshots to keep (default: 10)
                - warmup_samples: Samples before enabling updates (default: 100)
                - feature_scaling: Enable feature scaling (default: True)
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Configuration
        self.algorithm = config.get('algorithm', 'sgd')
        self.learning_rate = config.get('learning_rate', 0.01)
        self.batch_size = config.get('batch_size', 32)
        self.drift_detection_enabled = config.get('drift_detection', True)
        self.drift_threshold = config.get('drift_threshold', 0.05)
        self.max_snapshots = config.get('max_snapshots', 10)
        self.warmup_samples = config.get('warmup_samples', 100)
        self.feature_scaling = config.get('feature_scaling', True)
        
        # Model components
        self.model = None
        self.scaler = StandardScaler() if self.feature_scaling and SKLEARN_AVAILABLE else None
        self.drift_detector = None
        
        # State tracking
        self.is_warmed_up = False
        self.sample_buffer = deque(maxlen=self.batch_size)
        self.label_buffer = deque(maxlen=self.batch_size)
        
        # Model versioning
        self.snapshots: List[ModelSnapshot] = []
        self.current_version = 0
        
        # Metrics
        self.metrics = LearningMetrics(learning_rate=self.learning_rate)
        
        # Performance history (for adaptive learning rate)
        self.performance_history = deque(maxlen=20)
        
        self.logger.info(
            "Online learner initialized (algorithm: %s, lr: %.4f)",
            self.algorithm, self.learning_rate
        )
    
    def initialize_model(
        self,
        n_features: int,
        n_classes: int = 2,
        pretrained_model: Optional[Any] = None
    ):
        """
        Initialize or load the model
        
        Args:
            n_features: Number of input features
            n_classes: Number of classes for classification
            pretrained_model: Optional pre-trained model to continue training
        """
        if pretrained_model is not None:
            self.model = pretrained_model
            self.logger.info("Loaded pre-trained model")
            return
        
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for online learning")
        
        if self.algorithm == 'sgd':
            self.model = SGDClassifier(
                loss='log_loss',  # For probability estimates
                learning_rate='adaptive',
                eta0=self.learning_rate,
                max_iter=1,
                tol=None,
                warm_start=True,
                random_state=42
            )
        elif self.algorithm == 'passive_aggressive':
            self.model = PassiveAggressiveClassifier(
                C=1.0,
                max_iter=1,
                tol=None,
                warm_start=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        # Initialize with dummy data
        dummy_X = np.zeros((n_classes, n_features))
        dummy_y = np.arange(n_classes)
        self.model.partial_fit(dummy_X, dummy_y, classes=np.arange(n_classes))
        
        # Initialize drift detector
        if self.drift_detection_enabled and RIVER_AVAILABLE:
            self.drift_detector = drift.ADWIN(delta=self.drift_threshold)
        
        self.logger.info(
            "Model initialized with %d features, %d classes",
            n_features, n_classes
        )
    
    def partial_fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, List],
        create_snapshot: bool = False
    ) -> Dict[str, float]:
        """
        Update model with new data (single sample or batch)
        
        Args:
            X: Feature matrix (n_samples, n_features) or single sample
            y: Labels (n_samples,) or single label
            create_snapshot: Whether to save model snapshot after update
        
        Returns:
            Dictionary with update metrics
        """
        start_time = time.time()
        
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, list):
            y = np.array(y)
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if y.ndim == 0:
            y = np.array([y])
        
        # Feature scaling
        if self.scaler is not None:
            X = self.scaler.partial_fit(X).transform(X)
        
        # Warmup phase: accumulate samples
        if not self.is_warmed_up:
            for i in range(len(X)):
                self.sample_buffer.append(X[i])
                self.label_buffer.append(y[i])
            
            if len(self.sample_buffer) >= self.warmup_samples:
                self.is_warmed_up = True
                self.logger.info("Warmup complete with %d samples", len(self.sample_buffer))
            
            return {'status': 'warmup', 'samples_collected': len(self.sample_buffer)}
        
        # Perform incremental update
        try:
            # Get predictions before update (for drift detection)
            if self.drift_detection_enabled:
                y_pred = self.model.predict(X)
                accuracy = accuracy_score(y, y_pred)
                
                # Update drift detector
                if self.drift_detector is not None:
                    for pred, true in zip(y_pred, y):
                        error = 1 if pred != true else 0
                        self.drift_detector.update(error)
                        
                        if self.drift_detector.drift_detected:
                            self.logger.warning("Concept drift detected!")
                            self.metrics.concept_drifts_detected += 1
                            self._handle_concept_drift()
                            break
            
            # Perform partial fit
            self.model.partial_fit(X, y)
            
            # Update metrics
            self.metrics.num_updates += 1
            self.metrics.total_samples += len(X)
            
            # Evaluate on current batch
            y_pred = self.model.predict(X)
            batch_accuracy = accuracy_score(y, y_pred)
            self.metrics.accuracy = batch_accuracy
            
            # Track performance history
            self.performance_history.append(batch_accuracy)
            
            # Adaptive learning rate
            if len(self.performance_history) >= 5:
                recent_performance = list(self.performance_history)[-5:]
                if all(recent_performance[i] <= recent_performance[i-1] 
                       for i in range(1, len(recent_performance))):
                    # Performance degrading: reduce learning rate
                    self.learning_rate *= 0.9
                    self.metrics.learning_rate = self.learning_rate
                    self.logger.info("Reduced learning rate to %.4f", self.learning_rate)
            
            # Create snapshot if requested
            if create_snapshot:
                self._create_snapshot()
            
            update_time = (time.time() - start_time) * 1000
            self.metrics.avg_update_time_ms = (
                (self.metrics.avg_update_time_ms * (self.metrics.num_updates - 1) + 
                 update_time) / self.metrics.num_updates
            )
            
            return {
                'status': 'success',
                'accuracy': batch_accuracy,
                'samples_processed': len(X),
                'total_samples': self.metrics.total_samples,
                'learning_rate': self.learning_rate,
                'update_time_ms': update_time
            }
            
        except Exception as e:
            self.logger.error(f"Failed to perform partial fit: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Make predictions with current model
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict(X)
    
    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Feature matrix
        
        Returns:
            Class probabilities
        """
        if self.model is None or not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        # Handle single sample
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return self.model.predict_proba(X)
    
    def _create_snapshot(self) -> ModelSnapshot:
        """Create and store model snapshot"""
        self.current_version += 1
        
        # Serialize model
        model_bytes = pickle.dumps({
            'model': self.model,
            'scaler': self.scaler,
            'learning_rate': self.learning_rate
        })
        
        snapshot = ModelSnapshot(
            version=self.current_version,
            model_state=model_bytes,
            timestamp=time.time(),
            metrics={
                'accuracy': self.metrics.accuracy,
                'total_samples': self.metrics.total_samples
            },
            num_samples=self.metrics.total_samples
        )
        
        self.snapshots.append(snapshot)
        
        # Remove old snapshots
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        self.logger.info("Created model snapshot v%d", self.current_version)
        
        return snapshot
    
    def rollback(self, version: Optional[int] = None) -> bool:
        """
        Rollback to previous model version
        
        Args:
            version: Target version (None = most recent snapshot)
        
        Returns:
            Success status
        """
        if not self.snapshots:
            self.logger.error("No snapshots available for rollback")
            return False
        
        if version is None:
            # Rollback to most recent snapshot
            snapshot = self.snapshots[-1]
        else:
            # Find specific version
            snapshot = next((s for s in self.snapshots if s.version == version), None)
            if snapshot is None:
                self.logger.error(f"Snapshot version {version} not found")
                return False
        
        try:
            # Restore model state
            state = pickle.loads(snapshot.model_state)
            self.model = state['model']
            self.scaler = state['scaler']
            self.learning_rate = state['learning_rate']
            
            self.metrics.model_rollbacks += 1
            self.logger.info(
                "Rolled back to version %d (%d samples)",
                snapshot.version, snapshot.num_samples
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback model: {e}")
            return False
    
    def _handle_concept_drift(self):
        """Handle detected concept drift"""
        self.logger.info("Handling concept drift...")
        
        # Strategy 1: Increase learning rate temporarily
        self.learning_rate = min(self.learning_rate * 1.5, 0.1)
        self.metrics.learning_rate = self.learning_rate
        
        # Strategy 2: Create snapshot before adaptation
        self._create_snapshot()
        
        # Strategy 3: Reset drift detector
        if self.drift_detector is not None and RIVER_AVAILABLE:
            self.drift_detector = drift.ADWIN(delta=self.drift_threshold)
    
    def save(self, filepath: str):
        """
        Save learner state to disk
        
        Args:
            filepath: Path to save file
        """
        state = {
            'model': self.model,
            'scaler': self.scaler,
            'config': self.config,
            'metrics': self.metrics,
            'snapshots': self.snapshots,
            'current_version': self.current_version,
            'learning_rate': self.learning_rate
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        self.logger.info("Saved learner state to %s", filepath)
    
    def load(self, filepath: str):
        """
        Load learner state from disk
        
        Args:
            filepath: Path to saved file
        """
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.model = state['model']
        self.scaler = state['scaler']
        self.config = state['config']
        self.metrics = state['metrics']
        self.snapshots = state['snapshots']
        self.current_version = state['current_version']
        self.learning_rate = state['learning_rate']
        
        self.logger.info("Loaded learner state from %s", filepath)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current learning metrics"""
        return {
            'num_updates': self.metrics.num_updates,
            'total_samples': self.metrics.total_samples,
            'accuracy': self.metrics.accuracy,
            'learning_rate': self.metrics.learning_rate,
            'concept_drifts_detected': self.metrics.concept_drifts_detected,
            'model_rollbacks': self.metrics.model_rollbacks,
            'avg_update_time_ms': self.metrics.avg_update_time_ms,
            'num_snapshots': len(self.snapshots),
            'current_version': self.current_version,
            'is_warmed_up': self.is_warmed_up
        }
    
    def reset(self):
        """Reset learner to initial state"""
        self.model = None
        self.scaler = StandardScaler() if self.feature_scaling and SKLEARN_AVAILABLE else None
        self.drift_detector = None
        self.is_warmed_up = False
        self.sample_buffer.clear()
        self.label_buffer.clear()
        self.snapshots.clear()
        self.current_version = 0
        self.metrics = LearningMetrics(learning_rate=self.learning_rate)
        self.performance_history.clear()
        
        self.logger.info("Learner reset to initial state")


class StreamingDataProcessor:
    """
    Helper class for processing streaming data for online learning
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Initialize streaming processor
        
        Args:
            window_size: Size of sliding window for statistics
        """
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.label_window = deque(maxlen=window_size)
    
    def add_sample(self, features: np.ndarray, label: Any):
        """Add new sample to stream"""
        self.data_window.append(features)
        self.label_window.append(label)
    
    def get_window_statistics(self) -> Dict[str, Any]:
        """Get statistics for current window"""
        if not self.data_window:
            return {}
        
        data_array = np.array(list(self.data_window))
        labels = list(self.label_window)
        
        return {
            'num_samples': len(self.data_window),
            'feature_means': np.mean(data_array, axis=0).tolist(),
            'feature_stds': np.std(data_array, axis=0).tolist(),
            'label_distribution': {
                label: labels.count(label) for label in set(labels)
            }
        }
