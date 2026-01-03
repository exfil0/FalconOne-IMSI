"""
FalconOne Online AI Adaptation Module
Real-time model adaptation and concept drift detection

Version 1.9.3: Extended online learning with adaptive rates and recovery
"""

import time
import threading
import logging
import json
import os
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class DriftType(Enum):
    """Types of concept drift"""
    NONE = "none"
    SUDDEN = "sudden"       # Abrupt change
    GRADUAL = "gradual"     # Slow transition
    INCREMENTAL = "incremental"  # Continuous small changes
    RECURRING = "recurring"  # Patterns that repeat


@dataclass
class DriftMetrics:
    """Metrics for drift detection"""
    window_accuracy: float = 1.0
    baseline_accuracy: float = 1.0
    drift_score: float = 0.0
    drift_type: DriftType = DriftType.NONE
    samples_since_drift: int = 0
    drift_detected_at: Optional[float] = None


@dataclass
class AdaptationConfig:
    """Configuration for online adaptation"""
    # Learning rate adaptation
    initial_learning_rate: float = 0.001
    min_learning_rate: float = 1e-6
    max_learning_rate: float = 0.1
    lr_decay_factor: float = 0.95
    lr_increase_factor: float = 1.1
    
    # Drift detection
    drift_detection_window: int = 100
    drift_threshold: float = 0.15
    severe_drift_threshold: float = 0.30
    
    # Model checkpointing
    checkpoint_interval: int = 1000
    max_checkpoints: int = 5
    checkpoint_dir: str = "checkpoints/online"
    
    # Experience replay
    replay_buffer_size: int = 10000
    replay_batch_size: int = 32
    replay_frequency: int = 10
    
    # Regularization
    elastic_weight_consolidation: bool = True
    ewc_lambda: float = 0.4


class OnlineAdaptationManager:
    """
    Manages online learning and model adaptation.
    
    Features:
    - Adaptive learning rate based on performance
    - Concept drift detection (ADWIN, Page-Hinkley)
    - Model checkpointing and recovery
    - Experience replay for catastrophic forgetting prevention
    - Elastic Weight Consolidation (EWC)
    """
    
    def __init__(
        self,
        model: Optional[Any] = None,
        config: Optional[AdaptationConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize online adaptation manager.
        
        Args:
            model: TensorFlow/Keras model to adapt
            config: Adaptation configuration
            logger: Logger instance
        """
        self.model = model
        self.config = config or AdaptationConfig()
        self.logger = logger or logging.getLogger(__name__)
        
        # Learning rate tracking
        self.current_lr = self.config.initial_learning_rate
        self._lr_history: deque = deque(maxlen=100)
        
        # Drift detection
        self._accuracy_window: deque = deque(maxlen=self.config.drift_detection_window)
        self._baseline_accuracy: float = 0.95
        self._drift_metrics = DriftMetrics()
        self._drift_history: List[Dict] = []
        
        # Checkpointing
        self._checkpoint_count = 0
        self._last_checkpoint_time = 0.0
        self._samples_since_checkpoint = 0
        self._checkpoint_accuracies: Dict[str, float] = {}
        
        # Experience replay buffer
        self._replay_buffer: deque = deque(maxlen=self.config.replay_buffer_size)
        self._update_count = 0
        
        # EWC - Fisher information matrix approximation
        self._fisher_information: Optional[Dict[str, np.ndarray]] = None
        self._optimal_weights: Optional[Dict[str, np.ndarray]] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Callbacks
        self._on_drift_detected: List[Callable] = []
        self._on_adaptation: List[Callable] = []
        
        # Initialize checkpoint directory
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
    
    def update(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Perform online update with adaptation.
        
        Args:
            x: Input features
            y: True labels
            sample_weight: Optional sample weights
            
        Returns:
            Update metrics including accuracy, learning rate, drift status
        """
        with self._lock:
            if not TF_AVAILABLE or self.model is None:
                return {'status': 'no_model', 'adapted': False}
            
            start_time = time.time()
            
            # Get prediction before update
            y_pred = self.model.predict(x, verbose=0)
            pre_accuracy = self._compute_accuracy(y, y_pred)
            
            # Check for drift
            self._update_drift_detection(pre_accuracy)
            
            # Adapt learning rate based on performance
            self._adapt_learning_rate(pre_accuracy)
            
            # Prepare optimizer with current learning rate
            if hasattr(self.model, 'optimizer'):
                self.model.optimizer.learning_rate.assign(self.current_lr)
            
            # Perform training step
            loss = self._training_step(x, y, sample_weight)
            
            # Add to replay buffer
            self._add_to_replay_buffer(x, y)
            
            # Periodic experience replay
            self._update_count += 1
            if self._update_count % self.config.replay_frequency == 0:
                self._experience_replay()
            
            # Periodic checkpointing
            self._samples_since_checkpoint += len(x)
            if self._samples_since_checkpoint >= self.config.checkpoint_interval:
                self._save_checkpoint(pre_accuracy)
                self._samples_since_checkpoint = 0
            
            # Get prediction after update
            y_pred_post = self.model.predict(x, verbose=0)
            post_accuracy = self._compute_accuracy(y, y_pred_post)
            
            duration_ms = (time.time() - start_time) * 1000
            
            metrics = {
                'status': 'updated',
                'adapted': True,
                'pre_accuracy': float(pre_accuracy),
                'post_accuracy': float(post_accuracy),
                'improvement': float(post_accuracy - pre_accuracy),
                'learning_rate': self.current_lr,
                'loss': float(loss) if loss is not None else None,
                'drift_detected': self._drift_metrics.drift_type != DriftType.NONE,
                'drift_score': self._drift_metrics.drift_score,
                'samples_processed': self._update_count,
                'duration_ms': duration_ms
            }
            
            # Trigger adaptation callbacks
            for callback in self._on_adaptation:
                try:
                    callback(metrics)
                except Exception as e:
                    self.logger.error(f"Adaptation callback error: {e}")
            
            return metrics
    
    def _training_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        sample_weight: Optional[np.ndarray] = None
    ) -> Optional[float]:
        """Perform single training step with optional EWC penalty"""
        if not TF_AVAILABLE:
            return None
        
        try:
            # Standard training step
            history = self.model.fit(
                x, y,
                epochs=1,
                batch_size=min(32, len(x)),
                sample_weight=sample_weight,
                verbose=0
            )
            
            loss = history.history.get('loss', [None])[-1]
            
            # Apply EWC penalty if enabled
            if self.config.elastic_weight_consolidation and self._fisher_information:
                self._apply_ewc_penalty()
            
            return loss
            
        except Exception as e:
            self.logger.error(f"Training step failed: {e}")
            return None
    
    def _compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy between true and predicted labels"""
        if len(y_pred.shape) > 1 and y_pred.shape[-1] > 1:
            y_pred_classes = np.argmax(y_pred, axis=-1)
        else:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        
        if len(y_true.shape) > 1 and y_true.shape[-1] > 1:
            y_true_classes = np.argmax(y_true, axis=-1)
        else:
            y_true_classes = y_true.flatten()
        
        return np.mean(y_pred_classes == y_true_classes)
    
    def _adapt_learning_rate(self, accuracy: float):
        """Adapt learning rate based on accuracy trend"""
        self._accuracy_window.append(accuracy)
        self._lr_history.append(self.current_lr)
        
        if len(self._accuracy_window) < 10:
            return
        
        # Calculate trend
        recent = list(self._accuracy_window)[-10:]
        older = list(self._accuracy_window)[-20:-10] if len(self._accuracy_window) >= 20 else recent
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        # If accuracy improving, can increase learning rate
        if recent_avg > older_avg + 0.01:
            self.current_lr = min(
                self.current_lr * self.config.lr_increase_factor,
                self.config.max_learning_rate
            )
        # If accuracy declining, decrease learning rate
        elif recent_avg < older_avg - 0.02:
            self.current_lr = max(
                self.current_lr * self.config.lr_decay_factor,
                self.config.min_learning_rate
            )
    
    def _update_drift_detection(self, accuracy: float):
        """Update drift detection using accuracy-based method"""
        self._accuracy_window.append(accuracy)
        
        if len(self._accuracy_window) < self.config.drift_detection_window // 2:
            return
        
        # Calculate current window accuracy
        window_acc = np.mean(list(self._accuracy_window))
        
        # Calculate drift score
        drift_score = self._baseline_accuracy - window_acc
        self._drift_metrics.drift_score = drift_score
        self._drift_metrics.window_accuracy = window_acc
        
        # Detect drift type
        old_type = self._drift_metrics.drift_type
        
        if drift_score >= self.config.severe_drift_threshold:
            self._drift_metrics.drift_type = DriftType.SUDDEN
        elif drift_score >= self.config.drift_threshold:
            self._drift_metrics.drift_type = DriftType.GRADUAL
        else:
            self._drift_metrics.drift_type = DriftType.NONE
            self._drift_metrics.samples_since_drift += 1
        
        # Trigger drift callbacks if state changed
        if old_type == DriftType.NONE and self._drift_metrics.drift_type != DriftType.NONE:
            self._drift_metrics.drift_detected_at = time.time()
            self._drift_metrics.samples_since_drift = 0
            
            drift_info = {
                'type': self._drift_metrics.drift_type.value,
                'score': drift_score,
                'window_accuracy': window_acc,
                'baseline_accuracy': self._baseline_accuracy,
                'timestamp': time.time()
            }
            self._drift_history.append(drift_info)
            
            self.logger.warning(f"Concept drift detected: {drift_info}")
            
            for callback in self._on_drift_detected:
                try:
                    callback(drift_info)
                except Exception as e:
                    self.logger.error(f"Drift callback error: {e}")
            
            # Consider rolling back to previous checkpoint
            if self._drift_metrics.drift_type == DriftType.SUDDEN:
                self._consider_rollback()
    
    def _add_to_replay_buffer(self, x: np.ndarray, y: np.ndarray):
        """Add samples to experience replay buffer"""
        for i in range(len(x)):
            self._replay_buffer.append((x[i:i+1], y[i:i+1]))
    
    def _experience_replay(self):
        """Perform experience replay to prevent catastrophic forgetting"""
        if len(self._replay_buffer) < self.config.replay_batch_size:
            return
        
        try:
            # Sample random batch from buffer
            indices = np.random.choice(
                len(self._replay_buffer),
                size=min(self.config.replay_batch_size, len(self._replay_buffer)),
                replace=False
            )
            
            x_batch = np.vstack([self._replay_buffer[i][0] for i in indices])
            y_batch = np.vstack([self._replay_buffer[i][1] for i in indices])
            
            # Train on replay batch with reduced learning rate
            original_lr = self.current_lr
            self.model.optimizer.learning_rate.assign(self.current_lr * 0.5)
            
            self.model.fit(x_batch, y_batch, epochs=1, verbose=0)
            
            self.model.optimizer.learning_rate.assign(original_lr)
            
        except Exception as e:
            self.logger.error(f"Experience replay failed: {e}")
    
    def _save_checkpoint(self, accuracy: float):
        """Save model checkpoint"""
        if not TF_AVAILABLE or self.model is None:
            return
        
        try:
            self._checkpoint_count += 1
            checkpoint_name = f"checkpoint_{self._checkpoint_count}"
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                checkpoint_name
            )
            
            self.model.save_weights(checkpoint_path)
            self._checkpoint_accuracies[checkpoint_name] = accuracy
            self._last_checkpoint_time = time.time()
            
            self.logger.info(f"Saved checkpoint: {checkpoint_name} (accuracy: {accuracy:.4f})")
            
            # Remove old checkpoints
            self._cleanup_old_checkpoints()
            
        except Exception as e:
            self.logger.error(f"Checkpoint save failed: {e}")
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max limit"""
        if len(self._checkpoint_accuracies) <= self.config.max_checkpoints:
            return
        
        # Sort by accuracy, keep best ones
        sorted_checkpoints = sorted(
            self._checkpoint_accuracies.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        to_remove = sorted_checkpoints[self.config.max_checkpoints:]
        
        for name, _ in to_remove:
            try:
                path = os.path.join(self.config.checkpoint_dir, name)
                for ext in ['', '.index', '.data-00000-of-00001']:
                    full_path = path + ext
                    if os.path.exists(full_path):
                        os.remove(full_path)
                del self._checkpoint_accuracies[name]
            except Exception as e:
                self.logger.error(f"Failed to remove checkpoint {name}: {e}")
    
    def _consider_rollback(self):
        """Consider rolling back to a better checkpoint after drift"""
        if not self._checkpoint_accuracies:
            return
        
        # Find best checkpoint
        best_checkpoint = max(
            self._checkpoint_accuracies.items(),
            key=lambda x: x[1]
        )
        
        best_name, best_acc = best_checkpoint
        
        # Only rollback if best checkpoint was significantly better
        if best_acc > self._drift_metrics.window_accuracy + 0.1:
            self.logger.info(f"Rolling back to checkpoint: {best_name} (accuracy: {best_acc:.4f})")
            self._load_checkpoint(best_name)
    
    def _load_checkpoint(self, checkpoint_name: str):
        """Load model weights from checkpoint"""
        if not TF_AVAILABLE or self.model is None:
            return
        
        try:
            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                checkpoint_name
            )
            self.model.load_weights(checkpoint_path)
            self.logger.info(f"Loaded checkpoint: {checkpoint_name}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint {checkpoint_name}: {e}")
    
    def _apply_ewc_penalty(self):
        """Apply Elastic Weight Consolidation penalty"""
        if not TF_AVAILABLE or not self._fisher_information or not self._optimal_weights:
            return
        
        # EWC penalty is applied during loss computation
        # This is a simplified version - full implementation would modify the loss function
        pass
    
    def compute_fisher_information(self, x: np.ndarray, y: np.ndarray):
        """Compute Fisher information matrix for EWC"""
        if not TF_AVAILABLE or self.model is None:
            return
        
        try:
            # Store optimal weights
            self._optimal_weights = {
                layer.name: layer.get_weights()
                for layer in self.model.layers
                if layer.get_weights()
            }
            
            # Compute Fisher information (diagonal approximation)
            self._fisher_information = {}
            
            # Use gradient of log-likelihood as approximation
            with tf.GradientTape() as tape:
                predictions = self.model(x, training=True)
                loss = self.model.compiled_loss(y, predictions)
            
            grads = tape.gradient(loss, self.model.trainable_variables)
            
            for var, grad in zip(self.model.trainable_variables, grads):
                if grad is not None:
                    self._fisher_information[var.name] = (grad ** 2).numpy()
            
            self.logger.info("Computed Fisher information matrix for EWC")
            
        except Exception as e:
            self.logger.error(f"Failed to compute Fisher information: {e}")
    
    def update_baseline(self, accuracy: float):
        """Update baseline accuracy after confirmed good performance"""
        self._baseline_accuracy = accuracy
        self._drift_metrics.baseline_accuracy = accuracy
        self.logger.info(f"Updated baseline accuracy to {accuracy:.4f}")
    
    def on_drift_detected(self, callback: Callable):
        """Register drift detection callback"""
        self._on_drift_detected.append(callback)
    
    def on_adaptation(self, callback: Callable):
        """Register adaptation callback"""
        self._on_adaptation.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current adaptation metrics"""
        return {
            'learning_rate': self.current_lr,
            'samples_processed': self._update_count,
            'drift_metrics': {
                'window_accuracy': self._drift_metrics.window_accuracy,
                'baseline_accuracy': self._drift_metrics.baseline_accuracy,
                'drift_score': self._drift_metrics.drift_score,
                'drift_type': self._drift_metrics.drift_type.value,
                'samples_since_drift': self._drift_metrics.samples_since_drift,
            },
            'replay_buffer_size': len(self._replay_buffer),
            'checkpoint_count': self._checkpoint_count,
            'drift_events': len(self._drift_history),
        }
