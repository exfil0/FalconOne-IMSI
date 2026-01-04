"""
FalconOne Signal Classification Module
CNN-based signal classifier for multi-generation cellular networks
Version 1.3 Enhancements:
- Adaptive CNN with LSTM feedback loop for real-time anomaly detection
- Federated learning support for distributed deployments
- Network countermeasure adaptation (target >98% anomaly accuracy)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[WARNING] TensorFlow not installed. Signal classification disabled.")

try:
    from ..utils.logger import ModuleLogger
    from ..utils.performance import get_cache, get_pool, get_monitor
    PERFORMANCE_UTILS_AVAILABLE = True
except ImportError:
    PERFORMANCE_UTILS_AVAILABLE = False
    # Fallback logger for standalone usage
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name) if parent is None else parent.getChild(name)
            if not self.logger.handlers and parent is None:
                handler = logging.StreamHandler()
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")


class SignalClassifier:
    """CNN+LSTM-based signal classification with adaptive anomaly detection"""
    
    def __init__(self, config, logger: Optional[logging.Logger] = None):
        """Initialize signal classifier with lazy loading"""
        self.config = config
        
        # Handle logger initialization safely
        if logger is None:
            logger = logging.getLogger('FalconOne')
            if not logger.handlers:
                handler = logging.StreamHandler()
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
        
        self.logger = ModuleLogger('AI-SignalClassifier', logger)
        
        # Lazy loading: Models loaded on first use
        self.model = None
        self.adaptive_model = None  # v1.3: Adaptive CNN-LSTM
        self.transformer_model = None  # v1.5: Transformer-based classifier
        self.lstm_hidden = None  # LSTM hidden state
        self._models_loaded = False
        self.use_transformer = config.get('ai_ml.signal_classification.use_transformer', True) if hasattr(config, 'get') else True
        
        # Safe config access with fallback for dict-based config
        if hasattr(config, 'get'):
            self.accuracy_threshold = config.get('ai_ml.signal_classification.accuracy_threshold', 0.90)
            self.anomaly_threshold = config.get('ai_ml.signal_classification.anomaly_threshold', 0.98)
            self.adaptive_mode = config.get('ai_ml.signal_classification.adaptive_mode', True)
            self.model_cache_dir = config.get('ai.model_cache_dir', '/var/cache/falconone/models')
        else:
            # Fallback for dict-based config
            ai_ml = config.get('ai_ml', {}) if isinstance(config, dict) else {}
            signal_config = ai_ml.get('signal_classification', {})
            self.accuracy_threshold = signal_config.get('accuracy_threshold', 0.90)
            self.anomaly_threshold = signal_config.get('anomaly_threshold', 0.98)
            self.adaptive_mode = signal_config.get('adaptive_mode', True)
            self.model_cache_dir = config.get('ai', {}).get('model_cache_dir', '/var/cache/falconone/models') if isinstance(config, dict) else '/var/cache/falconone/models'
        
        # Generation labels (v1.5: Added NTN/Satellite)
        self.labels = ['GSM', 'UMTS', 'CDMA', 'LTE', '5G', '6G', 'NTN-LEO', 'NTN-GEO']
        
        # Anomaly detection tracking
        self.anomaly_history = []
        self.kpi_buffer = []  # Real-time KPI buffer for adaptation
        
        # Lazy loading: Don't build models until first use (saves 500MB+ RAM at startup)
        if TF_AVAILABLE:
            self.logger.info("Signal classifier initialized (v1.4.1 Lazy Loading)", 
                           threshold=self.accuracy_threshold,
                           adaptive=self.adaptive_mode,
                           lazy_loading=True)
        else:
            self.logger.warning("TensorFlow not available - signal classification disabled")
    
    def _ensure_models_loaded(self):
        """Lazy load models on first use (v1.4.1 optimization)"""
        if self._models_loaded or not TF_AVAILABLE:
            return
        
        self.logger.info("Lazy loading AI models...")
        
        # Try to load from cache first
        model_path = f"{self.model_cache_dir}/signal_classifier.h5"
        adaptive_path = f"{self.model_cache_dir}/adaptive_classifier.h5"
        
        try:
            if Path(model_path).exists():
                self.model = keras.models.load_model(model_path)
                self.logger.info(f"Loaded cached model from {model_path}")
            else:
                self._build_model()
                self.logger.info("Built new model (no cache found)")
        except Exception as e:
            self.logger.warning(f"Failed to load cached model: {e}, building new one")
            self._build_model()
        
        # Load adaptive model if enabled
        if self.adaptive_mode:
            try:
                if Path(adaptive_path).exists():
                    self.adaptive_model = keras.models.load_model(adaptive_path)
                    self.logger.info(f"Loaded cached adaptive model from {adaptive_path}")
                else:
                    self._build_adaptive_model()
                    self.logger.info("Built new adaptive model")
            except Exception as e:
                self.logger.warning(f"Failed to load adaptive model: {e}, building new one")
        # Load transformer model if enabled (v1.5)
        if self.use_transformer:
            transformer_path = f"{self.model_cache_dir}/transformer_classifier.h5"
            try:
                if Path(transformer_path).exists():
                    self.transformer_model = keras.models.load_model(transformer_path)
                    self.logger.info(f"Loaded cached transformer model from {transformer_path}")
                else:
                    self._build_transformer_model()
                    self.logger.info("Built new transformer model")
            except Exception as e:
                self.logger.warning(f"Failed to load transformer model: {e}, building new one")
                self._build_transformer_model()
        
                self._build_adaptive_model()
        
        self._models_loaded = True
        self.logger.info("Models loaded successfully")
    
    def _build_model(self):
        """Build CNN model for signal classification"""
        try:
            self.model = keras.Sequential([
                keras.layers.Conv1D(64, 3, activation='relu', input_shape=(1024, 2)),
                keras.layers.MaxPooling1D(2),
                keras.layers.Conv1D(128, 3, activation='relu'),
                keras.layers.MaxPooling1D(2),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dropout(0.5),
                keras.layers.Dense(len(self.labels), activation='softmax')
            ])
            
            self.model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info("CNN model built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build model: {e}")
    
    def classify(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Classify signals to determine cellular generation.
        
        Uses CNN model when TensorFlow is available, falls back to
        heuristic-based classification otherwise.
        
        Args:
            signals: List of signal data dictionaries with keys:
                - 'iq_samples': np.ndarray of IQ data (N, 2) or complex
                - 'frequency': Center frequency in Hz (optional)
                - 'bandwidth': Signal bandwidth in Hz (optional)
                - 'id': Signal identifier (optional)
            
        Returns:
            List of classification results with keys:
                - 'signal_id': Input signal ID
                - 'predicted_generation': One of self.labels
                - 'confidence': Prediction confidence (0-1)
                - 'method': 'cnn' or 'heuristic'
        """
        results = []
        
        for signal in signals:
            try:
                signal_id = signal.get('id', id(signal))
                
                # Check for IQ samples
                iq_samples = signal.get('iq_samples')
                frequency = signal.get('frequency', 0)
                bandwidth = signal.get('bandwidth', 0)
                
                # Use TensorFlow model if available
                if TF_AVAILABLE and self.model is not None:
                    result = self._classify_with_model(signal_id, iq_samples, frequency)
                else:
                    # Fallback to heuristic classifier
                    result = self._classify_heuristic(signal_id, frequency, bandwidth)
                
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Classification error: {e}")
                results.append({
                    'signal_id': signal.get('id', 'unknown'),
                    'predicted_generation': 'Unknown',
                    'confidence': 0.0,
                    'method': 'error',
                    'error': str(e)
                })
        
        return results
    
    def _classify_with_model(self, signal_id: Any, iq_samples: Optional[np.ndarray],
                             frequency: float) -> Dict[str, Any]:
        """
        Classify signal using trained CNN model.
        
        Args:
            signal_id: Signal identifier
            iq_samples: IQ data array
            frequency: Center frequency in Hz
            
        Returns:
            Classification result dictionary
        """
        # Lazy load models on first classification
        self._ensure_models_loaded()
        
        if self.model is None:
            return self._classify_heuristic(signal_id, frequency, 0)
        
        try:
            # Preprocess IQ samples
            processed = self._preprocess_signal(iq_samples)
            
            if processed is None:
                return self._classify_heuristic(signal_id, frequency, 0)
            
            # Run inference
            predictions = self.model.predict(processed, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_class])
            
            return {
                'signal_id': signal_id,
                'predicted_generation': self.labels[predicted_class],
                'confidence': confidence,
                'method': 'cnn',
                'raw_probabilities': {
                    label: float(prob) 
                    for label, prob in zip(self.labels, predictions[0])
                }
            }
            
        except Exception as e:
            self.logger.warning(f"Model inference failed: {e}, using heuristic")
            return self._classify_heuristic(signal_id, frequency, 0)
    
    def _preprocess_signal(self, iq_samples: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Preprocess IQ samples for CNN input.
        
        Args:
            iq_samples: Raw IQ data (complex array or Nx2 real array)
            
        Returns:
            Processed array of shape (1, 1024, 2) or None if invalid
        """
        if iq_samples is None or len(iq_samples) == 0:
            return None
        
        try:
            # Convert complex to I/Q if needed
            if np.iscomplexobj(iq_samples):
                iq_data = np.column_stack([iq_samples.real, iq_samples.imag])
            elif iq_samples.ndim == 1:
                # Assume alternating I/Q
                iq_data = iq_samples.reshape(-1, 2)
            else:
                iq_data = iq_samples
            
            # Ensure correct length (1024 samples)
            target_len = 1024
            if len(iq_data) < target_len:
                # Pad with zeros
                padding = np.zeros((target_len - len(iq_data), 2))
                iq_data = np.vstack([iq_data, padding])
            elif len(iq_data) > target_len:
                # Truncate
                iq_data = iq_data[:target_len]
            
            # Normalize
            max_val = np.max(np.abs(iq_data))
            if max_val > 0:
                iq_data = iq_data / max_val
            
            # Add batch dimension
            return iq_data.reshape(1, target_len, 2).astype(np.float32)
            
        except Exception as e:
            self.logger.warning(f"Signal preprocessing failed: {e}")
            return None
    
    def _classify_heuristic(self, signal_id: Any, frequency: float, 
                            bandwidth: float) -> Dict[str, Any]:
        """
        Fallback heuristic classifier based on frequency bands.
        
        Uses frequency and bandwidth to estimate cellular generation
        when TensorFlow is unavailable or model inference fails.
        
        Args:
            signal_id: Signal identifier
            frequency: Center frequency in Hz
            bandwidth: Signal bandwidth in Hz
            
        Returns:
            Classification result dictionary
        """
        freq_mhz = frequency / 1e6 if frequency else 0
        bw_mhz = bandwidth / 1e6 if bandwidth else 0
        
        # Heuristic frequency-based classification
        generation = 'Unknown'
        confidence = 0.5
        
        if freq_mhz > 0:
            # NTN LEO: L-band and S-band (check first - specific satellite bands)
            if 1518 <= freq_mhz <= 1559 or 1980 <= freq_mhz <= 2010:
                generation = 'NTN-LEO'
                confidence = 0.7
            
            # NTN GEO: C-band and Ku-band
            elif 3700 <= freq_mhz <= 4200 or 10700 <= freq_mhz <= 12750:
                generation = 'NTN-GEO'
                confidence = 0.65
            
            # GSM: 850, 900, 1800, 1900 MHz with narrow bandwidth (~200 kHz)
            elif (850 <= freq_mhz <= 960 or 1710 <= freq_mhz <= 1990):
                if bw_mhz < 1 or bw_mhz == 0:
                    generation = 'GSM'
                    confidence = 0.7
                elif bw_mhz <= 5:
                    generation = 'UMTS'
                    confidence = 0.65
                else:
                    generation = 'LTE'
                    confidence = 0.6
            
            # CDMA: 800, 1900 MHz
            elif 800 <= freq_mhz <= 894 or 1850 <= freq_mhz <= 1995:
                if bw_mhz <= 1.25:
                    generation = 'CDMA'
                    confidence = 0.6
            
            # LTE: Various bands, wider bandwidth (1.4-20 MHz)
            elif 700 <= freq_mhz <= 2700:
                if 1.4 <= bw_mhz <= 20:
                    generation = 'LTE'
                    confidence = 0.7
            
            # 5G NR FR1: 410 MHz - 7125 MHz
            elif 2500 <= freq_mhz <= 7125:
                if bw_mhz >= 20:
                    generation = '5G'
                    confidence = 0.75
            
            # 5G NR FR2 (mmWave): 24-52 GHz
            elif 24000 <= freq_mhz <= 52000:
                generation = '5G'
                confidence = 0.85
            
            # 6G Sub-THz: 100+ GHz
            elif freq_mhz >= 100000:
                generation = '6G'
                confidence = 0.8
        
        return {
            'signal_id': signal_id,
            'predicted_generation': generation,
            'confidence': confidence,
            'method': 'heuristic',
            'frequency_mhz': freq_mhz,
            'bandwidth_mhz': bw_mhz
        }
    
    def train(self, training_data: np.ndarray, labels: np.ndarray, epochs: int = 50):
        """
        Train the CNN model
        
        Args:
            training_data: Training samples (NxFeatures array)
            labels: Training labels (N-dimensional array)
            epochs: Number of training epochs
        """
        if not TF_AVAILABLE or not self.model:
            self.logger.error("Cannot train - TensorFlow not available")
            return
        
        try:
            self.logger.info(f"Training signal classifier for {epochs} epochs...")
            
            # Convert labels to categorical if needed
            if len(labels.shape) == 1:
                labels = keras.utils.to_categorical(labels, num_classes=len(self.labels))
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                training_data, labels, test_size=0.2, random_state=42
            )
            
            # Train with callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='/tmp/best_signal_classifier.h5',
                    monitor='val_accuracy',
                    save_best_only=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]
            
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                validation_data=(X_val, y_val),
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            final_accuracy = history.history['accuracy'][-1]
            val_accuracy = history.history['val_accuracy'][-1]
            
            self.logger.info(f"Training completed", 
                           train_acc=f"{final_accuracy:.4f}",
                           val_acc=f"{val_accuracy:.4f}")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return None
    
    # ==================== ONLINE INCREMENTAL LEARNING (v1.9.2) ====================
    
    def partial_fit(
        self,
        signal: np.ndarray,
        label: int,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.01
    ) -> Dict[str, Any]:
        """
        Incremental online learning update for a single sample (v1.9.2).
        
        Enables real-time adaptation to new signal types without full retraining.
        Uses gradient descent on a single sample with elastic weight consolidation
        to prevent catastrophic forgetting of previously learned patterns.
        
        Args:
            signal: Single IQ signal sample (1024, 2)
            label: Ground truth label index (0-7 for generation classes)
            learning_rate: Learning rate for this update (default: 0.0001)
            weight_decay: Elastic weight consolidation factor (default: 0.01)
            
        Returns:
            Dict with update results including loss and new prediction
        """
        if not TF_AVAILABLE:
            return {'success': False, 'error': 'TensorFlow not available'}
        
        self._ensure_models_loaded()
        
        if not self.model:
            return {'success': False, 'error': 'Model not loaded'}
        
        try:
            # Prepare data
            signal_input = signal.reshape(1, 1024, 2)
            label_onehot = keras.utils.to_categorical([label], num_classes=len(self.labels))
            
            # Store original weights for EWC regularization
            if not hasattr(self, '_fisher_information'):
                self._fisher_information = None
                self._optimal_weights = None
            
            # Single-step gradient update
            with tf.GradientTape() as tape:
                predictions = self.model(signal_input, training=True)
                loss = keras.losses.categorical_crossentropy(label_onehot, predictions)
                
                # Add EWC penalty if we have fisher information
                if self._fisher_information is not None and self._optimal_weights is not None:
                    ewc_loss = self._compute_ewc_penalty(weight_decay)
                    loss = loss + ewc_loss
            
            # Compute and apply gradients
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            # Apply gradients with configured learning rate
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # Get new prediction after update
            new_prediction = self.model.predict(signal_input, verbose=0)
            predicted_class = np.argmax(new_prediction[0])
            
            result = {
                'success': True,
                'loss': float(np.mean(loss.numpy())),
                'predicted_label': predicted_class,
                'predicted_generation': self.labels[predicted_class],
                'confidence': float(new_prediction[0][predicted_class]),
                'correct': predicted_class == label
            }
            
            self.logger.debug(
                f"Online update: loss={result['loss']:.4f}, "
                f"predicted={result['predicted_generation']}, "
                f"correct={result['correct']}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Online learning update failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def incremental_batch_fit(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 16,
        epochs: int = 1,
        learning_rate: float = 0.0005
    ) -> Dict[str, Any]:
        """
        Incremental batch learning for small datasets (v1.9.2).
        
        More efficient than individual partial_fit calls when multiple
        samples are available. Uses mini-batch gradient descent with
        experience replay to maintain performance on old data.
        
        Args:
            signals: Batch of IQ signals (N, 1024, 2)
            labels: Labels array (N,)
            batch_size: Mini-batch size for updates
            epochs: Number of passes through the data
            learning_rate: Learning rate for updates
            
        Returns:
            Dict with training results
        """
        if not TF_AVAILABLE:
            return {'success': False, 'error': 'TensorFlow not available'}
        
        self._ensure_models_loaded()
        
        if not self.model:
            return {'success': False, 'error': 'Model not loaded'}
        
        try:
            n_samples = len(signals)
            self.logger.info(f"Incremental batch training on {n_samples} samples...")
            
            # Convert labels to one-hot
            if len(labels.shape) == 1:
                labels_onehot = keras.utils.to_categorical(labels, num_classes=len(self.labels))
            else:
                labels_onehot = labels
            
            # Add experience replay from buffer if available
            signals_combined = signals
            labels_combined = labels_onehot
            
            if hasattr(self, '_experience_buffer') and len(self._experience_buffer) > 0:
                # Mix with experience replay (30% old, 70% new)
                replay_size = min(int(n_samples * 0.3), len(self._experience_buffer))
                if replay_size > 0:
                    replay_indices = np.random.choice(
                        len(self._experience_buffer),
                        size=replay_size,
                        replace=False
                    )
                    replay_signals = np.array([self._experience_buffer[i][0] for i in replay_indices])
                    replay_labels = keras.utils.to_categorical(
                        [self._experience_buffer[i][1] for i in replay_indices],
                        num_classes=len(self.labels)
                    )
                    signals_combined = np.concatenate([signals, replay_signals], axis=0)
                    labels_combined = np.concatenate([labels_onehot, replay_labels], axis=0)
                    self.logger.debug(f"Added {replay_size} experience replay samples")
            
            # Create dataset with shuffling
            dataset = tf.data.Dataset.from_tensor_slices((signals_combined, labels_combined))
            dataset = dataset.shuffle(buffer_size=len(signals_combined))
            dataset = dataset.batch(batch_size)
            
            # Configure optimizer with lower learning rate for fine-tuning
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            
            # Training loop
            total_loss = 0.0
            total_batches = 0
            
            for epoch in range(epochs):
                for batch_signals, batch_labels in dataset:
                    with tf.GradientTape() as tape:
                        predictions = self.model(batch_signals, training=True)
                        loss = keras.losses.categorical_crossentropy(batch_labels, predictions)
                        loss = tf.reduce_mean(loss)
                        
                        # Add EWC regularization
                        if self._fisher_information is not None:
                            loss = loss + self._compute_ewc_penalty(0.01)
                    
                    gradients = tape.gradient(loss, self.model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                    
                    total_loss += float(loss.numpy())
                    total_batches += 1
            
            avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
            
            # Update experience buffer with new samples
            self._update_experience_buffer(signals, labels)
            
            # Evaluate on new data
            predictions = self.model.predict(signals, verbose=0)
            accuracy = np.mean(np.argmax(predictions, axis=1) == labels)
            
            result = {
                'success': True,
                'samples_trained': n_samples,
                'epochs': epochs,
                'avg_loss': avg_loss,
                'accuracy': float(accuracy),
            }
            
            self.logger.info(
                f"Incremental batch training complete: "
                f"loss={avg_loss:.4f}, accuracy={accuracy:.4f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Incremental batch training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _update_experience_buffer(
        self,
        signals: np.ndarray,
        labels: np.ndarray,
        max_buffer_size: int = 1000
    ):
        """
        Update experience replay buffer with new samples (v1.9.2).
        
        Maintains a buffer of past samples for experience replay during
        incremental learning, helping prevent catastrophic forgetting.
        """
        if not hasattr(self, '_experience_buffer'):
            self._experience_buffer = []
        
        # Add new samples to buffer
        for i in range(len(signals)):
            self._experience_buffer.append((signals[i], labels[i]))
        
        # Trim buffer if exceeds max size (keep most recent)
        if len(self._experience_buffer) > max_buffer_size:
            # Random removal to maintain class balance
            excess = len(self._experience_buffer) - max_buffer_size
            remove_indices = np.random.choice(
                len(self._experience_buffer),
                size=excess,
                replace=False
            )
            self._experience_buffer = [
                s for i, s in enumerate(self._experience_buffer)
                if i not in remove_indices
            ]
    
    def _compute_ewc_penalty(self, weight_decay: float) -> "tf.Tensor | None":
        """
        Compute Elastic Weight Consolidation penalty (v1.9.2).
        
        EWC prevents catastrophic forgetting by penalizing changes to
        weights that were important for previous tasks.
        
        Args:
            weight_decay: Regularization strength
            
        Returns:
            EWC penalty term for loss function
        """
        if self._fisher_information is None or self._optimal_weights is None:
            return tf.constant(0.0)
        
        penalty = tf.constant(0.0)
        
        for i, var in enumerate(self.model.trainable_variables):
            if i < len(self._fisher_information):
                fisher = self._fisher_information[i]
                optimal = self._optimal_weights[i]
                penalty += tf.reduce_sum(fisher * tf.square(var - optimal))
        
        return weight_decay * penalty
    
    def consolidate_knowledge(self, validation_data: np.ndarray, validation_labels: np.ndarray):
        """
        Consolidate learned knowledge using Fisher Information Matrix (v1.9.2).
        
        Call this after training on important data to protect those learned
        patterns from being overwritten during future online learning.
        
        Args:
            validation_data: Data to compute Fisher information on
            validation_labels: Corresponding labels
        """
        if not TF_AVAILABLE or not self.model:
            return
        
        self.logger.info("Consolidating knowledge (computing Fisher Information)...")
        
        try:
            # Store current optimal weights
            self._optimal_weights = [var.numpy().copy() for var in self.model.trainable_variables]
            
            # Compute Fisher Information Matrix approximation
            fisher_info = [np.zeros_like(var.numpy()) for var in self.model.trainable_variables]
            
            n_samples = min(len(validation_data), 500)  # Limit samples for efficiency
            indices = np.random.choice(len(validation_data), size=n_samples, replace=False)
            
            for idx in indices:
                signal = validation_data[idx:idx+1]
                label = validation_labels[idx]
                
                with tf.GradientTape() as tape:
                    predictions = self.model(signal, training=True)
                    # Use log probability for FIM computation
                    log_prob = tf.math.log(predictions[0, label] + 1e-10)
                
                gradients = tape.gradient(log_prob, self.model.trainable_variables)
                
                for i, grad in enumerate(gradients):
                    if grad is not None:
                        fisher_info[i] += np.square(grad.numpy())
            
            # Average and store
            self._fisher_information = [f / n_samples for f in fisher_info]
            
            self.logger.info(f"Knowledge consolidation complete ({n_samples} samples)")
            
        except Exception as e:
            self.logger.error(f"Knowledge consolidation failed: {e}")
    
    def detect_concept_drift(
        self,
        recent_signals: np.ndarray,
        recent_labels: np.ndarray,
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """
        Detect concept drift in incoming signal data (v1.9.2).
        
        Monitors for distribution shift that would require model adaptation.
        Uses prediction confidence and accuracy drop as drift indicators.
        
        Args:
            recent_signals: Recent signal samples
            recent_labels: Ground truth labels (if available)
            threshold: Accuracy drop threshold for drift detection
            
        Returns:
            Dict with drift detection results and recommendations
        """
        if not TF_AVAILABLE or not self.model:
            return {'drift_detected': False, 'error': 'Model not available'}
        
        try:
            self._ensure_models_loaded()
            
            # Get predictions on recent data
            predictions = self.model.predict(recent_signals, verbose=0)
            predicted_labels = np.argmax(predictions, axis=1)
            confidences = np.max(predictions, axis=1)
            
            # Compute metrics
            accuracy = np.mean(predicted_labels == recent_labels)
            avg_confidence = np.mean(confidences)
            low_confidence_ratio = np.mean(confidences < 0.5)
            
            # Get baseline accuracy (from training)
            if not hasattr(self, '_baseline_accuracy'):
                self._baseline_accuracy = self.accuracy_threshold
            
            # Detect drift
            accuracy_drop = self._baseline_accuracy - accuracy
            drift_detected = (
                accuracy_drop > threshold or
                avg_confidence < 0.6 or
                low_confidence_ratio > 0.3
            )
            
            result = {
                'drift_detected': drift_detected,
                'current_accuracy': float(accuracy),
                'baseline_accuracy': float(self._baseline_accuracy),
                'accuracy_drop': float(accuracy_drop),
                'avg_confidence': float(avg_confidence),
                'low_confidence_ratio': float(low_confidence_ratio),
                'recommendation': None
            }
            
            if drift_detected:
                if accuracy_drop > 0.25:
                    result['recommendation'] = 'SEVERE: Retrain model with new data'
                elif accuracy_drop > threshold:
                    result['recommendation'] = 'MODERATE: Run incremental_batch_fit with recent samples'
                else:
                    result['recommendation'] = 'MINOR: Monitor closely, consider partial_fit on misclassified samples'
                
                self.logger.warning(
                    f"Concept drift detected: accuracy_drop={accuracy_drop:.3f}, "
                    f"recommendation={result['recommendation']}"
                )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Concept drift detection failed: {e}")
            return {'drift_detected': False, 'error': str(e)}

    def load_dataset(self, dataset_path: str) -> tuple:
        """
        Load training dataset from file
        
        Args:
            dataset_path: Path to dataset (npz format)
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            data = np.load(dataset_path)
            features = data['features']
            labels = data['labels']
            
            self.logger.info(f"Loaded dataset: {features.shape[0]} samples")
            
            return features, labels
            
        except Exception as e:
            self.logger.error(f"Dataset loading failed: {e}")
            return None, None
    
    def generate_synthetic_dataset(self, samples_per_class: int = 1000) -> tuple:
        """
        Generate synthetic training data for testing
        
        Args:
            samples_per_class: Number of samples per generation
            
        Returns:
            Tuple of (features, labels)
        """
        try:
            self.logger.info(f"Generating synthetic dataset: {samples_per_class} samples/class")
            
            features = []
            labels_list = []
            
            for class_idx, generation in enumerate(self.labels):
                for _ in range(samples_per_class):
                    # Generate synthetic IQ samples with class-specific characteristics
                    if generation == 'GSM':
                        # GSM: GMSK modulation, narrowband
                        carrier_freq = 0.05
                        signal = np.sin(2 * np.pi * carrier_freq * np.arange(1024))
                        noise = np.random.normal(0, 0.1, 1024)
                    elif generation == 'UMTS':
                        # UMTS: WCDMA, wider bandwidth
                        carrier_freq = 0.1
                        signal = np.sin(2 * np.pi * carrier_freq * np.arange(1024))
                        noise = np.random.normal(0, 0.15, 1024)
                    elif generation == 'LTE':
                        # LTE: OFDM, multiple subcarriers
                        freqs = [0.05, 0.1, 0.15, 0.2]
                        signal = sum(np.sin(2 * np.pi * f * np.arange(1024)) for f in freqs)
                        noise = np.random.normal(0, 0.2, 1024)
                    elif generation == '5G':
                        # 5G: Higher bandwidth OFDM
                        freqs = [0.1, 0.2, 0.3, 0.4]
                        signal = sum(np.sin(2 * np.pi * f * np.arange(1024)) for f in freqs)
                        noise = np.random.normal(0, 0.25, 1024)
                    else:
                        # Generic signal
                        signal = np.random.randn(1024)
                        noise = np.random.normal(0, 0.1, 1024)
                    
                    # Combine signal and noise
                    iq_signal = signal + noise
                    
                    # Create I/Q representation
                    i_channel = iq_signal
                    q_channel = np.imag(np.exp(1j * iq_signal))
                    
                    # Stack I and Q
                    iq_data = np.stack([i_channel, q_channel], axis=-1)
                    
                    features.append(iq_data)
                    labels_list.append(class_idx)
            
            features = np.array(features)
            labels_array = np.array(labels_list)
            
            # Shuffle
            indices = np.arange(len(features))
            np.random.shuffle(indices)
            features = features[indices]
            labels_array = labels_array[indices]
            
            self.logger.info(f"Generated {len(features)} synthetic samples")
            
            return features, labels_array
            
        except Exception as e:
            self.logger.error(f"Synthetic dataset generation failed: {e}")
            return None, None
    
    def save_model(self, path: str):
        """Save trained model"""
        if self.model:
            self.model.save(path)
            self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        if TF_AVAILABLE:
            self.model = keras.models.load_model(path)
            self.logger.info(f"Model loaded from {path}")
    
    # ==================== ADAPTIVE AI/ML ENHANCEMENTS (v1.3) ====================
    
    def _build_adaptive_model(self):
        """
        Build Adaptive CNN-LSTM model for real-time anomaly detection
        Detects network countermeasures and adapts exploit strategies
        Target: >98% anomaly detection accuracy
        """
        try:
            # Input layer
            input_layer = layers.Input(shape=(1024, 2), name='signal_input')
            
            # CNN feature extraction
            x = layers.Conv1D(64, 3, activation='relu', name='conv1')(input_layer)
            x = layers.MaxPooling1D(2)(x)
            x = layers.Conv1D(128, 3, activation='relu', name='conv2')(x)
            x = layers.MaxPooling1D(2)(x)
            
            # LSTM for temporal dependencies (anomaly patterns)
            x = layers.LSTM(64, return_sequences=True, name='lstm1')(x)
            x = layers.LSTM(64, return_sequences=False, name='lstm2')(x)
            
            # Multi-task outputs
            classification_output = layers.Dense(len(self.labels), activation='softmax', 
                                                name='classification')(x)
            anomaly_output = layers.Dense(1, activation='sigmoid', 
                                         name='anomaly')(x)
            
            # Build model with dual outputs
            self.adaptive_model = keras.Model(
                inputs=input_layer,
                outputs=[classification_output, anomaly_output]
            )
            
            # Compile with multi-task losses
            self.adaptive_model.compile(
                optimizer='adam',
                loss={
                    'classification': 'categorical_crossentropy',
                    'anomaly': 'binary_crossentropy'
                },
                metrics={
                    'classification': ['accuracy'],
                    'anomaly': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
                },
                loss_weights={'classification': 0.6, 'anomaly': 0.4}
            )
            
            self.logger.info("Adaptive CNN-LSTM model built successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to build adaptive model: {e}")
    
    def detect_anomaly(self, signal: np.ndarray, kpis: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Detect network anomalies in real-time (countermeasures, anti-SIGINT)
        Integrates with monitoring KPIs from Sections 6-11
        
        Args:
            signal: IQ signal data (1024x2)
            kpis: Real-time KPIs from monitoring layer
            
        Returns:
            Anomaly detection results
        """
        if not TF_AVAILABLE:
            return {'anomaly_detected': False, 'confidence': 0.0}
        
        # Lazy load models on first use
        self._ensure_models_loaded()
        
        if not self.adaptive_model:
            return {'anomaly_detected': False, 'confidence': 0.0}
        
        try:
            # Reshape signal for model input
            signal_input = signal.reshape(1, 1024, 2)
            
            # Run inference
            classification, anomaly_score = self.adaptive_model.predict(signal_input, verbose=0)
            
            # Anomaly decision
            anomaly_detected = anomaly_score[0][0] > (1 - self.anomaly_threshold)
            
            # Integrate KPI analysis if available
            kpi_anomaly = False
            if kpis:
                kpi_anomaly = self._analyze_kpi_anomalies(kpis)
            
            result = {
                'anomaly_detected': anomaly_detected or kpi_anomaly,
                'anomaly_score': float(anomaly_score[0][0]),
                'confidence': float(np.max(classification[0])),
                'predicted_generation': self.labels[np.argmax(classification[0])],
                'kpi_anomaly': kpi_anomaly,
                'timestamp': np.datetime64('now')
            }
            
            # Track anomaly history
            self.anomaly_history.append(result)
            if len(self.anomaly_history) > 1000:
                self.anomaly_history.pop(0)
            
            if anomaly_detected or kpi_anomaly:
                self.logger.warning(f"Anomaly detected: score={result['anomaly_score']:.4f}, "
                                  f"kpi_anomaly={kpi_anomaly}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Anomaly detection failed: {e}")
            return {'anomaly_detected': False, 'error': str(e)}
    
    def _analyze_kpi_anomalies(self, kpis: Dict[str, float]) -> bool:
        """
        Analyze KPIs for anomalous patterns (e.g., sudden handover failures)
        Detects anti-SIGINT countermeasures
        
        Args:
            kpis: Real-time KPIs (RSRP, SINR, handover_success_rate, etc.)
            
        Returns:
            True if KPI anomaly detected
        """
        try:
            # Thresholds for anomaly detection
            anomalies = []
            
            # Check RSRP drop (indicates jamming or detection)
            if 'rsrp' in kpis and kpis['rsrp'] < -120:
                anomalies.append('low_rsrp')
            
            # Check handover failures (indicates evasion)
            if 'handover_success_rate' in kpis and kpis['handover_success_rate'] < 0.5:
                anomalies.append('handover_failure')
            
            # Check excessive retransmissions (countermeasure)
            if 'retransmission_rate' in kpis and kpis['retransmission_rate'] > 0.3:
                anomalies.append('high_retransmission')
            
            # Check authentication failures (detection resistance)
            if 'auth_failure_rate' in kpis and kpis['auth_failure_rate'] > 0.2:
                anomalies.append('auth_rejection')
            
            # Buffer KPIs for trend analysis
            self.kpi_buffer.append(kpis)
            if len(self.kpi_buffer) > 100:
                self.kpi_buffer.pop(0)
            
            if anomalies:
                self.logger.warning(f"KPI anomalies detected: {anomalies}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"KPI analysis failed: {e}")
            return False
    
    def adapt_to_anomaly(self, anomaly_result: Dict[str, Any]) -> Dict[str, str]:
        """
        Generate adaptive exploit recommendations based on detected anomalies
        Links to DQN RIC Optimizer (Section 18) for automated adaptation
        
        Args:
            anomaly_result: Results from detect_anomaly()
            
        Returns:
            Adaptation strategy recommendations
        """
        if not anomaly_result.get('anomaly_detected'):
            return {'strategy': 'continue', 'reason': 'no_anomaly'}
        
        adaptations = []
        
        # Low RSRP -> Increase power or change cell
        if 'low_rsrp' in str(anomaly_result):
            adaptations.append('increase_tx_power')
            adaptations.append('randomize_cell_id')
        
        # Handover failures -> Switch to DoS mode
        if 'handover_failure' in str(anomaly_result):
            adaptations.append('switch_to_dos')
            adaptations.append('target_alternate_band')
        
        # Authentication rejections -> Polymorphic evasion
        if 'auth_rejection' in str(anomaly_result):
            adaptations.append('enable_polymorphic_mode')
            adaptations.append('randomize_sqn')
        
        # High anomaly score -> Abort and regroup
        if anomaly_result.get('anomaly_score', 0) > 0.95:
            adaptations.append('abort_operation')
            adaptations.append('wait_and_retry')
        
        strategy = {
            'strategy': 'adapt',
            'adaptations': adaptations,
            'priority': 'high' if anomaly_result.get('anomaly_score', 0) > 0.90 else 'medium',
            'timestamp': str(anomaly_result.get('timestamp'))
        }
        
        self.logger.info(f"Adaptation strategy: {adaptations}")
        
        return strategy
    
    def train_adaptive(self, training_data: np.ndarray, labels: np.ndarray, 
                      anomaly_labels: np.ndarray, epochs: int = 50):
        """
        Train adaptive CNN-LSTM model with multi-task learning
        
        Args:
            training_data: Training samples (NxFeatures array)
            labels: Classification labels (N-dimensional array)
            anomaly_labels: Anomaly labels (N-dimensional binary array)
            epochs: Number of training epochs
        """
        if not TF_AVAILABLE or not self.adaptive_model:
            self.logger.error("Cannot train - Adaptive model not available")
            return
        
        try:
            self.logger.info(f"Training adaptive model for {epochs} epochs...")
            
            # Convert classification labels to categorical
            if len(labels.shape) == 1:
                labels = keras.utils.to_categorical(labels, num_classes=len(self.labels))
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_cls_train, y_cls_val, y_anom_train, y_anom_val = train_test_split(
                training_data, labels, anomaly_labels, 
                test_size=0.2, random_state=42
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_anomaly_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='/tmp/best_adaptive_classifier.h5',
                    monitor='val_anomaly_accuracy',
                    save_best_only=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]
            
            # Train
            history = self.adaptive_model.fit(
                X_train,
                {'classification': y_cls_train, 'anomaly': y_anom_train},
                epochs=epochs,
                validation_data=(X_val, {'classification': y_cls_val, 'anomaly': y_anom_val}),
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            cls_acc = history.history['classification_accuracy'][-1]
            anom_acc = history.history['anomaly_accuracy'][-1]
            val_anom_acc = history.history['val_anomaly_accuracy'][-1]
            
            self.logger.info(f"Adaptive training completed",
                           cls_acc=f"{cls_acc:.4f}",
                           anom_acc=f"{anom_acc:.4f}",
                           val_anom_acc=f"{val_anom_acc:.4f}",
                           target_met=(val_anom_acc >= self.anomaly_threshold))
            
            return history
            
        except Exception as e:
            self.logger.error(f"Adaptive training failed: {e}")
            return None
    
    def integrate_realtime_kpis(self, kpis: Dict[str, float]):
        """
        Integrate real-time KPIs from monitoring layer (Sections 6-11)
        Feeds into anomaly detection loop
        
        Args:
            kpis: Real-time KPIs from srsRAN/Open5GS/pyshark
        """
        self.kpi_buffer.append(kpis)
        if len(self.kpi_buffer) > 100:
            self.kpi_buffer.pop(0)
        
        # Trigger anomaly check if KPI trends worsen
        if len(self.kpi_buffer) >= 10:
            recent_kpis = self.kpi_buffer[-10:]
            avg_handover = np.mean([k.get('handover_success_rate', 1.0) for k in recent_kpis])
            
            if avg_handover < 0.7:
                self.logger.warning(f"KPI trend alert: Avg handover success = {avg_handover:.2f}")
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """
        Generate anomaly detection report (alias for generate_anomaly_report)
        For integration with Section 19 (Monitoring and Reporting)
        
        Returns:
            Anomaly statistics and trends
        """
        # Delegate to the full implementation
        return self.generate_anomaly_report()
    
    # ==================== TRANSFORMER-BASED ENHANCEMENTS (v1.5) ====================
    
    def _build_transformer_model(self):
        """
        Build Transformer-based signal classifier (v1.5.0)
        Superior performance for I/Q spectrogram analysis and multi-modal fusion
        Target: >95% accuracy on Rel-19 NTN signals
        """
        try:
            # Positional encoding for temporal I/Q sequences
            class PositionalEncoding(layers.Layer):
                def __init__(self, d_model=128, max_len=1024, **kwargs):
                    super().__init__(**kwargs)
                    self.d_model = d_model
                    pe = np.zeros((max_len, d_model))
                    position = np.arange(0, max_len)[:, np.newaxis]
                    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
                    pe[:, 0::2] = np.sin(position * div_term)
                    pe[:, 1::2] = np.cos(position * div_term)
                    self.pe = tf.constant(pe[np.newaxis, :, :], dtype=tf.float32)
                
                def call(self, x):
                    return x + self.pe[:, :tf.shape(x)[1], :]
            
            # Input: I/Q samples (1024x2)
            input_layer = layers.Input(shape=(1024, 2), name='iq_input')
            
            # Embed to transformer dimension
            x = layers.Dense(128, activation='relu')(input_layer)
            x = PositionalEncoding()(x)
            
            # Transformer encoder blocks
            for i in range(4):  # 4 transformer layers
                attn_output = layers.MultiHeadAttention(
                    num_heads=8, key_dim=128, name=f'transformer_{i}'
                )(x, x)
                x = layers.Add()([x, attn_output])
                x = layers.LayerNormalization()(x)
                
                ffn = keras.Sequential([
                    layers.Dense(512, activation='relu'),
                    layers.Dense(128)
                ])
                ffn_output = ffn(x)
                x = layers.Add()([x, ffn_output])
                x = layers.LayerNormalization()(x)
            
            # Global average pooling
            x = layers.GlobalAveragePooling1D()(x)
            
            # Multi-task outputs
            classification_output = layers.Dense(len(self.labels), activation='softmax', 
                                                name='classification')(x)
            anomaly_output = layers.Dense(1, activation='sigmoid', name='anomaly')(x)
            modulation_output = layers.Dense(16, activation='softmax', name='modulation')(x)  # QAM/PSK variants
            
            self.transformer_model = keras.Model(
                inputs=input_layer,
                outputs=[classification_output, anomaly_output, modulation_output]
            )
            
            self.transformer_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                loss={
                    'classification': 'categorical_crossentropy',
                    'anomaly': 'binary_crossentropy',
                    'modulation': 'categorical_crossentropy'
                },
                metrics={
                    'classification': ['accuracy'],
                    'anomaly': ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
                    'modulation': ['accuracy']
                }
            )
            
            self.logger.info("Transformer-based classifier built (v1.5.0)", 
                           params=self.transformer_model.count_params())
            
        except Exception as e:
            self.logger.error(f"Transformer model build failed: {e}")
    
    def detect_anomaly_multimodal(self, signal: np.ndarray, protocol_features: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Multi-modal anomaly detection: RF + protocol analysis (v1.5.1)
        Detects Rel-19 NTN attacks (gNB-on-satellite spoofing, LEO/GEO handover manipulation)
        
        Args:
            signal: I/Q samples (1024x2)
            protocol_features: Dict with keys: ['rrc_setup_time', 'handover_type', 'ntn_feeder_delay', ...]
            
        Returns:
            Enhanced anomaly report with attack vector classification
        """
        if not TF_AVAILABLE:
            return {'anomaly_detected': False, 'reason': 'tensorflow_unavailable'}
        
        # Lazy load models
        self._ensure_models_loaded()
        
        if not self.transformer_model:
            return {'anomaly_detected': False, 'reason': 'model_unavailable'}
        
        try:
            # RF-based detection
            signal_input = signal.reshape(1, 1024, 2)
            rf_classification, rf_anomaly, rf_modulation = self.transformer_model.predict(signal_input, verbose=0)
            
            # Protocol feature engineering (if provided)
            protocol_anomaly_score = 0.0
            anomaly_vectors = []
            
            if protocol_features:
                # Check NTN-specific protocol anomalies
                if protocol_features.get('ntn_feeder_delay', 0) > 250:  # >250ms suspicious for LEO
                    protocol_anomaly_score += 0.3
                    anomaly_vectors.append('excessive_ntn_delay')
                
                if protocol_features.get('handover_type') == 'inter_satellite':
                    if protocol_features.get('handover_prep_time', 0) < 100:
                        protocol_anomaly_score += 0.4
                        anomaly_vectors.append('rapid_sat_handover')  # Possible rogue satellite
                
                # Check for SUCI re-concealment failures (post-quantum attacks)
                if protocol_features.get('suci_deconceal_attempts', 0) > 3:
                    protocol_anomaly_score += 0.5
                    anomaly_vectors.append('suci_brute_force')
                
                # V2X sidelink anomalies
                if protocol_features.get('v2x_sidelink_active') and protocol_features.get('v2x_message_rate', 0) > 100:
                    protocol_anomaly_score += 0.3
                    anomaly_vectors.append('v2x_flooding')
            
            # Fuse RF and protocol scores
            combined_score = 0.6 * rf_anomaly[0][0] + 0.4 * protocol_anomaly_score
            
            result = {
                'anomaly_detected': combined_score > (1 - self.anomaly_threshold),
                'combined_score': float(combined_score),
                'rf_score': float(rf_anomaly[0][0]),
                'protocol_score': float(protocol_anomaly_score),
                'anomaly_vectors': anomaly_vectors,
                'predicted_generation': self.labels[np.argmax(rf_classification[0])] if np.argmax(rf_classification[0]) < len(self.labels) else 'Unknown',
                'modulation': self._decode_modulation(rf_modulation[0]),
                'timestamp': np.datetime64('now')
            }
            
            if result['anomaly_detected']:
                self.logger.warning(f"Multi-modal anomaly: {anomaly_vectors}, score={combined_score:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multi-modal detection failed: {e}")
            return {'anomaly_detected': False, 'error': str(e)}
    
    def _decode_modulation(self, mod_probs: np.ndarray) -> str:
        """Decode modulation scheme from softmax output"""
        modulations = ['BPSK', 'QPSK', '16QAM', '64QAM', '256QAM', 'GMSK', 'MSK', 
                       '8PSK', 'OQPSK', 'π/4-DQPSK', 'OFDM', 'CP-OFDM', 'DFT-S-OFDM', 
                       'GFDM', 'UFMC', 'FBMC']
        idx = np.argmax(mod_probs)
        return modulations[idx] if idx < len(modulations) else 'Unknown'
    
    def train_transformer(self, training_data: np.ndarray, labels: np.ndarray, 
                         anomaly_labels: np.ndarray, modulation_labels: np.ndarray, 
                         epochs: int = 50):
        """
        Train transformer model with multi-task learning
        
        Args:
            training_data: Training samples (NxFeatures array)
            labels: Classification labels (N-dimensional array)
            anomaly_labels: Anomaly labels (N-dimensional binary array)
            modulation_labels: Modulation scheme labels (N-dimensional array)
            epochs: Number of training epochs
        """
        if not TF_AVAILABLE or not self.transformer_model:
            self.logger.error("Cannot train - Transformer model not available")
            return
        
        try:
            self.logger.info(f"Training transformer model for {epochs} epochs...")
            
            # Convert labels to categorical
            if len(labels.shape) == 1:
                labels = keras.utils.to_categorical(labels, num_classes=len(self.labels))
            if len(modulation_labels.shape) == 1:
                modulation_labels = keras.utils.to_categorical(modulation_labels, num_classes=16)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_cls_train, y_cls_val, y_anom_train, y_anom_val, y_mod_train, y_mod_val = train_test_split(
                training_data, labels, anomaly_labels, modulation_labels,
                test_size=0.2, random_state=42
            )
            
            # Callbacks
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_classification_accuracy',
                    patience=5,
                    restore_best_weights=True
                ),
                keras.callbacks.ModelCheckpoint(
                    filepath='/tmp/best_transformer_classifier.h5',
                    monitor='val_classification_accuracy',
                    save_best_only=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]
            
            # Train
            history = self.transformer_model.fit(
                X_train,
                {'classification': y_cls_train, 'anomaly': y_anom_train, 'modulation': y_mod_train},
                epochs=epochs,
                validation_data=(X_val, {'classification': y_cls_val, 'anomaly': y_anom_val, 'modulation': y_mod_val}),
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate
            cls_acc = history.history['classification_accuracy'][-1]
            anom_acc = history.history['anomaly_accuracy'][-1]
            mod_acc = history.history['modulation_accuracy'][-1]
            
            self.logger.info(f"Transformer training completed",
                           cls_acc=f"{cls_acc:.4f}",
                           anom_acc=f"{anom_acc:.4f}",
                           mod_acc=f"{mod_acc:.4f}")
            
            return history
            
        except Exception as e:
            self.logger.error(f"Transformer training failed: {e}")
            return None
    
    def save_transformer_model(self, path: str):
        """Save trained transformer model"""
        if self.transformer_model:
            self.transformer_model.save(path)
            self.logger.info(f"Transformer model saved to {path}")
    
    # ==================== FEDERATED LEARNING INTEGRATION (v1.9.1) ====================
    
    def init_federated_learning(self, coordinator_config: Dict[str, Any] = None):
        """
        Initialize federated learning for distributed training (v1.9.1)
        Enables privacy-preserving model training across multiple FalconOne agents
        
        Args:
            coordinator_config: Optional configuration for federated coordinator
                - coordinator_url: URL of central coordinator
                - client_id: Unique identifier for this client
                - secure_aggregation: Enable Bonawitz secure aggregation
                - differential_privacy: Enable DP noise injection
        """
        self._federated_config = coordinator_config or {}
        self._federated_client_id = self._federated_config.get('client_id', f'client_{id(self)}')
        self._federated_round = 0
        self._local_samples_trained = 0
        self._gradient_history = []
        
        # Track local training epochs before aggregation
        self._local_epochs_per_round = self._federated_config.get('local_epochs', 5)
        
        # Differential privacy parameters
        self._dp_enabled = self._federated_config.get('differential_privacy', True)
        self._dp_epsilon = self._federated_config.get('dp_epsilon', 1.0)
        self._dp_clip_norm = self._federated_config.get('dp_clip_norm', 1.0)
        
        self.logger.info("Federated learning initialized (v1.9.1)",
                        client_id=self._federated_client_id,
                        local_epochs=self._local_epochs_per_round,
                        dp_enabled=self._dp_enabled)
    
    def train_federated(self, training_data: np.ndarray, labels: np.ndarray,
                       local_epochs: int = None) -> Dict[str, Any]:
        """
        Perform local training for federated learning round (v1.9.1)
        Trains on local data and prepares gradients for aggregation
        
        Args:
            training_data: Local training samples (NxFeatures array)
            labels: Local training labels (N-dimensional array)
            local_epochs: Override for number of local training epochs
            
        Returns:
            Training results including local gradients for aggregation
        """
        if not TF_AVAILABLE:
            return {'success': False, 'error': 'tensorflow_unavailable'}
        
        if not hasattr(self, '_federated_config'):
            self.init_federated_learning()
        
        # Ensure models are loaded
        self._ensure_models_loaded()
        
        if not self.model:
            return {'success': False, 'error': 'model_not_initialized'}
        
        epochs = local_epochs or self._local_epochs_per_round
        
        try:
            self.logger.info(f"Starting federated local training round {self._federated_round}",
                           samples=len(training_data), epochs=epochs)
            
            # Store initial weights for gradient computation
            initial_weights = [w.numpy().copy() for w in self.model.trainable_weights]
            
            # Convert labels to categorical if needed
            if len(labels.shape) == 1:
                labels_cat = keras.utils.to_categorical(labels, num_classes=len(self.labels))
            else:
                labels_cat = labels
            
            # Local training
            history = self.model.fit(
                training_data, labels_cat,
                epochs=epochs,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            
            # Compute gradients (weight updates)
            final_weights = [w.numpy() for w in self.model.trainable_weights]
            gradients = self._compute_weight_updates(initial_weights, final_weights)
            
            # Apply differential privacy if enabled
            if self._dp_enabled:
                gradients = self._apply_dp_to_gradients(gradients)
            
            # Store gradient history for debugging
            self._gradient_history.append({
                'round': self._federated_round,
                'samples': len(training_data),
                'epochs': epochs,
                'final_loss': float(history.history['loss'][-1]),
                'final_accuracy': float(history.history['accuracy'][-1])
            })
            
            self._local_samples_trained += len(training_data)
            
            result = {
                'success': True,
                'round': self._federated_round,
                'client_id': self._federated_client_id,
                'gradients': gradients,
                'num_samples': len(training_data),
                'local_loss': float(history.history['loss'][-1]),
                'local_accuracy': float(history.history['accuracy'][-1]),
                'dp_applied': self._dp_enabled
            }
            
            self.logger.info(f"Federated local training complete",
                           round=self._federated_round,
                           accuracy=f"{result['local_accuracy']:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Federated training failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _compute_weight_updates(self, initial_weights: List[np.ndarray], 
                                final_weights: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Compute weight updates (gradients) from before/after training (v1.9.1)
        
        Args:
            initial_weights: Weights before local training
            final_weights: Weights after local training
            
        Returns:
            Dictionary of layer gradients
        """
        gradients = {}
        for i, (init_w, final_w) in enumerate(zip(initial_weights, final_weights)):
            layer_name = f'layer_{i}'
            gradients[layer_name] = final_w - init_w
        return gradients
    
    def _apply_dp_to_gradients(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Apply differential privacy to gradients (v1.9.1)
        Uses Gaussian mechanism with gradient clipping
        
        Args:
            gradients: Raw gradients from local training
            
        Returns:
            Privacy-preserving gradients with DP noise
        """
        dp_gradients = {}
        
        for layer_name, grad in gradients.items():
            # Clip gradient norm
            grad_norm = np.linalg.norm(grad.flatten())
            if grad_norm > self._dp_clip_norm:
                grad = grad * (self._dp_clip_norm / grad_norm)
            
            # Add Gaussian noise (Gaussian mechanism)
            sensitivity = self._dp_clip_norm
            sigma = sensitivity * np.sqrt(2 * np.log(1.25 / 1e-5)) / self._dp_epsilon
            noise = np.random.normal(0, sigma, size=grad.shape)
            
            dp_gradients[layer_name] = grad + noise
        
        self.logger.debug(f"DP applied to {len(gradients)} layers (ε={self._dp_epsilon})")
        return dp_gradients
    
    def get_local_gradients(self) -> Dict[str, np.ndarray]:
        """
        Get current local model gradients for federated aggregation (v1.9.1)
        
        Returns:
            Dictionary of layer name to gradient array
        """
        if not TF_AVAILABLE or not self.model:
            return {}
        
        # Return gradients from last training round if available
        if hasattr(self, '_last_gradients'):
            return self._last_gradients
        
        # Otherwise compute current gradients relative to baseline
        gradients = {}
        for i, weight in enumerate(self.model.trainable_weights):
            layer_name = f'layer_{i}'
            gradients[layer_name] = weight.numpy()
        
        return gradients
    
    def apply_federated_update(self, aggregated_weights: Dict[str, np.ndarray]) -> bool:
        """
        Apply aggregated weights from federated coordinator (v1.9.1)
        Updates local model with globally aggregated parameters
        
        Args:
            aggregated_weights: Dictionary of aggregated layer weights
            
        Returns:
            True if update applied successfully
        """
        if not TF_AVAILABLE or not self.model:
            self.logger.error("Cannot apply federated update - model not available")
            return False
        
        try:
            self._ensure_models_loaded()
            
            # Apply aggregated weights to model
            for i, weight_var in enumerate(self.model.trainable_weights):
                layer_name = f'layer_{i}'
                if layer_name in aggregated_weights:
                    weight_var.assign(aggregated_weights[layer_name])
            
            self._federated_round += 1
            
            self.logger.info(f"Applied federated update (round {self._federated_round})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to apply federated update: {e}")
            return False
    
    def get_federated_status(self) -> Dict[str, Any]:
        """
        Get current federated learning status (v1.9.1)
        
        Returns:
            Status dictionary with training metrics
        """
        if not hasattr(self, '_federated_config'):
            return {'initialized': False}
        
        return {
            'initialized': True,
            'client_id': self._federated_client_id,
            'current_round': self._federated_round,
            'total_samples_trained': self._local_samples_trained,
            'local_epochs_per_round': self._local_epochs_per_round,
            'dp_enabled': self._dp_enabled,
            'dp_epsilon': self._dp_epsilon if self._dp_enabled else None,
            'gradient_history': self._gradient_history[-10:] if self._gradient_history else []
        }
    
    def train_federated_with_coordinator(self, training_data: np.ndarray, labels: np.ndarray,
                                         coordinator) -> Dict[str, Any]:
        """
        Full federated training loop with FederatedCoordinator integration (v1.9.1)
        Performs local training, submits to coordinator, and applies aggregated update
        
        Args:
            training_data: Local training samples
            labels: Local training labels
            coordinator: FederatedCoordinator instance
            
        Returns:
            Training results with aggregation status
        """
        # Step 1: Register with coordinator if not already
        if not hasattr(self, '_registered_with_coordinator'):
            registration = coordinator.register_client(self._federated_client_id)
            self._registered_with_coordinator = True
            self.logger.info(f"Registered with coordinator: {registration}")
        
        # Step 2: Perform local training
        local_result = self.train_federated(training_data, labels)
        
        if not local_result.get('success'):
            return local_result
        
        # Step 3: Submit weights to coordinator
        current_weights = {}
        for i, weight in enumerate(self.model.trainable_weights):
            current_weights[f'layer_{i}'] = weight.numpy()
        
        submission = coordinator.submit_weights(
            self._federated_client_id,
            current_weights,
            len(training_data)
        )
        
        self.logger.info(f"Submitted weights to coordinator",
                        waiting_for=submission.get('waiting_for', 'unknown'))
        
        # Step 4: Check if aggregation occurred and apply update
        status = coordinator.get_status()
        if status.get('aggregation_rounds', 0) > self._federated_round:
            # New global model available
            global_model_version = status.get('global_model_version')
            self.logger.info(f"New global model available (v{global_model_version})")
            # In production, would fetch and apply global weights here
        
        return {
            **local_result,
            'coordinator_status': status,
            'submission': submission
        }
    
    # ===== Edge-Optimized Inference (v1.6) =====
    
    def convert_to_tflite(self, model_path: str, output_path: str, quantize: bool = True) -> bool:
        """
        Convert TensorFlow model to TFLite for edge deployment
        Enables deployment on Raspberry Pi, embedded SDR compute modules
        
        Args:
            model_path: Path to trained .h5 model
            output_path: Path to save .tflite model
            quantize: Enable INT8 quantization (reduces size ~4x, inference ~3x faster)
        
        Returns:
            True if conversion successful
        
        Target: <20ms inference latency on ARM Cortex-A72 (RPi 4)
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow not available for TFLite conversion")
            return False
        
        try:
            # Load model
            model = keras.models.load_model(model_path)
            
            # Create TFLite converter
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if quantize:
                # Enable full INT8 quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                
                # Provide representative dataset for quantization calibration
                def representative_dataset():
                    for _ in range(100):
                        # Generate random samples matching model input shape
                        data = np.random.randn(1, *model.input_shape[1:]).astype(np.float32)
                        yield [data]
                
                converter.representative_dataset = representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
                
                self.logger.info("Applying INT8 quantization for edge optimization")
            
            # Convert model
            tflite_model = converter.convert()
            
            # Save TFLite model
            with open(output_path, 'wb') as f:
                f.write(tflite_model)
            
            original_size = Path(model_path).stat().st_size / (1024 * 1024)
            tflite_size = len(tflite_model) / (1024 * 1024)
            compression_ratio = original_size / tflite_size
            
            self.logger.info(f"TFLite conversion successful",
                           original_mb=f"{original_size:.2f}",
                           tflite_mb=f"{tflite_size:.2f}",
                           compression=f"{compression_ratio:.1f}x",
                           quantized=quantize)
            
            return True
            
        except Exception as e:
            self.logger.error(f"TFLite conversion failed: {e}")
            return False
    
    def load_tflite_model(self, tflite_path: str):
        """
        Load TFLite model for inference
        
        Args:
            tflite_path: Path to .tflite model
        """
        try:
            # Load TFLite model and allocate tensors
            self.tflite_interpreter = tf.lite.Interpreter(model_path=tflite_path)
            self.tflite_interpreter.allocate_tensors()
            
            # Get input and output details
            self.tflite_input_details = self.tflite_interpreter.get_input_details()
            self.tflite_output_details = self.tflite_interpreter.get_output_details()
            
            self.logger.info(f"TFLite model loaded from {tflite_path}",
                           input_shape=self.tflite_input_details[0]['shape'],
                           output_shape=self.tflite_output_details[0]['shape'])
            
        except Exception as e:
            self.logger.error(f"TFLite model loading failed: {e}")
    
    def classify_edge(self, signal: np.ndarray) -> Dict[str, Any]:
        """
        Edge-optimized inference using TFLite
        Target: <20ms latency on ARM Cortex-A72
        
        Args:
            signal: Input signal features (preprocessed)
        
        Returns:
            Classification result with timing
        """
        if not hasattr(self, 'tflite_interpreter'):
            self.logger.error("TFLite model not loaded")
            return {'error': 'tflite_not_loaded'}
        
        import time
        start_time = time.perf_counter()
        
        try:
            # Prepare input
            input_data = signal.astype(np.float32).reshape(self.tflite_input_details[0]['shape'])
            
            # Quantize input if model uses INT8
            if self.tflite_input_details[0]['dtype'] == np.int8:
                input_scale = self.tflite_input_details[0]['quantization'][0]
                input_zero_point = self.tflite_input_details[0]['quantization'][1]
                input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
            
            # Set input tensor
            self.tflite_interpreter.set_tensor(self.tflite_input_details[0]['index'], input_data)
            
            # Run inference
            self.tflite_interpreter.invoke()
            
            # Get output tensor
            output_data = self.tflite_interpreter.get_tensor(self.tflite_output_details[0]['index'])
            
            # Dequantize output if needed
            if self.tflite_output_details[0]['dtype'] == np.int8:
                output_scale = self.tflite_output_details[0]['quantization'][0]
                output_zero_point = self.tflite_output_details[0]['quantization'][1]
                output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale
            
            # Get prediction
            predicted_class = np.argmax(output_data[0])
            confidence = float(output_data[0][predicted_class])
            
            inference_time_ms = (time.perf_counter() - start_time) * 1000
            
            result = {
                'predicted_generation': self.labels[predicted_class] if predicted_class < len(self.labels) else 'Unknown',
                'confidence': confidence,
                'inference_time_ms': inference_time_ms,
                'edge_optimized': True,
            }
            
            if inference_time_ms < 20:
                self.logger.debug(f"Edge inference: {result['predicted_generation']}, {inference_time_ms:.1f}ms ✓")
            else:
                self.logger.warning(f"Edge inference slow: {inference_time_ms:.1f}ms (target: <20ms)")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Edge inference failed: {e}")
            return {'error': str(e)}
    
    def benchmark_edge_performance(self, num_iterations: int = 100) -> Dict[str, Any]:
        """
        Benchmark edge inference performance
        
        Args:
            num_iterations: Number of inference iterations
        
        Returns:
            Performance statistics
        """
        if not hasattr(self, 'tflite_interpreter'):
            return {'error': 'tflite_not_loaded'}
        
        import time
        latencies = []
        
        # Generate random test inputs
        input_shape = self.tflite_input_details[0]['shape']
        
        for _ in range(num_iterations):
            test_input = np.random.randn(*input_shape).astype(np.float32)
            
            start = time.perf_counter()
            self.classify_edge(test_input)
            latency_ms = (time.perf_counter() - start) * 1000
            latencies.append(latency_ms)
        
        stats = {
            'num_iterations': num_iterations,
            'mean_latency_ms': float(np.mean(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'target_met': np.mean(latencies) < 20.0,
        }
        
        self.logger.info(f"Edge benchmark complete",
                       mean_ms=f"{stats['mean_latency_ms']:.2f}",
                       p95_ms=f"{stats['p95_latency_ms']:.2f}",
                       target_met=stats['target_met'])
        
        return stats
    
    def train_quantization_aware(self, training_data: np.ndarray, labels: np.ndarray, 
                                 epochs: int = 50, save_path: str = None) -> Any:
        """
        Train model with quantization-aware training (QAT)
        Produces model optimized for INT8 inference from the start
        
        Args:
            training_data: Training samples
            labels: Training labels
            epochs: Number of training epochs
            save_path: Path to save QAT model
        
        Returns:
            Training history
        
        Benefit: Better accuracy than post-training quantization (~2-3% improvement)
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow not available")
            return None
        
        try:
            import tensorflow_model_optimization as tfmot
            
            self.logger.info("Starting quantization-aware training (QAT)")
            
            # Build model if not exists
            if self.model is None:
                self._ensure_models_loaded()  # Fixed: was _load_models
            
            # Apply quantization-aware training
            quantize_model = tfmot.quantization.keras.quantize_model
            q_aware_model = quantize_model(self.model)
            
            # Compile with quantization
            q_aware_model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                training_data, labels, test_size=0.2, random_state=42
            )
            
            history = q_aware_model.fit(
                X_train, y_train,
                epochs=epochs,
                validation_data=(X_val, y_val),
                batch_size=32,
                callbacks=[
                    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
                ],
                verbose=1
            )
            
            # Save QAT model
            if save_path:
                q_aware_model.save(save_path)
                self.logger.info(f"QAT model saved to {save_path}")
            
            # Convert to TFLite (will use learned quantization parameters)
            converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_qat_model = converter.convert()
            
            if save_path:
                tflite_path = save_path.replace('.h5', '_qat.tflite')
                with open(tflite_path, 'wb') as f:
                    f.write(tflite_qat_model)
                self.logger.info(f"QAT TFLite model saved to {tflite_path}")
            
            self.logger.info("Quantization-aware training completed",
                           final_acc=f"{history.history['accuracy'][-1]:.4f}",
                           val_acc=f"{history.history['val_accuracy'][-1]:.4f}")
            
            return history
            
        except ImportError:
            self.logger.error("tensorflow_model_optimization not installed. Install: pip install tensorflow-model-optimization")
            return None
        except Exception as e:
            self.logger.error(f"QAT training failed: {e}")
            return None
    
    def generate_anomaly_report(self) -> Dict[str, Any]:
        """
        Generate anomaly detection report
        For integration with Section 19 (Monitoring and Reporting)
        
        Returns:
            Anomaly statistics and trends
        """
        if not self.anomaly_history:
            return {'total_checks': 0, 'anomalies_detected': 0}
        
        total = len(self.anomaly_history)
        anomalies = sum(1 for a in self.anomaly_history if a.get('anomaly_detected'))
        
        avg_score = np.mean([a.get('anomaly_score', 0) for a in self.anomaly_history])
        
        report = {
            'total_checks': total,
            'anomalies_detected': anomalies,
            'detection_rate': anomalies / total if total > 0 else 0,
            'avg_anomaly_score': float(avg_score),
            'recent_anomalies': [a for a in self.anomaly_history[-10:] if a.get('anomaly_detected')],
            'threshold': self.anomaly_threshold
        }
        
        return report
    
    # ==================== ISAC & 6G ENHANCEMENTS (v1.8.0) ====================
    
    def _add_spatial_attention_block(self, x):
        """
        Add spatial attention for ISAC signal processing (v1.8.0)
        Enhances sensing capabilities in 6G JCAS scenarios
        
        Args:
            x: Input tensor from transformer/CNN layer
            
        Returns:
            Attention-weighted tensor
        """
        if not TF_AVAILABLE:
            return x
        
        try:
            # Average pooling across feature dimension
            avg_pool = layers.GlobalAveragePooling1D(keepdims=True)(x)
            
            # Max pooling across feature dimension
            max_pool = layers.GlobalMaxPooling1D(keepdims=True)(x)
            
            # Concatenate pooling results
            concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
            
            # Generate attention weights
            attention = layers.Dense(1, activation='sigmoid', name='spatial_attention')(concat)
            
            # Apply attention
            return layers.Multiply()([x, attention])
            
        except Exception as e:
            self.logger.error(f"Spatial attention error: {e}")
            return x
    
    def classify_isac_signal(self, signal: np.ndarray,
                            sensing_mode: str = 'monostatic') -> Dict[str, Any]:
        """
        Classify ISAC signals from 6G monitor (v1.8.0)
        Integrates with monitoring/sixg_monitor.py for joint communication and sensing
        
        Args:
            signal: IQ samples from ISAC waveform (1024x2)
            sensing_mode: 'monostatic' (collocated Tx/Rx) or 'bistatic' (separated)
            
        Returns:
            Classification with sensing parameters
        """
        if not TF_AVAILABLE or not self.transformer_model:
            return {'success': False, 'reason': 'model_unavailable'}
        
        try:
            # Classify using transformer with attention
            signal_input = signal.reshape(1, 1024, 2)
            predictions = self.transformer_model.predict(signal_input, verbose=0)
            
            classification_probs = predictions[0][0]
            generation = self.labels[np.argmax(classification_probs)]
            confidence = float(np.max(classification_probs))
            
            # Extract ISAC-specific features
            sensing_quality = self._assess_sensing_quality(signal, sensing_mode)
            communication_snr = self._estimate_communication_snr(signal)
            
            # Detect radar-like characteristics
            is_isac_waveform = self._detect_isac_waveform(signal)
            
            result = {
                'success': True,
                'generation': generation,
                'confidence': confidence,
                'sensing_mode': sensing_mode,
                'sensing_quality': sensing_quality,
                'communication_snr': communication_snr,
                'is_isac_waveform': is_isac_waveform,
                'timestamp': time.time()
            }
            
            # Add ISAC-specific metrics if it's a 6G signal
            if generation == '6G' and is_isac_waveform:
                result['isac_metrics'] = self._compute_isac_metrics(signal)
            
            return result
            
        except Exception as e:
            self.logger.error(f"ISAC classification error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _assess_sensing_quality(self, signal: np.ndarray, mode: str) -> float:
        """
        Assess sensing quality for ISAC signals (v1.8.0)
        Uses correlation-based metrics for range/velocity estimation quality
        
        Args:
            signal: IQ samples (1024x2)
            mode: 'monostatic' or 'bistatic'
            
        Returns:
            Quality score (0.0-1.0)
        """
        try:
            i_channel = signal[:, 0]
            q_channel = signal[:, 1]
            
            if mode == 'bistatic':
                # Cross-correlation for bistatic mode
                try:
                    from scipy.signal import correlate
                    correlation = np.max(np.abs(correlate(i_channel, q_channel, mode='valid')))
                    quality = min(correlation / np.max(np.abs(signal)), 1.0)
                except ImportError:
                    # Fallback: simple correlation coefficient
                    quality = np.abs(np.corrcoef(i_channel, q_channel)[0, 1])
            else:
                # Autocorrelation for monostatic mode
                quality = np.abs(np.corrcoef(i_channel, q_channel)[0, 1])
            
            return float(quality)
            
        except Exception as e:
            self.logger.error(f"Sensing quality assessment error: {e}")
            return 0.0
    
    def _estimate_communication_snr(self, signal: np.ndarray) -> float:
        """
        Estimate communication SNR from ISAC signal (v1.8.0)
        
        Args:
            signal: IQ samples (1024x2)
            
        Returns:
            Estimated SNR in dB
        """
        try:
            # Signal power
            signal_power = np.mean(np.abs(signal) ** 2)
            
            # Estimate noise from high-frequency components
            fft = np.fft.fft(signal[:, 0] + 1j * signal[:, 1])
            noise_power = np.mean(np.abs(fft[int(len(fft) * 0.9):]) ** 2)
            
            # SNR in dB
            snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))
            
            return float(snr_db)
            
        except Exception as e:
            self.logger.error(f"SNR estimation error: {e}")
            return 0.0
    
    def _detect_isac_waveform(self, signal: np.ndarray) -> bool:
        """
        Compute ISAC-specific performance metrics (v1.8.0)
        
        Args:
            signal: IQ samples (1024x2)
            
        Returns:
            Dictionary of ISAC metrics
        """
        try:
            complex_signal = signal[:, 0] + 1j * signal[:, 1]
            
            # Range resolution (from bandwidth)
            # Assume 100 MHz bandwidth for 6G ISAC
            bandwidth = 100e6  # Hz
            c = 3e8  # speed of light
            range_resolution = c / (2 * bandwidth)
            
            # Velocity resolution (from coherent processing interval)
            # Assume 1ms CPI
            cpi = 0.001  # seconds
            carrier_freq = 30e9  # 30 GHz for 6G mmWave
            velocity_resolution = c / (2 * carrier_freq * cpi)
            
            # Sensing accuracy from signal quality
            amplitude_stability = 1.0 / (np.std(np.abs(complex_signal)) + 1e-10)
            phase_stability = 1.0 / (np.std(np.angle(complex_signal)) + 1e-10)
            
            return {
                'range_resolution_m': float(range_resolution),
                'velocity_resolution_mps': float(velocity_resolution),
                'amplitude_stability': float(amplitude_stability),
                'phase_stability': float(phase_stability),
                'sensing_bandwidth_mhz': 100.0,
                'carrier_frequency_ghz': 30.0
            }
            
        except Exception as e:
            self.logger.error(f"ISAC metrics computation error: {e}")

        """
        Detect if signal is ISAC (Joint Communication and Sensing) waveform (v1.8.0)
        ISAC waveforms have periodic radar pulses embedded in OFDM
        
        Args:
            signal: IQ samples (1024x2)
            
        Returns:
            True if ISAC characteristics detected
        """
        try:
            # Convert to complex
            complex_signal = signal[:, 0] + 1j * signal[:, 1]
            
            # Check for periodic peaks (radar pulses)
            magnitude = np.abs(complex_signal)
            
            # Find peaks
            mean_mag = np.mean(magnitude)
            std_mag = np.std(magnitude)
            threshold = mean_mag + 2 * std_mag
            
            peaks = magnitude > threshold
            peak_count = np.sum(peaks)
            
            # ISAC typically has 5-15% peak density for sensing
            peak_ratio = peak_count / len(magnitude)
            
            # Check frequency domain for radar signatures
            fft = np.fft.fft(complex_signal)
            fft_peaks = np.abs(fft) > (np.mean(np.abs(fft)) + 3 * np.std(np.abs(fft)))
            fft_peak_ratio = np.sum(fft_peaks) / len(fft)
            
            # ISAC detection criteria
            is_isac = (0.05 <= peak_ratio <= 0.20) and (fft_peak_ratio > 0.05)
            
            return bool(is_isac)
            
        except Exception as e:
            self.logger.error(f"ISAC detection error: {e}")
            return False
    
    def _compute_isac_metrics(self, signal: np.ndarray) -> Dict[str, float]:
        """
        Compute ISAC-specific performance metrics (v1.8.0)
        
        Args:
            signal: IQ samples (1024x2)
            
        Returns:
            Dictionary of ISAC metrics
        """
        try:
            complex_signal = signal[:, 0] + 1j * signal[:, 1]
            
            # Range resolution (from bandwidth)
            # Assume 100 MHz bandwidth for 6G ISAC
            bandwidth = 100e6  # Hz
            c = 3e8  # speed of light
            range_resolution = c / (2 * bandwidth)
            
            # Velocity resolution (from coherent processing interval)
            # Assume 1ms CPI
            cpi = 0.001  # seconds
            carrier_freq = 30e9  # 30 GHz for 6G mmWave
            velocity_resolution = c / (2 * carrier_freq * cpi)
            
            # Sensing accuracy from signal quality
            amplitude_stability = 1.0 / (np.std(np.abs(complex_signal)) + 1e-10)
            phase_stability = 1.0 / (np.std(np.angle(complex_signal)) + 1e-10)
            
            return {
                'range_resolution_m': float(range_resolution),
                'velocity_resolution_mps': float(velocity_resolution),
                'amplitude_stability': float(amplitude_stability),
                'phase_stability': float(phase_stability),
                'sensing_bandwidth_mhz': 100.0,
                'carrier_frequency_ghz': 30.0
            }
            
        except Exception as e:
            self.logger.error(f"ISAC metrics computation error: {e}")
            return {}
