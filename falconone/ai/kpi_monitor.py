"""
FalconOne KPI Monitoring Module
LSTM-based KPI prediction and anomaly detection
"""

import numpy as np
from typing import Dict, List, Any
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from ..utils.logger import ModuleLogger


class KPIMonitor:
    """LSTM-based KPI monitoring and prediction"""
    
    def __init__(self, config, logger: logging.Logger):
        """Initialize KPI monitor"""
        self.config = config
        self.logger = ModuleLogger('AI-KPIMonitor', logger)
        
        self.model = None
        self.kpi_history = []
        
        if TF_AVAILABLE:
            self._build_model()
            self.logger.info("KPI monitor initialized")
        else:
            self.logger.warning("TensorFlow not available - KPI monitoring disabled")
    
    def _build_model(self):
        """Build LSTM model for KPI prediction"""
        try:
            self.model = keras.Sequential([
                keras.layers.LSTM(64, return_sequences=True, input_shape=(10, 5)),
                keras.layers.LSTM(32),
                keras.layers.Dense(16, activation='relu'),
                keras.layers.Dense(5)  # Predict 5 KPIs
            ])
            
            self.model.compile(optimizer='adam', loss='mse')
            self.logger.info("LSTM KPI model built")
            
        except Exception as e:
            self.logger.error(f"Failed to build LSTM model: {e}")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update current KPI metrics
        
        Args:
            metrics: Dictionary of KPI values (rsrp, rsrq, sinr, throughput, latency)
        """
        self.kpi_history.append({
            'timestamp': np.datetime64('now'),
            'rsrp': metrics.get('rsrp', -100),
            'rsrq': metrics.get('rsrq', -10),
            'sinr': metrics.get('sinr', 0),
            'throughput_mbps': metrics.get('throughput', 0),
            'latency_ms': metrics.get('latency', 0)
        })
        
        # Keep only last 1000 samples
        if len(self.kpi_history) > 1000:
            self.kpi_history = self.kpi_history[-1000:]
        
        self.logger.debug(f"KPI metrics updated", samples=len(self.kpi_history))
    
    def predict(self, kpi_sequence: np.ndarray) -> np.ndarray:
        """Predict future KPIs using LSTM or statistical fallback
        
        Args:
            kpi_sequence: Sequence of KPI measurements (shape: [sequence_length, 5])
            
        Returns:
            Predicted KPI values for next timestep
        """
        if self.model and TF_AVAILABLE:
            return self.model.predict(kpi_sequence, verbose=0)
        
        # Fallback: Use exponential moving average
        if len(kpi_sequence) > 0:
            alpha = 0.3  # Smoothing factor
            ema = kpi_sequence[0]
            for i in range(1, len(kpi_sequence)):
                ema = alpha * kpi_sequence[i] + (1 - alpha) * ema
            return ema
        
        return np.zeros(5)  # Default prediction
    
    def detect_anomalies(self, threshold: float = 2.5) -> List[Dict[str, Any]]:
        """Detect anomalies in KPI history using statistical methods
        
        Args:
            threshold: Number of standard deviations for anomaly detection
            
        Returns:
            List of detected anomalies with timestamps and values
        """
        if len(self.kpi_history) < 30:
            return []  # Need sufficient history
        
        anomalies = []
        
        # Convert history to numpy array
        kpi_names = ['rsrp', 'rsrq', 'sinr', 'throughput_mbps', 'latency_ms']
        for kpi_name in kpi_names:
            values = np.array([h[kpi_name] for h in self.kpi_history])
            
            # Calculate rolling statistics (window=20)
            window = 20
            for i in range(window, len(values)):
                window_data = values[i-window:i]
                mean = np.mean(window_data)
                std = np.std(window_data)
                
                if std > 0:
                    z_score = abs((values[i] - mean) / std)
                    
                    if z_score > threshold:
                        anomalies.append({
                            'timestamp': self.kpi_history[i]['timestamp'],
                            'kpi': kpi_name,
                            'value': float(values[i]),
                            'expected': float(mean),
                            'z_score': float(z_score),
                            'severity': 'high' if z_score > 3.5 else 'medium'
                        })
        
        return anomalies
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of KPI history
        
        Returns:
            Dictionary of statistics for each KPI
        """
        if len(self.kpi_history) == 0:
            return {}
        
        kpi_names = ['rsrp', 'rsrq', 'sinr', 'throughput_mbps', 'latency_ms']
        stats = {}
        
        for kpi_name in kpi_names:
            values = np.array([h[kpi_name] for h in self.kpi_history])
            stats[kpi_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95)),
                'p99': float(np.percentile(values, 99))
            }
        
        return stats
