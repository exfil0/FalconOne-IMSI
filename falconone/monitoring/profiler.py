"""
Profiling Dashboard Module - v1.7.0 Phase 1
===========================================
Production-grade observability with Prometheus/Grafana integration.

Features:
- Prometheus metric exporters (Gauge, Counter, Histogram)
- Grafana dashboard JSON templates
- Real-time latency tracking (p50, p95, p99 percentiles)
- Component-level accuracy metrics
- Resource utilization monitoring
"""

import time
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import defaultdict, deque
import threading
import logging
import numpy as np


@dataclass
class LatencyMetric:
    """Latency measurement with percentiles"""
    operation: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    count: int
    timestamp: str


@dataclass
class AccuracyMetric:
    """Accuracy measurement for ML/signal processing components"""
    component: str
    accuracy_percent: float
    precision_percent: float
    recall_percent: float
    f1_score: float
    sample_count: int
    timestamp: str


class PrometheusExporter:
    """
    Exports metrics in Prometheus format.
    
    Metric Types:
    - Gauge: Current value (CPU, memory, signal strength)
    - Counter: Cumulative count (packets processed, errors)
    - Histogram: Distribution (latency, accuracy)
    """
    
    def __init__(self, namespace: str = "falconone"):
        self.namespace = namespace
        self.gauges: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
        
        # Histogram buckets for latency (ms)
        self.latency_buckets = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000]
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Set gauge metric value."""
        with self._lock:
            label_str = self._format_labels(labels)
            key = f"{name}{label_str}"
            self.gauges[key] = value
    
    def increment_counter(self, name: str, value: int = 1, labels: Optional[Dict] = None):
        """Increment counter metric."""
        with self._lock:
            label_str = self._format_labels(labels)
            key = f"{name}{label_str}"
            self.counters[key] = self.counters.get(key, 0) + value
    
    def observe_histogram(self, name: str, value: float, labels: Optional[Dict] = None):
        """Add observation to histogram."""
        with self._lock:
            label_str = self._format_labels(labels)
            key = f"{name}{label_str}"
            self.histograms[key].append(value)
            
            # Keep only recent 1000 samples
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
    
    def _format_labels(self, labels: Optional[Dict]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in labels.items()]
        return "{" + ",".join(label_pairs) + "}"
    
    def export_text(self) -> str:
        """
        Export metrics in Prometheus text format.
        
        Returns:
            Prometheus-compatible metric dump
        """
        lines = []
        
        with self._lock:
            # Export gauges
            for key, value in self.gauges.items():
                metric_name = f"{self.namespace}_{key}"
                lines.append(f"# TYPE {metric_name} gauge")
                lines.append(f"{metric_name} {value}")
            
            # Export counters
            for key, value in self.counters.items():
                metric_name = f"{self.namespace}_{key}"
                lines.append(f"# TYPE {metric_name} counter")
                lines.append(f"{metric_name} {value}")
            
            # Export histograms with percentiles
            for key, values in self.histograms.items():
                if not values:
                    continue
                
                metric_name = f"{self.namespace}_{key}"
                lines.append(f"# TYPE {metric_name} histogram")
                
                # Calculate percentiles
                sorted_values = sorted(values)
                count = len(sorted_values)
                total = sum(sorted_values)
                
                # Add bucket counts
                for bucket in self.latency_buckets:
                    bucket_count = sum(1 for v in sorted_values if v <= bucket)
                    lines.append(f'{metric_name}_bucket{{le="{bucket}"}} {bucket_count}')
                
                # Add +Inf bucket
                lines.append(f'{metric_name}_bucket{{le="+Inf"}} {count}')
                lines.append(f"{metric_name}_sum {total}")
                lines.append(f"{metric_name}_count {count}")
        
        return "\n".join(lines) + "\n"
    
    def get_histogram_percentiles(self, name: str, labels: Optional[Dict] = None) -> Dict:
        """Calculate histogram percentiles."""
        with self._lock:
            label_str = self._format_labels(labels)
            key = f"{name}{label_str}"
            values = self.histograms.get(key, [])
            
            if not values:
                return {}
            
            sorted_values = sorted(values)
            count = len(sorted_values)
            
            return {
                "p50": sorted_values[int(count * 0.50)],
                "p95": sorted_values[int(count * 0.95)],
                "p99": sorted_values[int(count * 0.99)],
                "mean": np.mean(sorted_values),
                "max": max(sorted_values),
                "count": count
            }


class GrafanaTemplateGenerator:
    """Generates Grafana dashboard JSON templates."""
    
    def __init__(self, namespace: str = "falconone"):
        self.namespace = namespace
    
    def generate_dashboard(self) -> Dict:
        """
        Generate complete Grafana dashboard JSON.
        
        Returns:
            Grafana dashboard configuration
        """
        dashboard = {
            "dashboard": {
                "title": "FalconOne v1.7 Operations Dashboard",
                "tags": ["falconone", "5g", "security"],
                "timezone": "utc",
                "refresh": "5s",
                "panels": [
                    self._create_latency_panel(0),
                    self._create_throughput_panel(1),
                    self._create_accuracy_panel(2),
                    self._create_resource_panel(3),
                    self._create_error_panel(4),
                    self._create_cellular_panel(5)
                ]
            },
            "overwrite": True
        }
        return dashboard
    
    def _create_latency_panel(self, panel_id: int) -> Dict:
        """Create latency monitoring panel."""
        return {
            "id": panel_id,
            "title": "Operation Latency (p50, p95, p99)",
            "type": "graph",
            "gridPos": {"x": 0, "y": 0, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": f'histogram_quantile(0.50, rate({self.namespace}_operation_latency_bucket[5m]))',
                    "legendFormat": "p50",
                    "refId": "A"
                },
                {
                    "expr": f'histogram_quantile(0.95, rate({self.namespace}_operation_latency_bucket[5m]))',
                    "legendFormat": "p95",
                    "refId": "B"
                },
                {
                    "expr": f'histogram_quantile(0.99, rate({self.namespace}_operation_latency_bucket[5m]))',
                    "legendFormat": "p99",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {"format": "ms", "label": "Latency"},
                {"format": "short"}
            ]
        }
    
    def _create_throughput_panel(self, panel_id: int) -> Dict:
        """Create throughput monitoring panel."""
        return {
            "id": panel_id,
            "title": "Signal Processing Throughput",
            "type": "graph",
            "gridPos": {"x": 12, "y": 0, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": f'rate({self.namespace}_packets_processed_total[1m])',
                    "legendFormat": "Packets/sec",
                    "refId": "A"
                },
                {
                    "expr": f'rate({self.namespace}_signals_classified_total[1m])',
                    "legendFormat": "Signals/sec",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {"format": "ops", "label": "Throughput"},
                {"format": "short"}
            ]
        }
    
    def _create_accuracy_panel(self, panel_id: int) -> Dict:
        """Create ML accuracy monitoring panel."""
        return {
            "id": panel_id,
            "title": "Component Accuracy",
            "type": "graph",
            "gridPos": {"x": 0, "y": 8, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": f'{self.namespace}_component_accuracy{{component="signal_classifier"}}',
                    "legendFormat": "Signal Classifier",
                    "refId": "A"
                },
                {
                    "expr": f'{self.namespace}_component_accuracy{{component="ric_optimizer"}}',
                    "legendFormat": "RIC Optimizer",
                    "refId": "B"
                },
                {
                    "expr": f'{self.namespace}_component_accuracy{{component="crypto_analyzer"}}',
                    "legendFormat": "Crypto Analyzer",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {"format": "percent", "label": "Accuracy", "max": 100},
                {"format": "short"}
            ]
        }
    
    def _create_resource_panel(self, panel_id: int) -> Dict:
        """Create resource utilization panel."""
        return {
            "id": panel_id,
            "title": "Resource Utilization",
            "type": "graph",
            "gridPos": {"x": 12, "y": 8, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": f'{self.namespace}_cpu_percent',
                    "legendFormat": "CPU %",
                    "refId": "A"
                },
                {
                    "expr": f'{self.namespace}_memory_mb / 1024',
                    "legendFormat": "Memory GB",
                    "refId": "B"
                },
                {
                    "expr": f'{self.namespace}_gpu_utilization',
                    "legendFormat": "GPU %",
                    "refId": "C"
                }
            ],
            "yaxes": [
                {"format": "percent", "label": "Utilization"},
                {"format": "short"}
            ]
        }
    
    def _create_error_panel(self, panel_id: int) -> Dict:
        """Create error monitoring panel."""
        return {
            "id": panel_id,
            "title": "Error Rates",
            "type": "graph",
            "gridPos": {"x": 0, "y": 16, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": f'rate({self.namespace}_errors_total[5m])',
                    "legendFormat": "{{component}} errors/min",
                    "refId": "A"
                },
                {
                    "expr": f'rate({self.namespace}_warnings_total[5m])',
                    "legendFormat": "Warnings/min",
                    "refId": "B"
                }
            ],
            "yaxes": [
                {"format": "ops", "label": "Errors/min"},
                {"format": "short"}
            ]
        }
    
    def _create_cellular_panel(self, panel_id: int) -> Dict:
        """Create cellular monitoring panel."""
        return {
            "id": panel_id,
            "title": "Cellular Network Status",
            "type": "stat",
            "gridPos": {"x": 12, "y": 16, "w": 12, "h": 8},
            "targets": [
                {
                    "expr": f'{self.namespace}_cells_detected',
                    "legendFormat": "Cells Detected",
                    "refId": "A"
                },
                {
                    "expr": f'{self.namespace}_rogue_cells_detected',
                    "legendFormat": "Rogue Cells",
                    "refId": "B"
                }
            ],
            "options": {
                "colorMode": "value",
                "graphMode": "area"
            }
        }


class Profiler:
    """
    Main profiling coordinator with Prometheus export and Grafana integration.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        self.exporter = PrometheusExporter()
        self.grafana = GrafanaTemplateGenerator()
        
        # Latency tracking
        self.latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # Accuracy tracking
        self.accuracy_history: Dict[str, List[AccuracyMetric]] = defaultdict(list)
        
        self.logger.info("[Profiler] Production profiler initialized")
    
    def track_latency(self, operation: str, duration_ms: float):
        """Track operation latency."""
        self.exporter.observe_histogram(
            "operation_latency",
            duration_ms,
            labels={"operation": operation}
        )
        self.latency_history[operation].append(duration_ms)
    
    def track_accuracy(
        self,
        component: str,
        accuracy: float,
        precision: float = 0.0,
        recall: float = 0.0,
        f1: float = 0.0,
        sample_count: int = 0
    ):
        """Track component accuracy."""
        self.exporter.set_gauge(
            "component_accuracy",
            accuracy,
            labels={"component": component}
        )
        
        metric = AccuracyMetric(
            component=component,
            accuracy_percent=accuracy * 100,
            precision_percent=precision * 100,
            recall_percent=recall * 100,
            f1_score=f1,
            sample_count=sample_count,
            timestamp=datetime.utcnow().isoformat()
        )
        self.accuracy_history[component].append(metric)
        
        # Keep only recent 100 measurements
        if len(self.accuracy_history[component]) > 100:
            self.accuracy_history[component] = self.accuracy_history[component][-100:]
    
    def track_counter(self, name: str, value: int = 1, labels: Optional[Dict] = None):
        """Track counter metric."""
        self.exporter.increment_counter(name, value, labels)
    
    def track_gauge(self, name: str, value: float, labels: Optional[Dict] = None):
        """Track gauge metric."""
        self.exporter.set_gauge(name, value, labels)
    
    def get_latency_report(self, operation: Optional[str] = None) -> List[LatencyMetric]:
        """Get latency report for operation(s)."""
        report = []
        
        operations = [operation] if operation else self.latency_history.keys()
        
        for op in operations:
            percentiles = self.exporter.get_histogram_percentiles(
                "operation_latency",
                labels={"operation": op}
            )
            
            if percentiles:
                report.append(LatencyMetric(
                    operation=op,
                    p50_ms=percentiles["p50"],
                    p95_ms=percentiles["p95"],
                    p99_ms=percentiles["p99"],
                    mean_ms=percentiles["mean"],
                    max_ms=percentiles["max"],
                    count=percentiles["count"],
                    timestamp=datetime.utcnow().isoformat()
                ))
        
        return report
    
    def get_accuracy_report(self, component: Optional[str] = None) -> List[AccuracyMetric]:
        """Get accuracy report for component(s)."""
        if component:
            return self.accuracy_history.get(component, [])
        
        # Return all components' latest measurements
        report = []
        for comp, metrics in self.accuracy_history.items():
            if metrics:
                report.append(metrics[-1])
        
        return report
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format."""
        return self.exporter.export_text()
    
    def export_grafana_dashboard(self, filepath: Optional[str] = None) -> Dict:
        """
        Export Grafana dashboard JSON.
        
        Args:
            filepath: Optional path to save JSON file
            
        Returns:
            Dashboard configuration dict
        """
        dashboard = self.grafana.generate_dashboard()
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(dashboard, f, indent=2)
            self.logger.info(f"[Profiler] Grafana dashboard saved to {filepath}")
        
        return dashboard
    
    def get_statistics(self) -> Dict:
        """Get profiler statistics."""
        return {
            "tracked_operations": len(self.latency_history),
            "tracked_components": len(self.accuracy_history),
            "total_latency_measurements": sum(
                len(measurements) for measurements in self.latency_history.values()
            ),
            "total_accuracy_measurements": sum(
                len(metrics) for metrics in self.accuracy_history.values()
            )
        }
