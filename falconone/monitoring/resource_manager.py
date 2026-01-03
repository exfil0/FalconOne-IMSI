"""
FalconOne Resource Manager
v1.9.4: Resource monitoring, auto-scaling, and throttling

Features:
- Real-time CPU/Memory/Disk monitoring
- Automatic garbage collection triggering
- Resource throttling for overload protection
- Process priority management
- Thread pool sizing
- Prometheus metrics integration
"""

import os
import gc
import sys
import time
import psutil
import logging
import threading
import queue
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import statistics

try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from ..utils.logger import ModuleLogger


class ResourceLevel(Enum):
    """Resource utilization levels"""
    LOW = "low"           # <30% utilization
    NORMAL = "normal"     # 30-70% utilization
    HIGH = "high"         # 70-85% utilization
    CRITICAL = "critical" # >85% utilization


class ThrottleAction(Enum):
    """Actions to take when throttling"""
    NONE = "none"
    REDUCE_WORKERS = "reduce_workers"
    PAUSE_PROCESSING = "pause_processing"
    FORCE_GC = "force_gc"
    SHED_LOAD = "shed_load"


@dataclass
class ResourceMetrics:
    """Current resource metrics snapshot"""
    timestamp: float
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_per_core: List[float] = field(default_factory=list)
    cpu_freq_mhz: float = 0.0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    
    # Memory metrics
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_total_mb: float = 0.0
    swap_percent: float = 0.0
    
    # Disk metrics
    disk_percent: float = 0.0
    disk_used_gb: float = 0.0
    disk_free_gb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    
    # Network metrics
    net_bytes_sent_mb: float = 0.0
    net_bytes_recv_mb: float = 0.0
    net_connections: int = 0
    
    # Process-specific
    process_memory_mb: float = 0.0
    process_cpu_percent: float = 0.0
    process_threads: int = 0
    process_fds: int = 0
    
    # GC metrics
    gc_collections: Dict[int, int] = field(default_factory=dict)
    gc_objects: int = 0


@dataclass
class ResourceThreshold:
    """Thresholds for resource alerts"""
    cpu_high: float = 70.0
    cpu_critical: float = 85.0
    memory_high: float = 70.0
    memory_critical: float = 85.0
    disk_high: float = 80.0
    disk_critical: float = 90.0
    
    # Response times
    gc_interval_normal_s: float = 300.0   # 5 minutes
    gc_interval_high_s: float = 60.0      # 1 minute
    gc_interval_critical_s: float = 10.0  # 10 seconds


class ResourceMonitor:
    """
    Monitors system and process resources with alerting.
    """
    
    def __init__(
        self,
        config,
        logger: logging.Logger,
        thresholds: ResourceThreshold = None
    ):
        self.config = config
        self.logger = ModuleLogger('ResourceMonitor', logger)
        self.thresholds = thresholds or ResourceThreshold()
        
        # Current process
        self._process = psutil.Process(os.getpid())
        
        # Monitoring state
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = config.get('resources.monitor_interval_s', 5.0)
        
        # Metrics history (for trend analysis)
        self._history: deque = deque(maxlen=100)
        self._lock = threading.Lock()
        
        # Callbacks
        self._on_threshold_exceeded: Optional[Callable[[str, float, float], None]] = None
        self._on_critical: Optional[Callable[[ResourceMetrics], None]] = None
        
        # Initialize Prometheus metrics
        self._init_prometheus_metrics()
        
        self.logger.info("Resource monitor initialized",
                        interval_s=self._monitor_interval)
    
    def _init_prometheus_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            self.metrics = {}
            return
        
        try:
            self.metrics = {
                'cpu_usage': Gauge(
                    'falconone_cpu_usage_percent',
                    'CPU usage percentage'
                ),
                'memory_usage': Gauge(
                    'falconone_memory_usage_percent',
                    'Memory usage percentage'
                ),
                'memory_used_mb': Gauge(
                    'falconone_memory_used_mb',
                    'Memory used in MB'
                ),
                'disk_usage': Gauge(
                    'falconone_disk_usage_percent',
                    'Disk usage percentage'
                ),
                'process_threads': Gauge(
                    'falconone_process_threads',
                    'Number of process threads'
                ),
                'gc_collections': Counter(
                    'falconone_gc_collections_total',
                    'Total garbage collections',
                    ['generation']
                ),
            }
        except Exception as e:
            self.logger.warning(f"Failed to initialize Prometheus metrics: {e}")
            self.metrics = {}
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(percpu=True)
            cpu_freq = psutil.cpu_freq()
            
            # Handle different OS for load average
            try:
                load_avg = os.getloadavg()
            except AttributeError:
                # Windows doesn't have getloadavg
                load_avg = (cpu_percent / 100, cpu_percent / 100, cpu_percent / 100)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Disk I/O
            try:
                disk_io = psutil.disk_io_counters()
                disk_read_mb = disk_io.read_bytes / (1024 * 1024)
                disk_write_mb = disk_io.write_bytes / (1024 * 1024)
            except:
                disk_read_mb = 0
                disk_write_mb = 0
            
            # Network metrics
            try:
                net_io = psutil.net_io_counters()
                net_connections = len(psutil.net_connections())
            except:
                net_io = None
                net_connections = 0
            
            # Process-specific metrics
            process_memory = self._process.memory_info()
            process_cpu = self._process.cpu_percent()
            
            try:
                process_fds = self._process.num_fds()
            except:
                process_fds = len(self._process.open_files()) if hasattr(self._process, 'open_files') else 0
            
            # GC metrics
            gc_stats = gc.get_stats()
            gc_collections = {i: gc_stats[i]['collections'] for i in range(len(gc_stats))}
            
            metrics = ResourceMetrics(
                timestamp=time.time(),
                
                # CPU
                cpu_percent=cpu_percent,
                cpu_per_core=cpu_per_core,
                cpu_freq_mhz=cpu_freq.current if cpu_freq else 0,
                load_average_1m=load_avg[0],
                load_average_5m=load_avg[1],
                load_average_15m=load_avg[2],
                
                # Memory
                memory_percent=memory.percent,
                memory_used_mb=memory.used / (1024 * 1024),
                memory_available_mb=memory.available / (1024 * 1024),
                memory_total_mb=memory.total / (1024 * 1024),
                swap_percent=swap.percent,
                
                # Disk
                disk_percent=disk.percent,
                disk_used_gb=disk.used / (1024 * 1024 * 1024),
                disk_free_gb=disk.free / (1024 * 1024 * 1024),
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                
                # Network
                net_bytes_sent_mb=(net_io.bytes_sent / (1024 * 1024)) if net_io else 0,
                net_bytes_recv_mb=(net_io.bytes_recv / (1024 * 1024)) if net_io else 0,
                net_connections=net_connections,
                
                # Process
                process_memory_mb=process_memory.rss / (1024 * 1024),
                process_cpu_percent=process_cpu,
                process_threads=self._process.num_threads(),
                process_fds=process_fds,
                
                # GC
                gc_collections=gc_collections,
                gc_objects=len(gc.get_objects())
            )
            
            # Update Prometheus metrics
            self._update_prometheus_metrics(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
            return ResourceMetrics(timestamp=time.time())
    
    def _update_prometheus_metrics(self, metrics: ResourceMetrics):
        """Update Prometheus metrics"""
        if not self.metrics:
            return
        
        try:
            self.metrics['cpu_usage'].set(metrics.cpu_percent)
            self.metrics['memory_usage'].set(metrics.memory_percent)
            self.metrics['memory_used_mb'].set(metrics.memory_used_mb)
            self.metrics['disk_usage'].set(metrics.disk_percent)
            self.metrics['process_threads'].set(metrics.process_threads)
        except Exception as e:
            self.logger.debug(f"Prometheus update error: {e}")
    
    def get_resource_level(self, metrics: ResourceMetrics = None) -> Dict[str, ResourceLevel]:
        """Get resource utilization levels"""
        if metrics is None:
            metrics = self.get_metrics()
        
        levels = {}
        
        # CPU level
        if metrics.cpu_percent < 30:
            levels['cpu'] = ResourceLevel.LOW
        elif metrics.cpu_percent < self.thresholds.cpu_high:
            levels['cpu'] = ResourceLevel.NORMAL
        elif metrics.cpu_percent < self.thresholds.cpu_critical:
            levels['cpu'] = ResourceLevel.HIGH
        else:
            levels['cpu'] = ResourceLevel.CRITICAL
        
        # Memory level
        if metrics.memory_percent < 30:
            levels['memory'] = ResourceLevel.LOW
        elif metrics.memory_percent < self.thresholds.memory_high:
            levels['memory'] = ResourceLevel.NORMAL
        elif metrics.memory_percent < self.thresholds.memory_critical:
            levels['memory'] = ResourceLevel.HIGH
        else:
            levels['memory'] = ResourceLevel.CRITICAL
        
        # Disk level
        if metrics.disk_percent < 50:
            levels['disk'] = ResourceLevel.LOW
        elif metrics.disk_percent < self.thresholds.disk_high:
            levels['disk'] = ResourceLevel.NORMAL
        elif metrics.disk_percent < self.thresholds.disk_critical:
            levels['disk'] = ResourceLevel.HIGH
        else:
            levels['disk'] = ResourceLevel.CRITICAL
        
        return levels
    
    def start(self):
        """Start continuous monitoring"""
        if self._running:
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="ResourceMonitor"
        )
        self._monitor_thread.start()
        self.logger.info("Resource monitoring started")
    
    def stop(self):
        """Stop monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self._running:
            try:
                metrics = self.get_metrics()
                
                with self._lock:
                    self._history.append(metrics)
                
                # Check thresholds
                self._check_thresholds(metrics)
                
                time.sleep(self._monitor_interval)
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(self._monitor_interval)
    
    def _check_thresholds(self, metrics: ResourceMetrics):
        """Check if thresholds are exceeded"""
        alerts = []
        
        if metrics.cpu_percent >= self.thresholds.cpu_critical:
            alerts.append(('cpu', metrics.cpu_percent, self.thresholds.cpu_critical))
        elif metrics.cpu_percent >= self.thresholds.cpu_high:
            alerts.append(('cpu', metrics.cpu_percent, self.thresholds.cpu_high))
        
        if metrics.memory_percent >= self.thresholds.memory_critical:
            alerts.append(('memory', metrics.memory_percent, self.thresholds.memory_critical))
        elif metrics.memory_percent >= self.thresholds.memory_high:
            alerts.append(('memory', metrics.memory_percent, self.thresholds.memory_high))
        
        if metrics.disk_percent >= self.thresholds.disk_critical:
            alerts.append(('disk', metrics.disk_percent, self.thresholds.disk_critical))
        
        # Trigger callbacks
        for resource, value, threshold in alerts:
            if self._on_threshold_exceeded:
                try:
                    self._on_threshold_exceeded(resource, value, threshold)
                except Exception as e:
                    self.logger.error(f"Threshold callback error: {e}")
        
        # Check for critical state
        levels = self.get_resource_level(metrics)
        if ResourceLevel.CRITICAL in levels.values():
            if self._on_critical:
                try:
                    self._on_critical(metrics)
                except Exception as e:
                    self.logger.error(f"Critical callback error: {e}")
    
    def get_history(self, duration_s: float = 300) -> List[ResourceMetrics]:
        """Get metrics history"""
        with self._lock:
            cutoff = time.time() - duration_s
            return [m for m in self._history if m.timestamp >= cutoff]
    
    def get_trends(self) -> Dict[str, float]:
        """Get resource usage trends (rate of change)"""
        history = self.get_history(60)  # Last minute
        
        if len(history) < 2:
            return {'cpu': 0, 'memory': 0, 'disk': 0}
        
        cpu_values = [m.cpu_percent for m in history]
        memory_values = [m.memory_percent for m in history]
        
        # Calculate rate of change (per minute)
        duration = history[-1].timestamp - history[0].timestamp
        if duration > 0:
            cpu_trend = (cpu_values[-1] - cpu_values[0]) / (duration / 60)
            memory_trend = (memory_values[-1] - memory_values[0]) / (duration / 60)
        else:
            cpu_trend = 0
            memory_trend = 0
        
        return {
            'cpu': cpu_trend,
            'memory': memory_trend,
            'cpu_avg': statistics.mean(cpu_values) if cpu_values else 0,
            'memory_avg': statistics.mean(memory_values) if memory_values else 0,
        }
    
    def on_threshold_exceeded(self, callback: Callable[[str, float, float], None]):
        """Register threshold exceeded callback"""
        self._on_threshold_exceeded = callback
    
    def on_critical(self, callback: Callable[[ResourceMetrics], None]):
        """Register critical state callback"""
        self._on_critical = callback


class ResourceThrottler:
    """
    Automatic resource throttling and load shedding.
    """
    
    def __init__(
        self,
        config,
        logger: logging.Logger,
        monitor: ResourceMonitor
    ):
        self.config = config
        self.logger = ModuleLogger('ResourceThrottler', logger)
        self.monitor = monitor
        
        # Throttling state
        self._throttle_active = False
        self._current_action = ThrottleAction.NONE
        self._throttle_start_time: Optional[float] = None
        
        # Managed resources
        self._thread_pools: List[Any] = []
        self._work_queues: List[queue.Queue] = []
        
        # GC management
        self._last_gc_time = 0
        self._gc_lock = threading.Lock()
        
        # Configuration
        self.auto_gc_enabled = config.get('resources.auto_gc', True)
        self.load_shedding_enabled = config.get('resources.load_shedding', True)
        
        # Register with monitor
        monitor.on_threshold_exceeded(self._handle_threshold)
        monitor.on_critical(self._handle_critical)
        
        self.logger.info("Resource throttler initialized",
                        auto_gc=self.auto_gc_enabled,
                        load_shedding=self.load_shedding_enabled)
    
    def register_thread_pool(self, pool: Any):
        """Register a thread pool for management"""
        self._thread_pools.append(pool)
    
    def register_work_queue(self, q: queue.Queue):
        """Register a work queue for load shedding"""
        self._work_queues.append(q)
    
    def _handle_threshold(self, resource: str, value: float, threshold: float):
        """Handle threshold exceeded event"""
        self.logger.warning(f"Resource threshold exceeded: {resource}={value:.1f}% (threshold={threshold}%)")
        
        if resource == 'memory':
            self._trigger_gc(force=True)
        
        if not self._throttle_active:
            self._start_throttling(ThrottleAction.REDUCE_WORKERS)
    
    def _handle_critical(self, metrics: ResourceMetrics):
        """Handle critical resource state"""
        self.logger.error("Critical resource state detected",
                        cpu=f"{metrics.cpu_percent:.1f}%",
                        memory=f"{metrics.memory_percent:.1f}%")
        
        # Force garbage collection
        self._trigger_gc(force=True, full=True)
        
        # Start aggressive throttling
        if self.load_shedding_enabled:
            self._start_throttling(ThrottleAction.SHED_LOAD)
    
    def _start_throttling(self, action: ThrottleAction):
        """Start throttling with specified action"""
        self._throttle_active = True
        self._current_action = action
        self._throttle_start_time = time.time()
        
        self.logger.warning(f"Throttling started: {action.value}")
        
        if action == ThrottleAction.REDUCE_WORKERS:
            self._reduce_workers()
        elif action == ThrottleAction.PAUSE_PROCESSING:
            self._pause_processing()
        elif action == ThrottleAction.SHED_LOAD:
            self._shed_load()
    
    def _reduce_workers(self):
        """Reduce thread pool workers"""
        for pool in self._thread_pools:
            if hasattr(pool, '_max_workers') and pool._max_workers > 1:
                # Reduce workers by 50%
                new_max = max(1, pool._max_workers // 2)
                self.logger.info(f"Reducing workers: {pool._max_workers} -> {new_max}")
                pool._max_workers = new_max
    
    def _pause_processing(self):
        """Pause work queue processing temporarily"""
        self.logger.info("Pausing queue processing for 5 seconds")
        time.sleep(5)
    
    def _shed_load(self):
        """Shed load from work queues"""
        for q in self._work_queues:
            shed_count = 0
            while q.qsize() > 100:  # Keep only 100 items
                try:
                    q.get_nowait()
                    shed_count += 1
                except queue.Empty:
                    break
            
            if shed_count > 0:
                self.logger.warning(f"Shed {shed_count} items from queue")
    
    def _trigger_gc(self, force: bool = False, full: bool = False):
        """Trigger garbage collection"""
        if not self.auto_gc_enabled and not force:
            return
        
        with self._gc_lock:
            now = time.time()
            
            # Rate limit GC
            if not force and now - self._last_gc_time < 10:
                return
            
            self._last_gc_time = now
            
            if full:
                # Full GC all generations
                gc.collect(2)
                gc.collect(1)
                gc.collect(0)
                self.logger.info("Full garbage collection triggered")
            else:
                # Just collect generation 0
                gc.collect(0)
                self.logger.debug("Garbage collection (gen 0) triggered")
    
    def stop_throttling(self):
        """Stop throttling"""
        self._throttle_active = False
        self._current_action = ThrottleAction.NONE
        
        duration = time.time() - self._throttle_start_time if self._throttle_start_time else 0
        self.logger.info(f"Throttling stopped after {duration:.1f}s")
        
        self._throttle_start_time = None
    
    def is_throttling(self) -> bool:
        """Check if currently throttling"""
        return self._throttle_active
    
    def get_status(self) -> Dict[str, Any]:
        """Get throttler status"""
        return {
            'throttle_active': self._throttle_active,
            'current_action': self._current_action.value,
            'throttle_duration_s': (
                time.time() - self._throttle_start_time
                if self._throttle_start_time else 0
            ),
            'managed_pools': len(self._thread_pools),
            'managed_queues': len(self._work_queues),
        }


class ResourceScaler:
    """
    Automatic resource scaling based on load.
    """
    
    def __init__(
        self,
        config,
        logger: logging.Logger,
        monitor: ResourceMonitor
    ):
        self.config = config
        self.logger = ModuleLogger('ResourceScaler', logger)
        self.monitor = monitor
        
        # Scaling configuration
        self.min_workers = config.get('resources.min_workers', 2)
        self.max_workers = config.get('resources.max_workers', 16)
        self.scale_up_threshold = config.get('resources.scale_up_threshold', 70)
        self.scale_down_threshold = config.get('resources.scale_down_threshold', 30)
        self.cooldown_s = config.get('resources.cooldown_s', 60)
        
        # State
        self._current_workers = self.min_workers
        self._last_scale_time = 0
        self._managed_pools: Dict[str, Any] = {}
        
        self.logger.info("Resource scaler initialized",
                        min_workers=self.min_workers,
                        max_workers=self.max_workers)
    
    def register_pool(self, name: str, pool: Any):
        """Register a thread pool for scaling"""
        self._managed_pools[name] = pool
        self.logger.debug(f"Registered pool: {name}")
    
    def evaluate_and_scale(self) -> str:
        """Evaluate current load and scale if needed"""
        # Check cooldown
        if time.time() - self._last_scale_time < self.cooldown_s:
            return "cooldown"
        
        metrics = self.monitor.get_metrics()
        trends = self.monitor.get_trends()
        
        # Determine scaling action
        avg_cpu = trends.get('cpu_avg', metrics.cpu_percent)
        
        if avg_cpu > self.scale_up_threshold and self._current_workers < self.max_workers:
            return self._scale_up()
        elif avg_cpu < self.scale_down_threshold and self._current_workers > self.min_workers:
            return self._scale_down()
        
        return "no_change"
    
    def _scale_up(self) -> str:
        """Scale up workers"""
        new_workers = min(self._current_workers + 2, self.max_workers)
        
        if new_workers == self._current_workers:
            return "at_max"
        
        self._current_workers = new_workers
        self._last_scale_time = time.time()
        
        self._apply_scaling()
        
        self.logger.info(f"Scaled UP to {new_workers} workers")
        return "scaled_up"
    
    def _scale_down(self) -> str:
        """Scale down workers"""
        new_workers = max(self._current_workers - 1, self.min_workers)
        
        if new_workers == self._current_workers:
            return "at_min"
        
        self._current_workers = new_workers
        self._last_scale_time = time.time()
        
        self._apply_scaling()
        
        self.logger.info(f"Scaled DOWN to {new_workers} workers")
        return "scaled_down"
    
    def _apply_scaling(self):
        """Apply scaling to managed pools"""
        for name, pool in self._managed_pools.items():
            if hasattr(pool, '_max_workers'):
                pool._max_workers = self._current_workers
                self.logger.debug(f"Set {name} workers to {self._current_workers}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scaler status"""
        return {
            'current_workers': self._current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'managed_pools': list(self._managed_pools.keys()),
            'last_scale_time': self._last_scale_time,
            'cooldown_remaining': max(
                0,
                self.cooldown_s - (time.time() - self._last_scale_time)
            ),
        }


class ResourceManager:
    """
    Unified resource management facade.
    """
    
    def __init__(self, config, logger: logging.Logger):
        self.config = config
        self.logger = ModuleLogger('ResourceManager', logger)
        
        # Initialize components
        self.monitor = ResourceMonitor(config, logger)
        self.throttler = ResourceThrottler(config, logger, self.monitor)
        self.scaler = ResourceScaler(config, logger, self.monitor)
        
        # Auto-scaling thread
        self._auto_scale_enabled = config.get('resources.auto_scale', True)
        self._auto_scale_thread: Optional[threading.Thread] = None
        self._running = False
        
        self.logger.info("Resource manager initialized")
    
    def start(self):
        """Start resource management"""
        self.monitor.start()
        
        if self._auto_scale_enabled:
            self._running = True
            self._auto_scale_thread = threading.Thread(
                target=self._auto_scale_loop,
                daemon=True,
                name="AutoScaler"
            )
            self._auto_scale_thread.start()
        
        self.logger.info("Resource management started")
    
    def stop(self):
        """Stop resource management"""
        self._running = False
        self.monitor.stop()
        
        if self._auto_scale_thread:
            self._auto_scale_thread.join(timeout=5)
        
        self.logger.info("Resource management stopped")
    
    def _auto_scale_loop(self):
        """Auto-scaling loop"""
        while self._running:
            try:
                result = self.scaler.evaluate_and_scale()
                if result not in ('no_change', 'cooldown'):
                    self.logger.info(f"Auto-scale result: {result}")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scale error: {e}")
                time.sleep(30)
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics"""
        return self.monitor.get_metrics()
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status"""
        metrics = self.monitor.get_metrics()
        levels = self.monitor.get_resource_level(metrics)
        
        return {
            'metrics': {
                'cpu_percent': metrics.cpu_percent,
                'memory_percent': metrics.memory_percent,
                'disk_percent': metrics.disk_percent,
                'process_memory_mb': metrics.process_memory_mb,
                'process_threads': metrics.process_threads,
            },
            'levels': {k: v.value for k, v in levels.items()},
            'throttler': self.throttler.get_status(),
            'scaler': self.scaler.get_status(),
            'trends': self.monitor.get_trends(),
        }
