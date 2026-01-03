"""
FalconOne Async Monitoring Framework (v1.9.1)
=============================================
Provides async/await patterns for non-blocking monitoring operations.
Integrates CodeCarbon for carbon emissions tracking.

Features:
- AsyncMonitorBase: Abstract base class for async monitors
- AsyncEventLoop: Managed event loop for monitoring threads
- CodeCarbon integration for sustainability metrics
- Async context managers for resource management
- Background task scheduling with cancellation support
"""

import asyncio
import logging
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Coroutine
from datetime import datetime
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import weakref

try:
    from ..utils.logger import ModuleLogger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False
    class ModuleLogger:
        def __init__(self, name, parent):
            self.logger = logging.getLogger(name)
        def info(self, msg, **kw): self.logger.info(f"{msg} {kw if kw else ''}")
        def warning(self, msg, **kw): self.logger.warning(f"{msg} {kw if kw else ''}")
        def error(self, msg, **kw): self.logger.error(f"{msg} {kw if kw else ''}")
        def debug(self, msg, **kw): self.logger.debug(f"{msg} {kw if kw else ''}")

# CodeCarbon integration
try:
    from codecarbon import EmissionsTracker, OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False


@dataclass
class AsyncTaskMetrics:
    """Metrics for async task execution"""
    task_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    execution_count: int = 0
    error_count: int = 0
    total_duration_ms: float = 0.0
    last_error: Optional[str] = None
    carbon_emissions_kg: float = 0.0


@dataclass
class CarbonMetrics:
    """Carbon emissions tracking metrics"""
    total_emissions_kg: float = 0.0
    energy_consumed_kwh: float = 0.0
    duration_seconds: float = 0.0
    cpu_power_watts: float = 0.0
    gpu_power_watts: float = 0.0
    ram_power_watts: float = 0.0
    country_iso_code: str = "USA"
    region: str = ""
    cloud_provider: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class CarbonProfiler:
    """
    CodeCarbon integration for sustainability tracking (v1.9.1)
    Tracks carbon emissions for monitoring operations
    
    Supports:
    - Online tracking (with internet for grid carbon intensity)
    - Offline tracking (uses regional defaults)
    - Cloud provider detection (AWS, GCP, Azure)
    """
    
    def __init__(self, project_name: str = "falconone", 
                 offline: bool = True,
                 country_iso_code: str = "USA",
                 region: str = "",
                 logger: logging.Logger = None):
        """
        Initialize carbon profiler
        
        Args:
            project_name: Name for emissions tracking
            offline: Use offline tracking (no internet required)
            country_iso_code: ISO country code for grid carbon intensity
            region: Specific region/state for better accuracy
            logger: Logger instance
        """
        self.project_name = project_name
        self.offline = offline
        self.country_iso_code = country_iso_code
        self.region = region
        self.logger = ModuleLogger('CarbonProfiler', logger)
        
        self._tracker = None
        self._tracking = False
        self._total_emissions = 0.0
        self._session_metrics: List[CarbonMetrics] = []
        
        if CODECARBON_AVAILABLE:
            self.logger.info("CodeCarbon available for emissions tracking",
                           mode="offline" if offline else "online")
        else:
            self.logger.warning("CodeCarbon not installed - emissions tracking disabled")
    
    def start_tracking(self, task_name: str = "monitoring"):
        """Start emissions tracking for a task"""
        if not CODECARBON_AVAILABLE:
            return
        
        try:
            if self.offline:
                self._tracker = OfflineEmissionsTracker(
                    project_name=f"{self.project_name}_{task_name}",
                    country_iso_code=self.country_iso_code,
                    log_level="warning"
                )
            else:
                self._tracker = EmissionsTracker(
                    project_name=f"{self.project_name}_{task_name}",
                    log_level="warning"
                )
            
            self._tracker.start()
            self._tracking = True
            self.logger.debug(f"Started carbon tracking for '{task_name}'")
            
        except Exception as e:
            self.logger.error(f"Failed to start carbon tracking: {e}")
    
    def stop_tracking(self) -> Optional[CarbonMetrics]:
        """
        Stop emissions tracking and return metrics
        
        Returns:
            CarbonMetrics with emissions data or None if not tracking
        """
        if not CODECARBON_AVAILABLE or not self._tracking:
            return None
        
        try:
            emissions_kg = self._tracker.stop()
            self._tracking = False
            
            if emissions_kg is not None:
                self._total_emissions += emissions_kg
                
                # Get detailed metrics from tracker
                metrics = CarbonMetrics(
                    total_emissions_kg=emissions_kg,
                    energy_consumed_kwh=getattr(self._tracker, 'final_emissions_data', {}).get('energy_consumed', 0.0),
                    duration_seconds=getattr(self._tracker, 'final_emissions_data', {}).get('duration', 0.0),
                    country_iso_code=self.country_iso_code,
                    region=self.region
                )
                
                self._session_metrics.append(metrics)
                self.logger.info(f"Emissions tracked: {emissions_kg:.6f} kg CO2")
                
                return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to stop carbon tracking: {e}")
        
        return None
    
    @asynccontextmanager
    async def track_async(self, task_name: str = "async_task"):
        """
        Async context manager for emissions tracking
        
        Usage:
            async with profiler.track_async("my_task"):
                await do_work()
        """
        self.start_tracking(task_name)
        try:
            yield self
        finally:
            self.stop_tracking()
    
    def get_total_emissions(self) -> float:
        """Get total emissions for this session in kg CO2"""
        return self._total_emissions
    
    def get_session_report(self) -> Dict[str, Any]:
        """Get comprehensive session report"""
        return {
            'total_emissions_kg': self._total_emissions,
            'num_tracked_operations': len(self._session_metrics),
            'metrics_history': [m.__dict__ for m in self._session_metrics[-10:]],
            'codecarbon_available': CODECARBON_AVAILABLE
        }


class AsyncMonitorBase(ABC):
    """
    Abstract base class for async monitoring components (v1.9.1)
    Provides common async patterns for all monitors
    
    Subclasses must implement:
    - _capture_async(): Main capture coroutine
    - _process_sample(): Process captured samples
    """
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger = None):
        """
        Initialize async monitor
        
        Args:
            config: Monitor configuration
            logger: Logger instance
        """
        self.config = config
        self.logger = ModuleLogger(self.__class__.__name__, logger)
        
        # Async state
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Metrics
        self._metrics = AsyncTaskMetrics(task_name=self.__class__.__name__)
        
        # Carbon profiling
        self._carbon_profiler = CarbonProfiler(
            project_name=f"falconone_{self.__class__.__name__.lower()}",
            logger=logger
        )
        
        # Thread pool for blocking operations
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Sample queue for async processing
        self._sample_queue: asyncio.Queue = None
        
        # Background tasks registry
        self._background_tasks: weakref.WeakSet = weakref.WeakSet()
    
    async def start(self):
        """Start async monitoring"""
        if self._running:
            self.logger.warning("Monitor already running")
            return
        
        self._running = True
        self._sample_queue = asyncio.Queue(maxsize=1000)
        self._loop = asyncio.get_running_loop()
        
        self.logger.info("Starting async monitor")
        
        # Start main capture task
        self._task = asyncio.create_task(self._main_loop())
        
        # Start sample processor
        processor_task = asyncio.create_task(self._sample_processor_loop())
        self._background_tasks.add(processor_task)
    
    async def stop(self):
        """Stop async monitoring"""
        if not self._running:
            return
        
        self.logger.info("Stopping async monitor")
        self._running = False
        
        # Cancel main task
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # Cancel background tasks
        for task in list(self._background_tasks):
            if not task.done():
                task.cancel()
        
        # Shutdown executor
        self._executor.shutdown(wait=False)
        
        self.logger.info("Async monitor stopped",
                        total_captures=self._metrics.execution_count,
                        errors=self._metrics.error_count)
    
    async def _main_loop(self):
        """Main capture loop with error handling"""
        while self._running:
            try:
                start_time = time.perf_counter()
                
                # Run capture with carbon tracking
                async with self._carbon_profiler.track_async("capture"):
                    sample = await self._capture_async()
                
                if sample is not None:
                    await self._sample_queue.put(sample)
                
                # Update metrics
                duration_ms = (time.perf_counter() - start_time) * 1000
                self._metrics.execution_count += 1
                self._metrics.total_duration_ms += duration_ms
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._metrics.error_count += 1
                self._metrics.last_error = str(e)
                self.logger.error(f"Capture error: {e}")
                await asyncio.sleep(1.0)  # Backoff on error
    
    async def _sample_processor_loop(self):
        """Process samples from queue"""
        while self._running:
            try:
                # Wait for sample with timeout
                sample = await asyncio.wait_for(
                    self._sample_queue.get(),
                    timeout=1.0
                )
                
                # Process sample
                await self._process_sample(sample)
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Sample processing error: {e}")
    
    @abstractmethod
    async def _capture_async(self) -> Optional[Any]:
        """
        Capture samples asynchronously
        Must be implemented by subclasses
        
        Returns:
            Captured sample or None
        """
        pass
    
    @abstractmethod
    async def _process_sample(self, sample: Any):
        """
        Process a captured sample
        Must be implemented by subclasses
        
        Args:
            sample: Sample to process
        """
        pass
    
    async def run_blocking(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run blocking function in thread pool
        
        Args:
            func: Blocking function to run
            *args, **kwargs: Function arguments
            
        Returns:
            Function result
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: func(*args, **kwargs)
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        avg_duration = (
            self._metrics.total_duration_ms / self._metrics.execution_count
            if self._metrics.execution_count > 0 else 0.0
        )
        
        return {
            'task_name': self._metrics.task_name,
            'running': self._running,
            'execution_count': self._metrics.execution_count,
            'error_count': self._metrics.error_count,
            'error_rate': (
                self._metrics.error_count / self._metrics.execution_count
                if self._metrics.execution_count > 0 else 0.0
            ),
            'avg_duration_ms': avg_duration,
            'total_duration_ms': self._metrics.total_duration_ms,
            'last_error': self._metrics.last_error,
            'carbon_emissions_kg': self._carbon_profiler.get_total_emissions(),
            'queue_size': self._sample_queue.qsize() if self._sample_queue else 0
        }


class AsyncEventLoop:
    """
    Managed async event loop for monitoring threads (v1.9.1)
    Runs event loop in dedicated thread for non-blocking monitoring
    """
    
    def __init__(self, name: str = "monitor_loop", logger: logging.Logger = None):
        """
        Initialize managed event loop
        
        Args:
            name: Loop name for identification
            logger: Logger instance
        """
        self.name = name
        self.logger = ModuleLogger(f"AsyncLoop-{name}", logger)
        
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._monitors: Dict[str, AsyncMonitorBase] = {}
    
    def start(self):
        """Start the event loop in a dedicated thread"""
        if self._running:
            return
        
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=f"AsyncLoop-{self.name}",
            daemon=True
        )
        self._running = True
        self._thread.start()
        
        self.logger.info(f"Async event loop '{self.name}' started")
    
    def _run_loop(self):
        """Run event loop (called in thread)"""
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
    
    def stop(self):
        """Stop the event loop"""
        if not self._running:
            return
        
        self._running = False
        
        # Stop all monitors
        for name, monitor in self._monitors.items():
            asyncio.run_coroutine_threadsafe(monitor.stop(), self._loop)
        
        # Stop loop
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        
        self.logger.info(f"Async event loop '{self.name}' stopped")
    
    def add_monitor(self, name: str, monitor: AsyncMonitorBase):
        """
        Add and start a monitor
        
        Args:
            name: Monitor name
            monitor: AsyncMonitorBase instance
        """
        self._monitors[name] = monitor
        
        if self._running and self._loop:
            asyncio.run_coroutine_threadsafe(monitor.start(), self._loop)
            self.logger.info(f"Monitor '{name}' added and started")
    
    def remove_monitor(self, name: str):
        """
        Stop and remove a monitor
        
        Args:
            name: Monitor name
        """
        if name in self._monitors:
            monitor = self._monitors.pop(name)
            if self._running and self._loop:
                asyncio.run_coroutine_threadsafe(monitor.stop(), self._loop)
            self.logger.info(f"Monitor '{name}' removed")
    
    def run_coroutine(self, coro: Coroutine) -> asyncio.Future:
        """
        Run a coroutine in the event loop
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Future for the coroutine
        """
        if not self._running or not self._loop:
            raise RuntimeError("Event loop not running")
        
        return asyncio.run_coroutine_threadsafe(coro, self._loop)
    
    def get_status(self) -> Dict[str, Any]:
        """Get event loop status"""
        monitor_metrics = {}
        for name, monitor in self._monitors.items():
            monitor_metrics[name] = monitor.get_metrics()
        
        return {
            'name': self.name,
            'running': self._running,
            'thread_alive': self._thread.is_alive() if self._thread else False,
            'monitors': list(self._monitors.keys()),
            'monitor_metrics': monitor_metrics
        }


@asynccontextmanager
async def async_monitor_context(monitor: AsyncMonitorBase):
    """
    Async context manager for monitor lifecycle
    
    Usage:
        async with async_monitor_context(my_monitor) as m:
            await asyncio.sleep(10)  # Monitor runs for 10 seconds
    """
    await monitor.start()
    try:
        yield monitor
    finally:
        await monitor.stop()


async def gather_with_cancellation(*coros, timeout: float = None) -> List[Any]:
    """
    Gather coroutines with proper cancellation handling
    
    Args:
        *coros: Coroutines to gather
        timeout: Optional timeout in seconds
        
    Returns:
        List of results (or exceptions)
    """
    tasks = [asyncio.create_task(coro) for coro in coros]
    
    try:
        if timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks, return_exceptions=True),
                timeout=timeout
            )
        else:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
        
    except asyncio.TimeoutError:
        # Cancel all tasks on timeout
        for task in tasks:
            if not task.done():
                task.cancel()
        raise


class PeriodicTask:
    """
    Periodic async task with configurable interval (v1.9.1)
    """
    
    def __init__(self, func: Callable[[], Coroutine], 
                 interval_seconds: float,
                 name: str = "periodic_task",
                 logger: logging.Logger = None):
        """
        Initialize periodic task
        
        Args:
            func: Async function to call periodically
            interval_seconds: Interval between calls
            name: Task name
            logger: Logger instance
        """
        self.func = func
        self.interval = interval_seconds
        self.name = name
        self.logger = ModuleLogger(f"PeriodicTask-{name}", logger)
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._execution_count = 0
        self._last_execution: Optional[float] = None
    
    async def start(self):
        """Start periodic execution"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run())
        self.logger.info(f"Periodic task '{self.name}' started (interval={self.interval}s)")
    
    async def stop(self):
        """Stop periodic execution"""
        self._running = False
        
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"Periodic task '{self.name}' stopped after {self._execution_count} executions")
    
    async def _run(self):
        """Main periodic loop"""
        while self._running:
            try:
                start_time = time.perf_counter()
                
                await self.func()
                
                self._execution_count += 1
                self._last_execution = time.time()
                
                # Calculate sleep time accounting for execution duration
                elapsed = time.perf_counter() - start_time
                sleep_time = max(0, self.interval - elapsed)
                
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Periodic task error: {e}")
                await asyncio.sleep(self.interval)
    
    def get_status(self) -> Dict[str, Any]:
        """Get task status"""
        return {
            'name': self.name,
            'running': self._running,
            'interval_seconds': self.interval,
            'execution_count': self._execution_count,
            'last_execution': self._last_execution
        }


# Example implementation for reference
class ExampleAsyncMonitor(AsyncMonitorBase):
    """Example async monitor implementation"""
    
    async def _capture_async(self) -> Optional[Dict[str, Any]]:
        """Capture sample (example)"""
        await asyncio.sleep(0.1)  # Simulate async I/O
        return {
            'timestamp': time.time(),
            'value': 42.0
        }
    
    async def _process_sample(self, sample: Any):
        """Process sample (example)"""
        self.logger.debug(f"Processed sample: {sample}")
