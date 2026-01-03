"""
FalconOne Async Monitoring Framework (v1.9.2)
=============================================
Provides async/await patterns for non-blocking monitoring operations.
Integrates CodeCarbon for carbon emissions tracking with NTN energy estimation.

Features:
- AsyncMonitorBase: Abstract base class for async monitors
- AsyncEventLoop: Managed event loop for monitoring threads
- CodeCarbon integration for sustainability metrics
- NTN (Non-Terrestrial Network) energy estimation
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


# =============================================================================
# NTN Energy Estimator (v1.9.2)
# =============================================================================

@dataclass
class NTNEnergyProfile:
    """Energy profile for NTN simulation"""
    simulation_type: str  # 'leo_tracking', 'geo_coverage', 'constellation'
    num_satellites: int
    simulation_duration_s: float
    cpu_energy_wh: float
    gpu_energy_wh: float
    total_energy_wh: float
    co2_emissions_kg: float
    equivalent_ev_km: float  # Electric vehicle km equivalent
    equivalent_smartphone_charges: float


class NTNEnergyEstimator:
    """
    Energy estimation for NTN (Non-Terrestrial Network) simulations
    
    Extends CodeCarbon tracking with:
    - Simulation-specific energy models
    - Satellite constellation energy profiling
    - Hardware-aware power estimation
    - Sustainability metrics and recommendations
    
    Based on:
    - Intel/AMD CPU TDP profiles
    - NVIDIA GPU power consumption
    - Typical server workload patterns
    """
    
    # Hardware power profiles (typical values in Watts)
    HARDWARE_PROFILES = {
        'cpu_desktop': {'idle': 20, 'load': 95, 'tdp': 125},
        'cpu_laptop': {'idle': 5, 'load': 35, 'tdp': 45},
        'cpu_server': {'idle': 40, 'load': 150, 'tdp': 250},
        'gpu_consumer': {'idle': 15, 'load': 200, 'tdp': 350},
        'gpu_datacenter': {'idle': 30, 'load': 350, 'tdp': 450},
        'ram_per_gb': 0.3,  # Watts per GB
        'storage_ssd': 2,   # Watts
        'storage_hdd': 6,   # Watts
    }
    
    # NTN simulation complexity factors
    SIMULATION_COMPLEXITY = {
        'single_satellite': 1.0,
        'leo_constellation_small': 5.0,    # < 100 satellites
        'leo_constellation_large': 25.0,   # Starlink-scale
        'geo_coverage': 2.0,
        'multipath_propagation': 3.0,
        'doppler_simulation': 1.5,
        'ionospheric_model': 2.0,
        'full_link_budget': 4.0,
    }
    
    # Carbon intensity by region (kg CO2 per kWh)
    CARBON_INTENSITY = {
        'USA': 0.42,
        'EUR': 0.30,
        'CHN': 0.58,
        'IND': 0.72,
        'GBR': 0.23,
        'FRA': 0.06,  # High nuclear
        'DEU': 0.38,
        'JPN': 0.47,
        'BRA': 0.09,  # High hydro
        'CAN': 0.13,
        'AUS': 0.66,
        'DEFAULT': 0.45,
    }
    
    def __init__(self, 
                 hardware_profile: str = 'cpu_desktop',
                 country_code: str = 'USA',
                 logger: logging.Logger = None):
        """
        Initialize NTN energy estimator
        
        Args:
            hardware_profile: Type of hardware ('cpu_desktop', 'cpu_laptop', 'cpu_server')
            country_code: ISO country code for carbon intensity
            logger: Logger instance
        """
        self.hardware_profile = hardware_profile
        self.country_code = country_code
        self.logger = ModuleLogger('NTNEnergyEstimator', logger)
        
        self._carbon_profiler = CarbonProfiler(
            project_name="falconone_ntn",
            country_iso_code=country_code,
            logger=logger
        )
        
        # Tracking state
        self._simulations: List[NTNEnergyProfile] = []
        self._total_energy_wh = 0.0
        self._total_emissions_kg = 0.0
        
        self.logger.info("NTN Energy Estimator initialized",
                        hardware=hardware_profile,
                        region=country_code)
    
    def estimate_simulation_energy(self,
                                   simulation_type: str,
                                   num_satellites: int = 1,
                                   duration_seconds: float = 60.0,
                                   features: List[str] = None,
                                   gpu_enabled: bool = False,
                                   ram_gb: float = 8.0) -> NTNEnergyProfile:
        """
        Estimate energy consumption for an NTN simulation
        
        Args:
            simulation_type: Type of simulation
            num_satellites: Number of satellites in simulation
            duration_seconds: Simulation duration
            features: List of enabled features (complexity factors)
            gpu_enabled: Whether GPU is used
            ram_gb: RAM usage in GB
            
        Returns:
            NTNEnergyProfile with energy estimates
        """
        features = features or []
        
        # Base hardware power consumption
        hw = self.HARDWARE_PROFILES.get(self.hardware_profile, 
                                         self.HARDWARE_PROFILES['cpu_desktop'])
        
        # Calculate CPU power (interpolate between idle and load)
        # Assume 70% average utilization during simulation
        cpu_utilization = 0.7
        cpu_power_w = hw['idle'] + (hw['load'] - hw['idle']) * cpu_utilization
        
        # Apply complexity factor
        base_complexity = self.SIMULATION_COMPLEXITY.get(simulation_type, 1.0)
        
        # Scale by number of satellites (sublinear - O(n log n) typical)
        import math
        satellite_factor = 1 + math.log10(max(1, num_satellites))
        
        # Feature complexity
        feature_factor = 1.0
        for feature in features:
            feature_factor *= self.SIMULATION_COMPLEXITY.get(feature, 1.0)
        
        total_complexity = base_complexity * satellite_factor * feature_factor
        
        # Adjusted CPU energy
        effective_cpu_power = cpu_power_w * min(total_complexity, 10.0)  # Cap at 10x
        
        # GPU power if enabled
        gpu_power_w = 0
        if gpu_enabled:
            gpu_hw = self.HARDWARE_PROFILES['gpu_consumer']
            gpu_power_w = gpu_hw['idle'] + (gpu_hw['load'] - gpu_hw['idle']) * 0.5
        
        # RAM power
        ram_power_w = ram_gb * self.HARDWARE_PROFILES['ram_per_gb']
        
        # Total power
        total_power_w = effective_cpu_power + gpu_power_w + ram_power_w
        
        # Energy (Wh)
        duration_hours = duration_seconds / 3600
        cpu_energy_wh = effective_cpu_power * duration_hours
        gpu_energy_wh = gpu_power_w * duration_hours
        total_energy_wh = total_power_w * duration_hours
        
        # Carbon emissions
        carbon_intensity = self.CARBON_INTENSITY.get(
            self.country_code, 
            self.CARBON_INTENSITY['DEFAULT']
        )
        co2_emissions_kg = (total_energy_wh / 1000) * carbon_intensity
        
        # Equivalents for context
        # EV uses ~150 Wh/km
        ev_km = total_energy_wh / 150
        # Smartphone battery ~12 Wh
        smartphone_charges = total_energy_wh / 12
        
        profile = NTNEnergyProfile(
            simulation_type=simulation_type,
            num_satellites=num_satellites,
            simulation_duration_s=duration_seconds,
            cpu_energy_wh=cpu_energy_wh,
            gpu_energy_wh=gpu_energy_wh,
            total_energy_wh=total_energy_wh,
            co2_emissions_kg=co2_emissions_kg,
            equivalent_ev_km=ev_km,
            equivalent_smartphone_charges=smartphone_charges
        )
        
        self._simulations.append(profile)
        self._total_energy_wh += total_energy_wh
        self._total_emissions_kg += co2_emissions_kg
        
        self.logger.info(f"NTN simulation energy estimated",
                        type=simulation_type,
                        satellites=num_satellites,
                        energy_wh=f"{total_energy_wh:.3f}",
                        co2_kg=f"{co2_emissions_kg:.6f}")
        
        return profile
    
    def estimate_constellation_day(self,
                                   constellation_name: str,
                                   num_satellites: int,
                                   simulation_complexity: str = 'medium') -> Dict[str, Any]:
        """
        Estimate energy for a full day of constellation simulation
        
        Args:
            constellation_name: Name (e.g., 'Starlink', 'OneWeb')
            num_satellites: Number of satellites
            simulation_complexity: 'low', 'medium', 'high'
            
        Returns:
            Dictionary with daily energy projections
        """
        complexity_hours = {
            'low': 1.0,     # Simple tracking, 1 hour equivalent
            'medium': 4.0,  # Full orbit simulation
            'high': 12.0,   # High-fidelity with propagation
        }
        
        equiv_hours = complexity_hours.get(simulation_complexity, 4.0)
        duration_s = equiv_hours * 3600
        
        features = ['doppler_simulation', 'ionospheric_model']
        if simulation_complexity == 'high':
            features.extend(['multipath_propagation', 'full_link_budget'])
        
        profile = self.estimate_simulation_energy(
            simulation_type='leo_constellation_large' if num_satellites > 100 else 'leo_constellation_small',
            num_satellites=num_satellites,
            duration_seconds=duration_s,
            features=features,
            gpu_enabled=(simulation_complexity == 'high')
        )
        
        return {
            'constellation': constellation_name,
            'satellites': num_satellites,
            'complexity': simulation_complexity,
            'daily_energy_kwh': profile.total_energy_wh / 1000,
            'daily_co2_kg': profile.co2_emissions_kg,
            'monthly_co2_kg': profile.co2_emissions_kg * 30,
            'yearly_co2_kg': profile.co2_emissions_kg * 365,
            'yearly_trees_to_offset': (profile.co2_emissions_kg * 365) / 21,  # Avg tree absorbs 21kg/yr
        }
    
    @asynccontextmanager
    async def track_ntn_simulation(self, 
                                   simulation_type: str,
                                   num_satellites: int = 1):
        """
        Async context manager for tracking NTN simulation energy
        
        Usage:
            async with estimator.track_ntn_simulation('leo_tracking', 10):
                await run_simulation()
        """
        start_time = time.perf_counter()
        self._carbon_profiler.start_tracking(f"ntn_{simulation_type}")
        
        try:
            yield self
        finally:
            duration = time.perf_counter() - start_time
            carbon_metrics = self._carbon_profiler.stop_tracking()
            
            # Create profile with actual timing
            profile = self.estimate_simulation_energy(
                simulation_type=simulation_type,
                num_satellites=num_satellites,
                duration_seconds=duration
            )
            
            # Update with actual carbon data if available
            if carbon_metrics and carbon_metrics.total_emissions_kg > 0:
                profile.co2_emissions_kg = carbon_metrics.total_emissions_kg
    
    def get_sustainability_report(self) -> Dict[str, Any]:
        """Generate comprehensive sustainability report"""
        if not self._simulations:
            return {'message': 'No simulations tracked yet'}
        
        total_satellites = sum(s.num_satellites for s in self._simulations)
        total_duration = sum(s.simulation_duration_s for s in self._simulations)
        
        return {
            'summary': {
                'total_simulations': len(self._simulations),
                'total_satellites_simulated': total_satellites,
                'total_simulation_time_hours': total_duration / 3600,
                'total_energy_kwh': self._total_energy_wh / 1000,
                'total_co2_kg': self._total_emissions_kg,
            },
            'equivalents': {
                'ev_km_equivalent': self._total_energy_wh / 150,
                'smartphone_charges': self._total_energy_wh / 12,
                'trees_needed_to_offset': self._total_emissions_kg / 21,
                'average_home_days': self._total_energy_wh / 30000,  # ~30kWh/day avg
            },
            'recommendations': self._get_recommendations(),
            'by_simulation_type': self._aggregate_by_type(),
            'region': self.country_code,
            'carbon_intensity_kg_per_kwh': self.CARBON_INTENSITY.get(
                self.country_code, 
                self.CARBON_INTENSITY['DEFAULT']
            )
        }
    
    def _aggregate_by_type(self) -> Dict[str, Any]:
        """Aggregate energy by simulation type"""
        by_type: Dict[str, Dict] = {}
        
        for sim in self._simulations:
            if sim.simulation_type not in by_type:
                by_type[sim.simulation_type] = {
                    'count': 0,
                    'total_energy_wh': 0,
                    'total_co2_kg': 0,
                    'total_satellites': 0
                }
            
            by_type[sim.simulation_type]['count'] += 1
            by_type[sim.simulation_type]['total_energy_wh'] += sim.total_energy_wh
            by_type[sim.simulation_type]['total_co2_kg'] += sim.co2_emissions_kg
            by_type[sim.simulation_type]['total_satellites'] += sim.num_satellites
        
        return by_type
    
    def _get_recommendations(self) -> List[str]:
        """Generate sustainability recommendations"""
        recommendations = []
        
        if self._total_energy_wh > 1000:  # > 1 kWh
            recommendations.append(
                "Consider batch processing multiple simulations to reduce startup overhead"
            )
        
        if self.country_code in ['AUS', 'IND', 'CHN']:
            recommendations.append(
                "Your region has high carbon intensity. Consider scheduling simulations "
                "during off-peak hours when renewable energy percentage is higher"
            )
        
        if any(s.gpu_energy_wh > 0 for s in self._simulations):
            recommendations.append(
                "GPU acceleration detected. Use mixed-precision (FP16) computation "
                "where possible to reduce energy consumption by ~50%"
            )
        
        if self._total_emissions_kg > 1:
            recommendations.append(
                f"Consider carbon offsets: {self._total_emissions_kg:.2f} kg CO2 "
                f"can be offset for ~${self._total_emissions_kg * 0.02:.2f}"
            )
        
        if not recommendations:
            recommendations.append(
                "Your NTN simulations are within sustainable limits. Keep up the good work!"
            )
        
        return recommendations



# =============================================================================
# NTN Energy Estimator (v1.9.2)
# =============================================================================

@dataclass
class NTNEnergyProfile:
    """Energy profile for NTN simulation operations"""
    operation_name: str
    duration_seconds: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    memory_energy_kwh: float
    total_energy_kwh: float
    co2_emissions_kg: float
    equivalent_km_driven: float  # For context
    equivalent_smartphone_charges: float  # For context


@dataclass
class SatelliteSimulationConfig:
    """Configuration for NTN satellite simulation"""
    constellation_size: int = 100
    orbit_type: str = "LEO"  # LEO, MEO, GEO
    simulation_duration_hours: float = 24.0
    time_step_seconds: float = 1.0
    propagation_model: str = "sgp4"  # sgp4, j2, high_precision
    include_atmospheric: bool = True
    include_ionospheric: bool = True
    num_ground_stations: int = 10


class NTNEnergyEstimator:
    """
    Energy estimator for Non-Terrestrial Network (NTN) simulations (v1.9.2)
    
    Provides estimates for computational energy consumption of:
    - Satellite orbit propagation (SGP4, J2)
    - Signal path calculations (Doppler, delay, attenuation)
    - Multi-satellite visibility analysis
    - Handover simulations
    - Positioning algorithms (GDOP optimization)
    
    Uses CodeCarbon for actual measurements when available,
    with fallback to analytical estimates based on operation counts.
    """
    
    # Energy coefficients (empirically derived)
    # Joules per million floating-point operations
    ENERGY_PER_MFLOP_CPU = 0.01  # ~10 mJ/MFLOP for typical CPU
    ENERGY_PER_MFLOP_GPU = 0.001  # ~1 mJ/MFLOP for GPU
    
    # CO2 emission factors by region (kg CO2 per kWh)
    CO2_FACTORS = {
        'USA': 0.42,
        'EU': 0.30,
        'CHN': 0.58,
        'IND': 0.71,
        'GBR': 0.23,
        'DEU': 0.35,
        'FRA': 0.06,  # Nuclear
        'NOR': 0.02,  # Hydro
        'GLOBAL': 0.45
    }
    
    # Operation complexity estimates (MFLOP per operation)
    OPERATION_MFLOPS = {
        'sgp4_propagation': 0.05,  # Per satellite per time step
        'j2_propagation': 0.01,
        'high_precision_propagation': 0.5,
        'visibility_check': 0.001,  # Per satellite-station pair
        'doppler_calculation': 0.002,
        'ionospheric_delay': 0.005,
        'tropospheric_delay': 0.003,
        'gdop_calculation': 0.1,  # Per receiver position
        'handover_decision': 0.01,
        'signal_path_loss': 0.001,
        'link_budget': 0.01,
        'multipath_model': 0.1,
    }
    
    def __init__(self, country_iso_code: str = "USA",
                 use_gpu: bool = False,
                 logger: logging.Logger = None):
        """
        Initialize NTN energy estimator
        
        Args:
            country_iso_code: ISO country code for CO2 factor
            use_gpu: Whether GPU is used for computations
            logger: Logger instance
        """
        self.country_code = country_iso_code
        self.use_gpu = use_gpu
        self.logger = ModuleLogger('NTNEnergyEstimator', logger)
        
        self.co2_factor = self.CO2_FACTORS.get(country_iso_code, self.CO2_FACTORS['GLOBAL'])
        
        # CodeCarbon profiler for actual measurements
        self._profiler = CarbonProfiler(
            project_name="falconone_ntn",
            country_iso_code=country_iso_code,
            logger=logger
        )
        
        # History of estimates
        self._history: List[NTNEnergyProfile] = []
        
        self.logger.info("NTN energy estimator initialized",
                        country=country_iso_code,
                        co2_factor=self.co2_factor)
    
    def estimate_simulation_energy(self, config: SatelliteSimulationConfig) -> NTNEnergyProfile:
        """
        Estimate energy consumption for a satellite simulation
        
        Args:
            config: Simulation configuration
            
        Returns:
            NTNEnergyProfile with energy and emissions estimates
        """
        # Calculate total operations
        num_time_steps = int(config.simulation_duration_hours * 3600 / config.time_step_seconds)
        
        # Propagation operations
        prop_key = f"{config.propagation_model}_propagation"
        prop_mflops = self.OPERATION_MFLOPS.get(prop_key, 0.05)
        total_propagation_ops = config.constellation_size * num_time_steps * prop_mflops
        
        # Visibility calculations (each satellite to each ground station)
        visibility_ops = (config.constellation_size * config.num_ground_stations * 
                         num_time_steps * self.OPERATION_MFLOPS['visibility_check'])
        
        # Doppler calculations for visible satellites (assume 10% visible on average)
        visible_fraction = 0.1
        doppler_ops = (config.constellation_size * config.num_ground_stations * 
                      visible_fraction * num_time_steps * self.OPERATION_MFLOPS['doppler_calculation'])
        
        # Atmospheric corrections
        atmo_ops = 0
        if config.include_ionospheric:
            atmo_ops += (config.constellation_size * visible_fraction * 
                        num_time_steps * self.OPERATION_MFLOPS['ionospheric_delay'])
        if config.include_atmospheric:
            atmo_ops += (config.constellation_size * visible_fraction * 
                        num_time_steps * self.OPERATION_MFLOPS['tropospheric_delay'])
        
        # Total MFLOP
        total_mflops = total_propagation_ops + visibility_ops + doppler_ops + atmo_ops
        
        # Convert to energy
        energy_per_mflop = self.ENERGY_PER_MFLOP_GPU if self.use_gpu else self.ENERGY_PER_MFLOP_CPU
        total_energy_joules = total_mflops * energy_per_mflop
        total_energy_kwh = total_energy_joules / 3600000  # J to kWh
        
        # Estimate duration (assume 1 GFLOPS for CPU, 10 GFLOPS for GPU)
        compute_rate_mflops = 10000 if self.use_gpu else 1000
        estimated_duration = total_mflops / compute_rate_mflops
        
        # CO2 emissions
        co2_kg = total_energy_kwh * self.co2_factor
        
        # Context metrics
        km_driven = co2_kg / 0.12  # Average car: 0.12 kg CO2/km
        phone_charges = total_energy_kwh * 1000 / 0.01  # ~10 Wh per smartphone charge
        
        # Create profile
        profile = NTNEnergyProfile(
            operation_name=f"NTN_Simulation_{config.orbit_type}_{config.constellation_size}sat",
            duration_seconds=estimated_duration,
            cpu_energy_kwh=total_energy_kwh if not self.use_gpu else 0,
            gpu_energy_kwh=total_energy_kwh if self.use_gpu else 0,
            memory_energy_kwh=total_energy_kwh * 0.1,  # ~10% for memory
            total_energy_kwh=total_energy_kwh * 1.1,  # Include memory overhead
            co2_emissions_kg=co2_kg,
            equivalent_km_driven=km_driven,
            equivalent_smartphone_charges=phone_charges
        )
        
        self._history.append(profile)
        
        self.logger.info(f"Estimated NTN simulation energy: {total_energy_kwh*1000:.2f} Wh, "
                        f"CO2: {co2_kg*1000:.2f} g")
        
        return profile
    
    def estimate_operation_energy(self, operation: str, count: int = 1) -> Dict[str, float]:
        """
        Estimate energy for a specific operation type
        
        Args:
            operation: Operation name from OPERATION_MFLOPS
            count: Number of operations
            
        Returns:
            Dict with energy metrics
        """
        mflops_per_op = self.OPERATION_MFLOPS.get(operation, 0.01)
        total_mflops = mflops_per_op * count
        
        energy_per_mflop = self.ENERGY_PER_MFLOP_GPU if self.use_gpu else self.ENERGY_PER_MFLOP_CPU
        energy_joules = total_mflops * energy_per_mflop
        energy_kwh = energy_joules / 3600000
        
        return {
            'operation': operation,
            'count': count,
            'mflops': total_mflops,
            'energy_joules': energy_joules,
            'energy_kwh': energy_kwh,
            'co2_grams': energy_kwh * self.co2_factor * 1000
        }
    
    def estimate_constellation_analysis(self, constellation_size: int,
                                        num_ground_stations: int,
                                        duration_hours: float = 1.0) -> Dict[str, Any]:
        """
        Estimate energy for constellation coverage analysis
        
        Args:
            constellation_size: Number of satellites
            num_ground_stations: Number of ground stations
            duration_hours: Analysis duration
            
        Returns:
            Detailed energy breakdown
        """
        config = SatelliteSimulationConfig(
            constellation_size=constellation_size,
            num_ground_stations=num_ground_stations,
            simulation_duration_hours=duration_hours,
            time_step_seconds=60.0,  # 1-minute resolution for coverage
            propagation_model='sgp4',
            include_atmospheric=False,
            include_ionospheric=False
        )
        
        profile = self.estimate_simulation_energy(config)
        
        return {
            'profile': profile,
            'breakdown': {
                'propagation': self.estimate_operation_energy(
                    'sgp4_propagation', 
                    constellation_size * int(duration_hours * 60)
                ),
                'visibility': self.estimate_operation_energy(
                    'visibility_check',
                    constellation_size * num_ground_stations * int(duration_hours * 60)
                ),
                'gdop': self.estimate_operation_energy(
                    'gdop_calculation',
                    num_ground_stations * int(duration_hours * 60)
                )
            }
        }
    
    def measure_actual_energy(self, func: Callable, *args, **kwargs) -> Tuple[Any, Optional[NTNEnergyProfile]]:
        """
        Measure actual energy consumption of a function using CodeCarbon
        
        Args:
            func: Function to measure
            *args, **kwargs: Function arguments
            
        Returns:
            Tuple of (function result, energy profile or None)
        """
        self._profiler.start_tracking(func.__name__)
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            metrics = self._profiler.stop_tracking()
        
        duration = time.time() - start_time
        
        if metrics:
            profile = NTNEnergyProfile(
                operation_name=func.__name__,
                duration_seconds=duration,
                cpu_energy_kwh=metrics.energy_consumed_kwh,
                gpu_energy_kwh=0,  # Would need separate GPU tracking
                memory_energy_kwh=0,
                total_energy_kwh=metrics.energy_consumed_kwh,
                co2_emissions_kg=metrics.total_emissions_kg,
                equivalent_km_driven=metrics.total_emissions_kg / 0.12,
                equivalent_smartphone_charges=metrics.energy_consumed_kwh * 1000 / 0.01
            )
            self._history.append(profile)
            return result, profile
        
        return result, None
    
    def get_sustainability_report(self) -> Dict[str, Any]:
        """Generate sustainability report for all tracked operations"""
        if not self._history:
            return {'message': 'No operations tracked yet'}
        
        total_energy = sum(p.total_energy_kwh for p in self._history)
        total_co2 = sum(p.co2_emissions_kg for p in self._history)
        total_duration = sum(p.duration_seconds for p in self._history)
        
        return {
            'summary': {
                'total_operations': len(self._history),
                'total_duration_hours': total_duration / 3600,
                'total_energy_kwh': total_energy,
                'total_energy_wh': total_energy * 1000,
                'total_co2_kg': total_co2,
                'total_co2_grams': total_co2 * 1000,
                'equivalent_km_driven': total_co2 / 0.12,
                'equivalent_smartphone_charges': total_energy * 1000 / 0.01,
                'equivalent_tree_days': total_co2 / 0.022,  # Tree absorbs ~22g CO2/day
            },
            'by_operation': {
                p.operation_name: {
                    'energy_wh': p.total_energy_kwh * 1000,
                    'co2_grams': p.co2_emissions_kg * 1000,
                    'duration_s': p.duration_seconds
                }
                for p in self._history[-20:]  # Last 20 operations
            },
            'recommendations': self._get_sustainability_recommendations(total_energy, total_co2)
        }
    
    def _get_sustainability_recommendations(self, energy_kwh: float, co2_kg: float) -> List[str]:
        """Generate sustainability recommendations based on usage"""
        recommendations = []
        
        if self.use_gpu:
            recommendations.append("GPU acceleration is efficient for large simulations")
        else:
            recommendations.append("Consider GPU acceleration for >10x energy efficiency on large simulations")
        
        if self.co2_factor > 0.3:
            recommendations.append(f"Your region ({self.country_code}) has high carbon intensity. "
                                 "Consider scheduling heavy computations during off-peak hours.")
        
        if energy_kwh > 1.0:
            recommendations.append("For repetitive simulations, consider caching intermediate results")
        
        if co2_kg > 0.1:
            recommendations.append("Consider carbon offsetting for significant computational workloads")
        
        recommendations.append("Use lower time resolution (larger time steps) when high precision isn't required")
        
        return recommendations
    
    def compare_simulation_configs(self, configs: List[SatelliteSimulationConfig]) -> Dict[str, Any]:
        """
        Compare energy consumption across different simulation configurations
        
        Args:
            configs: List of configurations to compare
            
        Returns:
            Comparison report
        """
        profiles = [self.estimate_simulation_energy(config) for config in configs]
        
        return {
            'configurations': [
                {
                    'name': p.operation_name,
                    'energy_wh': p.total_energy_kwh * 1000,
                    'co2_grams': p.co2_emissions_kg * 1000,
                    'duration_s': p.duration_seconds
                }
                for p in profiles
            ],
            'most_efficient': min(profiles, key=lambda p: p.total_energy_kwh).operation_name,
            'least_efficient': max(profiles, key=lambda p: p.total_energy_kwh).operation_name,
            'energy_range_wh': (
                min(p.total_energy_kwh for p in profiles) * 1000,
                max(p.total_energy_kwh for p in profiles) * 1000
            )
        }

