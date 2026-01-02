"""
Celery Beat Schedule Configuration
Periodic tasks for automated scanning and monitoring

Version 1.0: Phase 2.5.2 - Scheduled Scanning
- Hourly frequency sweeps
- Daily network discovery
- Weekly security audits
- Continuous KPI monitoring
"""

from celery.schedules import crontab
from datetime import timedelta

# Celery Beat schedule configuration
beat_schedule = {
    # Task 2.5.2: Scheduled Scanning
    
    # Hourly frequency sweep (900-6000 MHz)
    'hourly-frequency-sweep': {
        'task': 'falconone.tasks.scan_frequency_range',
        'schedule': crontab(minute=0),  # Every hour at :00
        'args': (900.0, 6000.0, 10.0, 5.0, 'wideband'),
        'options': {
            'queue': 'scans',
            'priority': 5
        }
    },
    
    # Daily network discovery
    'daily-network-discovery': {
        'task': 'falconone.tasks.scan_network_discovery',
        'schedule': crontab(hour=2, minute=0),  # Every day at 2:00 AM
        'args': ('full',),
        'options': {
            'queue': 'scans',
            'priority': 7
        }
    },
    
    # Cell tower scan every 6 hours
    'cell-tower-scan-6h': {
        'task': 'falconone.tasks.scan_cell_towers',
        'schedule': crontab(minute=0, hour='*/6'),  # Every 6 hours
        'args': (37.7749, -122.4194, 20.0),  # Example: San Francisco area
        'kwargs': {'generations': ['LTE', '5G']},
        'options': {
            'queue': 'scans',
            'priority': 4
        }
    },
    
    # Continuous KPI monitoring (every 5 minutes)
    'kpi-monitoring-5min': {
        'task': 'falconone.tasks.collect_kpi_metrics',
        'schedule': timedelta(minutes=5),
        'options': {
            'queue': 'monitoring',
            'priority': 3
        }
    },
    
    # System health check (every 2 minutes)
    'health-check-2min': {
        'task': 'falconone.tasks.health_check',
        'schedule': timedelta(minutes=2),
        'options': {
            'queue': 'monitoring',
            'priority': 2
        }
    },
    
    # Weekly comprehensive scan (Sunday at 3:00 AM)
    'weekly-comprehensive-scan': {
        'task': 'falconone.tasks.scan_frequency_range',
        'schedule': crontab(hour=3, minute=0, day_of_week=0),  # Sunday 3 AM
        'args': (300.0, 6000.0, 1.0, 10.0, 'full'),
        'options': {
            'queue': 'scans',
            'priority': 9
        }
    },
    
    # Daily signal quality monitoring (every 4 hours)
    'signal-quality-4h': {
        'task': 'falconone.tasks.monitor_signal_quality',
        'schedule': crontab(minute=0, hour='*/4'),
        'args': (2100.0, 300),  # Monitor 2100 MHz for 5 minutes
        'options': {
            'queue': 'monitoring',
            'priority': 3
        }
    },
}

# Apply schedule to Celery app
def setup_beat_schedule(celery_app):
    """
    Apply beat schedule to Celery app
    
    Args:
        celery_app: Celery application instance
    """
    celery_app.conf.beat_schedule = beat_schedule
    return celery_app


# Custom schedule management
class ScheduleManager:
    """Manager for dynamic schedule configuration"""
    
    def __init__(self, celery_app):
        self.celery_app = celery_app
        self.custom_schedules = {}
    
    def add_scheduled_scan(self, name: str, frequency_range: tuple, 
                          schedule: str, scan_type: str = 'wideband'):
        """
        Add custom scheduled scan
        
        Args:
            name: Schedule name
            frequency_range: (start_freq, end_freq, step)
            schedule: Cron expression or timedelta
            scan_type: Scan type (wideband, narrowband, adaptive)
        """
        start_freq, end_freq, step = frequency_range
        
        self.custom_schedules[f'custom-scan-{name}'] = {
            'task': 'falconone.tasks.scan_frequency_range',
            'schedule': schedule,
            'args': (start_freq, end_freq, step, 10.0, scan_type),
            'options': {'queue': 'scans', 'priority': 5}
        }
        
        # Update Celery beat schedule
        self.celery_app.conf.beat_schedule.update(self.custom_schedules)
    
    def remove_scheduled_scan(self, name: str):
        """Remove custom scheduled scan"""
        key = f'custom-scan-{name}'
        if key in self.custom_schedules:
            del self.custom_schedules[key]
            if key in self.celery_app.conf.beat_schedule:
                del self.celery_app.conf.beat_schedule[key]
    
    def list_schedules(self) -> dict:
        """List all active schedules"""
        return self.celery_app.conf.beat_schedule
    
    def get_schedule(self, name: str) -> dict:
        """Get specific schedule configuration"""
        return self.celery_app.conf.beat_schedule.get(name)
