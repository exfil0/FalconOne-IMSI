"""
FalconOne Celery Application Configuration
Task queue for asynchronous processing with Redis broker

Version 1.0: Phase 2.5.1 - Celery Task Queue
- Redis broker and result backend
- Task routing and rate limiting
- Task monitoring and status tracking
- Retry policies and error handling
"""

from celery import Celery
import os
from kombu import Queue, Exchange

# Celery configuration
CELERY_BROKER_URL = os.getenv('CELERY_BROKER_URL', 'redis://localhost:6379/0')
CELERY_RESULT_BACKEND = os.getenv('CELERY_RESULT_BACKEND', 'redis://localhost:6379/0')

# Create Celery app
celery_app = Celery(
    'falconone',
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=[
        'falconone.tasks.scan_tasks',
        'falconone.tasks.exploit_tasks',
        'falconone.tasks.monitoring_tasks'
    ]
)

# Celery configuration
celery_app.conf.update(
    # Task execution settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task result settings
    result_expires=3600,  # Results expire after 1 hour
    result_persistent=True,
    
    # Task routing
    task_routes={
        'falconone.tasks.scan_tasks.*': {'queue': 'scans'},
        'falconone.tasks.exploit_tasks.*': {'queue': 'exploits'},
        'falconone.tasks.monitoring_tasks.*': {'queue': 'monitoring'},
    },
    
    # Rate limiting
    task_annotations={
        'falconone.tasks.scan_tasks.scan_frequency_range': {'rate_limit': '10/m'},
        'falconone.tasks.exploit_tasks.execute_dos_attack': {'rate_limit': '5/m'},
        'falconone.tasks.exploit_tasks.execute_mitm_attack': {'rate_limit': '3/m'},
    },
    
    # Task execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=100,
    
    # Retry settings
    task_default_retry_delay=60,  # Retry after 60 seconds
    task_max_retries=3,
    
    # Beat scheduler settings (for Task 2.5.2)
    beat_scheduler='celery.beat:PersistentScheduler',
    beat_schedule_filename='/var/log/falconone/celerybeat-schedule',
)

# Task queues with priority
celery_app.conf.task_queues = (
    Queue('scans', Exchange('scans'), routing_key='scans', priority=5),
    Queue('exploits', Exchange('exploits'), routing_key='exploits', priority=8),
    Queue('monitoring', Exchange('monitoring'), routing_key='monitoring', priority=3),
    Queue('default', Exchange('default'), routing_key='default', priority=1),
)

# Task result backend settings
celery_app.conf.result_backend_transport_options = {
    'master_name': 'mymaster',
    'visibility_timeout': 3600,
}

if __name__ == '__main__':
    celery_app.start()
