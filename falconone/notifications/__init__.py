"""
Notification System
Email alerts and user notifications

Version 1.0: Phase 2.5.3 - Email Alerts
"""

from .email_alerts import EmailAlertManager, send_alert_email
from .alert_rules import AlertRulesEngine, AlertRule

__all__ = ['EmailAlertManager', 'send_alert_email', 'AlertRulesEngine', 'AlertRule']
