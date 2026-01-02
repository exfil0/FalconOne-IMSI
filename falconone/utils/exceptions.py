"""
FalconOne Exception Hierarchy
Specific exception types for better error handling and debugging
"""


class FalconOneException(Exception):
    """Base exception for all FalconOne errors"""
    pass


class ConfigurationError(FalconOneException):
    """Configuration file or parameter errors"""
    pass


class SafetyViolation(FalconOneException):
    """Safety check failures (Faraday cage, audit logging, power limits)"""
    pass


class SDRError(FalconOneException):
    """Hardware errors (device not found, connection lost, etc.)"""
    pass


class ExploitationError(FalconOneException):
    """Exploitation operation failures"""
    pass


class MonitoringError(FalconOneException):
    """Monitoring subsystem errors"""
    pass


class AIModelError(FalconOneException):
    """AI/ML model loading or inference errors"""
    pass


class CryptoAnalysisError(FalconOneException):
    """Cryptanalysis operation errors"""
    pass


class GeolocationError(FalconOneException):
    """Geolocation calculation errors"""
    pass


class IntegrationError(FalconOneException):
    """Cross-module integration errors"""
    pass
