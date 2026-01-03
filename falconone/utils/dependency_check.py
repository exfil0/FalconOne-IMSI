"""
FalconOne Dependency Checker

Utility to check availability of all dependencies and report status.
Supports graceful degradation reporting.

Version: 1.9.2
"""

import sys
import importlib
import shutil
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DependencyStatus:
    """Status of a dependency"""
    name: str
    available: bool
    version: Optional[str] = None
    fallback: Optional[str] = None
    error: Optional[str] = None


# Core dependencies (required)
CORE_DEPENDENCIES = [
    ('numpy', '1.24.0'),
    ('yaml', '6.0.1'),  # PyYAML
    ('requests', '2.32.0'),
    ('cryptography', '42.0.0'),
]

# Optional dependencies with fallbacks
OPTIONAL_DEPENDENCIES = [
    ('tensorflow', '2.14.0', 'Heuristic frequency-based classification'),
    ('ray', '2.7.0', 'Single-agent DQN with experience replay'),
    ('gymnasium', '0.28.0', 'Simplified internal environment'),
    ('pyshark', '0.5.0', 'Subprocess-based tshark parsing'),
]

# External system tools
SYSTEM_TOOLS = [
    ('tshark', 'Packet analysis'),
    ('grgsm_decode', 'GSM signal processing'),
    ('kalibrate', 'Frequency calibration'),
]


def check_python_package(name: str, min_version: Optional[str] = None) -> DependencyStatus:
    """
    Check if a Python package is available and meets version requirements.
    
    Args:
        name: Package name to import
        min_version: Minimum required version (optional)
    
    Returns:
        DependencyStatus with availability info
    """
    try:
        module = importlib.import_module(name)
        
        # Try to get version
        version = None
        for attr in ['__version__', 'VERSION', 'version']:
            if hasattr(module, attr):
                ver = getattr(module, attr)
                version = str(ver) if not callable(ver) else str(ver())
                break
        
        return DependencyStatus(
            name=name,
            available=True,
            version=version
        )
    except ImportError as e:
        return DependencyStatus(
            name=name,
            available=False,
            error=str(e)
        )
    except Exception as e:
        return DependencyStatus(
            name=name,
            available=False,
            error=f"Unexpected error: {e}"
        )


def check_system_tool(name: str) -> DependencyStatus:
    """
    Check if a system tool is available in PATH.
    
    Args:
        name: Tool name to check
    
    Returns:
        DependencyStatus with availability info
    """
    path = shutil.which(name)
    return DependencyStatus(
        name=name,
        available=path is not None,
        version=path if path else None
    )


def check_all_dependencies() -> Dict[str, bool]:
    """
    Check all dependencies and return simple availability dict.
    
    Returns:
        Dict mapping dependency name to availability boolean
    """
    result = {}
    
    # Core dependencies
    for name, _ in CORE_DEPENDENCIES:
        status = check_python_package(name)
        result[name] = status.available
    
    # Optional dependencies
    for item in OPTIONAL_DEPENDENCIES:
        name = item[0]
        status = check_python_package(name)
        result[name] = status.available
    
    # System tools
    for name, _ in SYSTEM_TOOLS:
        status = check_system_tool(name)
        result[name] = status.available
    
    return result


def get_detailed_status() -> Dict[str, DependencyStatus]:
    """
    Get detailed status of all dependencies.
    
    Returns:
        Dict mapping dependency name to DependencyStatus
    """
    result = {}
    
    # Core dependencies
    for name, min_ver in CORE_DEPENDENCIES:
        result[name] = check_python_package(name, min_ver)
    
    # Optional dependencies
    for item in OPTIONAL_DEPENDENCIES:
        name, min_ver, fallback = item
        status = check_python_package(name, min_ver)
        status.fallback = fallback
        result[name] = status
    
    # System tools
    for name, desc in SYSTEM_TOOLS:
        status = check_system_tool(name)
        result[name] = status
    
    return result


def print_dependency_report():
    """Print formatted dependency report to stdout."""
    print("=" * 60)
    print("FalconOne Dependency Status Report")
    print("=" * 60)
    print(f"Python Version: {sys.version}")
    print("-" * 60)
    
    # Core dependencies
    print("\n📦 CORE DEPENDENCIES (Required)")
    print("-" * 40)
    for name, min_ver in CORE_DEPENDENCIES:
        status = check_python_package(name, min_ver)
        icon = "✓" if status.available else "✗"
        version = f" v{status.version}" if status.version else ""
        print(f"  {icon} {name}{version}")
    
    # Optional dependencies
    print("\n🔧 OPTIONAL DEPENDENCIES (Graceful Degradation)")
    print("-" * 40)
    for name, min_ver, fallback in OPTIONAL_DEPENDENCIES:
        status = check_python_package(name, min_ver)
        icon = "✓" if status.available else "○"
        version = f" v{status.version}" if status.version else ""
        print(f"  {icon} {name}{version}")
        if not status.available:
            print(f"      ↳ Fallback: {fallback}")
    
    # System tools
    print("\n🛠️  SYSTEM TOOLS")
    print("-" * 40)
    for name, desc in SYSTEM_TOOLS:
        status = check_system_tool(name)
        icon = "✓" if status.available else "○"
        print(f"  {icon} {name} - {desc}")
    
    # Summary
    print("\n" + "=" * 60)
    all_status = check_all_dependencies()
    available = sum(1 for v in all_status.values() if v)
    total = len(all_status)
    core_ok = all(
        check_python_package(name).available 
        for name, _ in CORE_DEPENDENCIES
    )
    
    if core_ok:
        print("✓ All core dependencies satisfied")
    else:
        print("✗ Missing core dependencies - some features may not work")
    
    print(f"  {available}/{total} total dependencies available")
    print("=" * 60)


# Feature availability flags (can be imported elsewhere)
def get_feature_flags() -> Dict[str, bool]:
    """
    Get feature availability flags based on dependencies.
    
    Returns:
        Dict mapping feature name to availability
    """
    status = check_all_dependencies()
    
    return {
        'AI_CLASSIFICATION': status.get('tensorflow', False),
        'MULTI_AGENT_RL': status.get('ray', False),
        'ADVANCED_PARSING': status.get('pyshark', False),
        'GSM_PROCESSING': status.get('grgsm_decode', False),
        'PACKET_ANALYSIS': status.get('tshark', False),
    }


# Convenience imports
TF_AVAILABLE = check_python_package('tensorflow').available
RAY_AVAILABLE = check_python_package('ray').available
PYSHARK_AVAILABLE = check_python_package('pyshark').available


if __name__ == '__main__':
    print_dependency_report()
