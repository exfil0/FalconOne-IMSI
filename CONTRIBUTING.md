# Contributing Guide

Development setup and contribution guidelines for FalconOne v1.9.6.

## Development Setup

### 1. Fork & Clone

```bash
git clone https://github.com/YOUR_USERNAME/FalconOne-IMSI.git
cd FalconOne-IMSI
```

### 2. Create Development Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
pip install -r requirements-dev.txt  # Dev dependencies
```

### 3. Install Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### 4. Verify Setup

```bash
pytest falconone/tests/ -v --tb=short
python validate_system.py
```

---

## Project Structure

```
FalconOne-IMSI/
├── falconone/              # Main package
│   ├── ai/                 # AI/ML modules
│   │   ├── signal_classifier.py
│   │   ├── ric_optimizer.py
│   │   └── ...
│   ├── core/               # Core orchestration
│   ├── crypto/             # Cryptographic modules
│   │   ├── post_quantum.py # PQC implementations
│   │   └── analyzer.py
│   ├── sdr/                # SDR device handling
│   ├── voice/              # Voice processing
│   │   └── interceptor.py
│   ├── tests/              # Unit tests
│   └── utils/              # Utilities
├── config/                 # Configuration files
├── docs/                   # Additional documentation
├── tests/                  # Integration tests
└── requirements.txt
```

---

## Coding Standards

### Python Style

- **PEP 8** compliance (enforced by flake8)
- **Type hints** for all function signatures
- **Docstrings** for all public classes/methods (Google style)

```python
def process_signal(
    samples: np.ndarray,
    sample_rate: float = 2.4e6,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Process IQ signal samples.
    
    Args:
        samples: Complex IQ samples as numpy array
        sample_rate: Sample rate in Hz (default: 2.4 MHz)
        config: Optional configuration override
    
    Returns:
        Dictionary containing:
            - classification: Signal type string
            - confidence: Float 0.0-1.0
            - features: Extracted feature dict
    
    Raises:
        ValueError: If samples array is empty
    """
    ...
```

### Import Order

```python
# Standard library
import os
import sys
from typing import Dict, List, Optional

# Third-party
import numpy as np
import tensorflow as tf

# Local
from falconone.utils.logger import ModuleLogger
from falconone.core.orchestrator import FalconOneOrchestrator
```

---

## Testing

### Run All Tests

```bash
pytest falconone/tests/ -v
```

### Run Specific Test Module

```bash
pytest falconone/tests/test_post_quantum.py -v
pytest falconone/tests/test_voice_interceptor.py -v
pytest falconone/tests/test_marl.py -v
```

### Run with Coverage

```bash
pytest falconone/tests/ --cov=falconone --cov-report=html
open htmlcov/index.html
```

### Test Categories

| Test File | Coverage |
|-----------|----------|
| `test_post_quantum.py` | Hybrid KEM, Hybrid Signatures, OQS, Quantum Sim |
| `test_voice_interceptor.py` | Opus, Diarization, VAD, Call Analysis |
| `test_marl.py` | Multi-Agent RL, SIGINTMultiAgentEnv |
| `test_ric_optimizer.py` | RIC DQN, Action Selection, Training |
| `test_data_validator.py` | IQ Validation, Sanitization |
| `test_signal_classifier.py` | AI Classification, Anomaly Detection |

### Writing Tests

```python
"""
Test module docstring with version and coverage info
"""
import pytest
from unittest.mock import Mock

@pytest.fixture
def mock_config():
    """Reusable mock configuration"""
    config = Mock()
    config.get = Mock(return_value=default_value)
    return config

class TestFeatureName:
    """Group related tests in classes"""
    
    def test_happy_path(self, mock_config):
        """Test normal operation"""
        result = function_under_test(mock_config)
        assert result.is_valid
    
    def test_edge_case(self, mock_config):
        """Test boundary conditions"""
        result = function_under_test(empty_input)
        assert result is None
    
    def test_error_handling(self, mock_config):
        """Test error conditions"""
        with pytest.raises(ValueError):
            function_under_test(invalid_input)
```

---

## Pull Request Process

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow coding standards
- Add/update tests
- Update documentation

### 3. Run Checks

```bash
# Lint
flake8 falconone/

# Type check (optional)
mypy falconone/

# Tests
pytest falconone/tests/ -v

# Validation
python validate_system.py
```

### 4. Commit

```bash
git add -A
git commit -m "feat: Add feature description

- Detail 1
- Detail 2

Closes #123"
```

#### Commit Message Format

```
<type>: <short description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `test`: Tests
- `refactor`: Code refactoring
- `perf`: Performance improvement
- `chore`: Maintenance

### 5. Push & Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub.

---

## Module Development Guide

### Adding a New AI Module

1. Create module in `falconone/ai/`
2. Inherit from appropriate base class
3. Add configuration options to `config.yaml.example`
4. Create test file in `falconone/tests/`
5. Update `__init__.py` exports

### Adding a New Codec

1. Add decoder class to `falconone/voice/codecs.py`
2. Register in `VoiceCodecManager`
3. Add to `AudioCodec` enum
4. Create tests in `test_voice_interceptor.py`

### Adding PQC Algorithm

1. Add to `OQSWrapper.SUPPORTED_*` dicts
2. Test with OQS library and simulator fallback
3. Update `QuantumAttackSimulator.validate_hybrid_scheme()`
4. Add tests to `test_post_quantum.py`

---

## Documentation

### Update Changelog

Add entry to `CHANGELOG.md`:

```markdown
## [1.9.7] - YYYY-MM-DD

### Added
- Feature description

### Fixed
- Bug fix description
```

### Docstring Requirements

- All public classes and methods
- Include Args, Returns, Raises sections
- Provide usage examples for complex APIs

---

## Security

### Reporting Vulnerabilities

**Do not open public issues for security vulnerabilities.**

Email: security@falconone-project.example

### Security Testing

```bash
# Run security scan
python falconone/tests/security_scan.py

# Check dependencies
pip-audit
safety check
```

---

## Getting Help

- **Issues**: GitHub Issues for bugs/features
- **Discussions**: GitHub Discussions for questions
- **Documentation**: See `docs/` folder

---

## License

By contributing, you agree that your contributions will be licensed under the project's license.
