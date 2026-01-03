#!/usr/bin/env python3
"""
FalconOne Installation Script
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme = Path(__file__).parent / 'README.md'
long_description = readme.read_text(encoding='utf-8') if readme.exists() else ''

# Read requirements
requirements_file = Path(__file__).parent / 'requirements.txt'
requirements = []
if requirements_file.exists():
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f 
                       if line.strip() and not line.startswith('#')]

setup(
    name='falconone',
    version='1.9.0',
    description='Multi-Generation Cellular SIGINT Platform with Real-World Resilience Features',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='FalconOne Research Team',
    author_email='research@falconone.local',
    url='https://github.com/falconone/falconone-app',
    packages=find_packages(exclude=['tests', 'docs', 'examples']),
    python_requires='>=3.10',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.5.0'
        ],
        'quantum': [
            'qiskit>=0.45.0',
            'qiskit-aer>=0.13.0'
        ],
        'lattice': [
            'fpylll>=0.5.9'  # Optional: Production lattice reduction
        ]
    },
    entry_points={
        'console_scripts': [
            'falconone=falconone.cli.main:main',
        ],
    },
    include_package_data=True,
    package_data={
        'falconone': ['config/*.yaml'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Security',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: Other/Proprietary License',
        'Operating System :: POSIX :: Linux',
    ],
    keywords='sigint cellular 5g 6g imsi-catcher ml ai quantum',
    project_urls={
        'Documentation': 'https://github.com/falconone/falconone-app/blob/main/README.md',
        'Source': 'https://github.com/falconone/falconone-app',
        'Tracker': 'https://github.com/falconone/falconone-app/issues',
    },
)
