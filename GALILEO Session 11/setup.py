#!/usr/bin/env python3
"""
Setup script for Geophysics Benchmarking Suite
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="geophysics-bench",
    version="1.0.0",
    description="Verification and Benchmarking Suite for Geophysical Gravity Gradiometry Processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Geophysics Platform Team",
    author_email="support@geophysics-platform.org",
    url="https://github.com/geophysics-platform/bench",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'pytest-cov>=4.1.0',
            'pytest-benchmark>=4.0.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
            'mypy>=1.0.0',
        ],
        'docs': [
            'sphinx>=6.0.0',
            'sphinx-rtd-theme>=1.2.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'geobench=bench:main',
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: GIS",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires='>=3.9',
    keywords='geophysics gravity benchmarking testing verification',
    project_urls={
        'Documentation': 'https://docs.geophysics-platform.org',
        'Source': 'https://github.com/geophysics-platform/bench',
        'Tracker': 'https://github.com/geophysics-platform/bench/issues',
    },
)
