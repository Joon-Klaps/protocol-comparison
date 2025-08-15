#!/usr/bin/env python3
"""
Package setup script for the viral genomics protocol comparison dashboard.

This is the actual setup.py for creating a pip-installable package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "readme.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    name="viral-genomics-dashboard",
    version="1.0.0",
    author="Joon Klaps",
    description="Clean Streamlit dashboard for viral genomics protocol comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Joon-Klaps/protocol-comparison",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,

    # Include package data (CSS, templates, etc.)
    include_package_data=True,
    package_data={
        '': ['*.css', '*.html', '*.js', '*.json', '*.yaml', '*.yml'],
        'modules': ['**/*.py', '**/*.css', '**/*.html', '**/*.js'],
        'sample_data': ['**/*'],
    },

    # Entry points for command-line scripts
    entry_points={
        'console_scripts': [
            'viral-dashboard=streamlit_app:main',
        ],
    },

    # Additional metadata
    keywords="bioinformatics, genomics, viral, streamlit, dashboard, analysis",
    project_urls={
        "Bug Reports": "https://github.com/Joon-Klaps/protocol-comparison/issues",
        "Source": "https://github.com/Joon-Klaps/protocol-comparison",
        "Documentation": "https://github.com/Joon-Klaps/protocol-comparison/blob/main/README.md",
    },
)
