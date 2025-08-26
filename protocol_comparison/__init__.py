"""
Viral Genomics Protocol Comparison Dashboard

A modular Streamlit dashboard for comparing viral genomics analysis protocols.
"""

__version__ = "1.0.0"
__author__ = "Joon Klaps"

from .cli import main as cli_main

__all__ = ["cli_main", "__version__"]