"""
Protocol Comparison Analysis Modules

This package provides modular analysis components for viral genomics
protocol comparison studies including consensus analysis, coverage analysis,
and read statistics analysis.

It also provides a unified Streamlit page management system for creating
interactive web dashboards.
"""

from .base import DataManager, BaseAnalyzer
from .consensus import ConsensusAnalyzer, ConsensusDataManager
from .coverage import CoverageAnalyzer, CoverageDataManager
from .read_stats import ReadStatsAnalyzer

__all__ = [
    'DataManager',
    'BaseAnalyzer',
    'ConsensusAnalyzer',
    'ConsensusDataManager',
    'CoverageAnalyzer',
    'CoverageDataManager',
    'ReadStatsAnalyzer'
]

__version__ = '0.1.0'