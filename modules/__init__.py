"""
Protocol Comparison Analysis Modules

This package provides modular analysis components for viral genomics
protocol comparison studies including consensus analysis, coverage analysis,
and read statistics analysis.
"""

from .base import DataManager, BaseAnalyzer
from .consensus import ConsensusAnalyzer, ConsensusDataManager
from .coverage import CoverageAnalyzer, CoverageDataManager
from .read_stats import ReadStatsAnalyzer, ReadStatsDataManager

__all__ = [
    'DataManager',
    'BaseAnalyzer',
    'ConsensusAnalyzer',
    'ConsensusDataManager',
    'CoverageAnalyzer',
    'CoverageDataManager',
    'ReadStatsAnalyzer',
    'ReadStatsDataManager'
]

__version__ = '0.1.0'