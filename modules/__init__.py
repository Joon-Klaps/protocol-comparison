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

# Try to import Streamlit page management (optional)
try:
    from .main import ModulePageManager, get_global_page_manager, create_main_streamlit_app
    from .streamlit_base import PageRegistry, StreamlitPageComponent, PageConfig
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

__all__ = [
    'DataManager',
    'BaseAnalyzer',
    'ConsensusAnalyzer',
    'ConsensusDataManager',
    'CoverageAnalyzer',
    'CoverageDataManager',
    'ReadStatsAnalyzer'
]

# Add Streamlit components if available
if _STREAMLIT_AVAILABLE:
    __all__.extend([
        'ModulePageManager',
        'get_global_page_manager',
        'create_main_streamlit_app',
        'PageRegistry',
        'StreamlitPageComponent',
        'PageConfig'
    ])

__version__ = '0.1.0'