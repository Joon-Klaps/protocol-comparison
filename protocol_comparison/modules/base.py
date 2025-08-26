"""
Base data manager for the viral genomics analysis platform.

This module provides a base class for managing data loading and caching
functionality across different analysis modules.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataManager(ABC):
    """
    Abstract base class for data management across analysis modules.
    Provides common functionality for data loading, caching, and validation.
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize the data manager.

        Args:
            data_path: Path to the data directory
        """
        self.data_path = Path(data_path)
        self._cache: Dict[str, Any] = {}
        self._validate_data_path()

    def _validate_data_path(self) -> None:
        """Validate that the data path exists and is accessible."""
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")

        if not self.data_path.is_dir():
            raise ValueError(f"Data path is not a directory: {self.data_path}")

    def get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if available."""
        return self._cache.get(key)

    def set_cache(self, key: str, value: Any) -> None:
        """Store data in cache."""
        self._cache[key] = value

    def clear_cache(self) -> None:
        """Clear all cached data."""
        self._cache.clear()

    @abstractmethod
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load data specific to the analysis module.
        Must be implemented by subclasses.

        Returns:
            Dictionary of loaded DataFrames
        """
        pass

    @abstractmethod
    def get_available_samples(self) -> List[str]:
        """
        Get list of available samples for analysis.
        Must be implemented by subclasses.

        Returns:
            List of sample identifiers
        """
        pass


class BaseAnalyzer(ABC):
    """
    Abstract base class for analysis modules.
    Provides common functionality for data analysis and visualization.
    """

    def __init__(self, data_manager: DataManager):
        """
        Initialize the analyzer with a data manager.

        Args:
            data_manager: Data manager instance for this analyzer
        """
        self.data_manager = data_manager
        self.data: Dict[str, pd.DataFrame] = {}
        self._load_data()

    def _load_data(self) -> None:
        """Load data using the data manager."""
        self.data = self.data_manager.load_data()

    def get_samples(self) -> List[str]:
        """Get available samples."""
        return self.data_manager.get_available_samples()

    @abstractmethod
    def generate_summary_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate summary statistics for the analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary of summary statistics
        """
        pass

    @abstractmethod
    def create_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create visualizations for the analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        pass
