"""
Coverage data management module.

This module provides data management functionality specifically for coverage analysis,
including loading coverage depth data and managing coverage-related datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from ..base import DataManager

logger = logging.getLogger(__name__)

# Global cache for coverage data managers to avoid duplicate loading
_coverage_data_cache = {}


def clear_coverage_cache():
    """Clear the global coverage data cache."""
    global _coverage_data_cache
    _coverage_data_cache = {}
    logger.info("Coverage data cache cleared")


class CoverageDataManager(DataManager):
    """
    Data manager specifically for coverage analysis.

    Handles loading and management of coverage depth data from parquet files,
    providing access to coverage statistics and genome recovery calculations.
    Uses caching to avoid reloading data multiple times.
    """

    def __init__(self, data_path: Path):
        """
        Initialize coverage data manager with caching.

        Args:
            data_path: Path to the data directory containing coverage files
        """
        super().__init__(data_path)
        self.depth_dir = self.data_path / "custom_vcfs"
        
        # Use cache key based on data path
        self.cache_key = str(self.depth_dir.absolute())
        
        # Check if we already have cached data for this path
        if self.cache_key in _coverage_data_cache:
            logger.debug("Using cached coverage data for %s", self.cache_key)
            self._nested_data = _coverage_data_cache[self.cache_key]['nested_data']
            self._flat_data = _coverage_data_cache[self.cache_key]['flat_data']
        else:
            self._nested_data = None
            self._flat_data = None
            
        self._validate_coverage_data_path()

    def _validate_coverage_data_path(self) -> None:
        """Validate that the coverage data directory exists."""
        if not self.depth_dir.exists():
            logger.warning("Coverage depth directory does not exist: %s", self.depth_dir)
        elif not self.depth_dir.is_dir():
            raise ValueError(f"Coverage depth path is not a directory: {self.depth_dir}")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load coverage-related data files from parquet format.
        Implementation of abstract method from base class.

        Returns:
            Dictionary containing coverage data DataFrames with flattened structure
        """
        if self._flat_data is None:
            logger.info("Loading coverage data for the first time for %s", self.cache_key)
            nested_data = self._load_data()
            
            # Flatten the nested structure to match the base class interface
            flat_data = {}
            for sample_id, references in nested_data.items():
                for reference, df in references.items():
                    key = f"{sample_id}_{reference}"
                    flat_data[key] = df
                    
            # Cache both nested and flat data
            _coverage_data_cache[self.cache_key] = {
                'nested_data': nested_data,
                'flat_data': flat_data
            }
            self._nested_data = nested_data
            self._flat_data = flat_data
        else:
            logger.debug("Using cached flat coverage data for %s", self.cache_key)
                
        return self._flat_data

    def _load_data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Load coverage-related data files from parquet format.

        Returns:
            Nested dictionary containing coverage data DataFrames:
            {sample_id: {reference: DataFrame}}
        """
        data = {}

        if not self.depth_dir.exists():
            logger.warning("Coverage depth directory not found: %s", self.depth_dir)
            return data

        # Load coverage depth data from parquet files
        depth_files = list(self.depth_dir.glob("*.parquet"))
        logger.info("Found %d coverage depth files in %s", len(depth_files), self.depth_dir)

        for depth_file in depth_files:
            try:
                # Parse filename to extract sample_id and reference
                # Expected format: {sample_id}_{reference}.parquet
                filename_parts = depth_file.stem.split("_")
                if len(filename_parts) < 2:
                    logger.warning("Skipping file with unexpected format: %s", depth_file.name)
                    continue

                sample_id = filename_parts[0]
                reference = "_".join(filename_parts[1:])  # Join remaining parts as reference

                # Load DataFrame
                df = pd.read_parquet(depth_file, engine='pyarrow')
                df['sample_id'] = sample_id

                # Initialize nested dictionary structure
                if sample_id not in data:
                    data[sample_id] = {}

                data[sample_id][reference] = df
                logger.debug("Loaded coverage data for sample %s, reference %s", sample_id, reference)

            except (pd.errors.ParserError, FileNotFoundError, PermissionError) as e:
                logger.warning("Could not load coverage file %s: %s", depth_file, e)

        logger.info("Successfully loaded coverage data for %d samples", len(data))
        return data

    @property
    def data(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Lazily loads and returns the coverage data.

        Returns:
            Nested dictionary with coverage data for all samples and references
        """
        if self._nested_data is None:
            logger.info("Loading nested coverage data for the first time for %s", self.cache_key)
            self._nested_data = self._load_data()
            
            # Also update the cache
            if self.cache_key not in _coverage_data_cache:
                # If flat data also needs to be cached
                flat_data = {}
                for sample_id, references in self._nested_data.items():
                    for reference, df in references.items():
                        key = f"{sample_id}_{reference}"
                        flat_data[key] = df
                        
                _coverage_data_cache[self.cache_key] = {
                    'nested_data': self._nested_data,
                    'flat_data': flat_data
                }
                self._flat_data = flat_data
        else:
            logger.debug("Using cached nested coverage data for %s", self.cache_key)
            
        return self._nested_data

    def get_available_samples(self) -> List[str]:
        """
        Get available sample IDs from coverage data.

        Returns:
            Sorted list of sample identifiers
        """
        return sorted(list(self.data.keys()))

    def get_available_references(self, sample_id: Optional[str] = None) -> List[str]:
        """
        Get available reference IDs from coverage data.

        Args:
            sample_id: Optional sample ID to get references for specific sample

        Returns:
            Sorted list of reference identifiers
        """
        if sample_id is not None:
            if sample_id in self.data:
                return sorted(list(self.data[sample_id].keys()))
            else:
                logger.warning("Sample ID %s not found in coverage data", sample_id)
                return []

        # Get all references across all samples
        references = set()
        for sample_data in self.data.values():
            references.update(sample_data.keys())
        return sorted(list(references))

    def get_sample_data(self, sample_id: str, reference: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get coverage data for a specific sample.

        Args:
            sample_id: Sample identifier
            reference: Optional reference identifier to get specific reference data

        Returns:
            Dictionary of DataFrames for the sample (or single DataFrame if reference specified)
        """
        if sample_id not in self.data:
            logger.warning("Sample %s not found in coverage data", sample_id)
            return {}

        sample_data = self.data[sample_id]

        if reference is not None:
            if reference in sample_data:
                return {reference: sample_data[reference]}
            else:
                logger.warning("Reference %s not found for sample %s", reference, sample_id)
                return {}

        return sample_data

    def get_coverage_dataframe(self, sample_id: str, reference: str) -> pd.DataFrame:
        """
        Get coverage DataFrame for a specific sample and reference.

        Args:
            sample_id: Sample identifier
            reference: Reference identifier

        Returns:
            DataFrame with coverage data, empty if not found
        """
        sample_data = self.get_sample_data(sample_id, reference)
        if reference in sample_data:
            return sample_data[reference].copy()
        else:
            logger.warning("No coverage data found for sample %s, reference %s", sample_id, reference)
            return pd.DataFrame()

    def calculate_recovery(self, sample_id: str, reference: str, depth_threshold: int = 10) -> float:
        """
        Calculate genome recovery for a specific sample and reference.

        Args:
            sample_id: Sample identifier
            reference: Reference identifier
            depth_threshold: Minimum depth for considering a position recovered

        Returns:
            Recovery percentage (0.0 to 1.0)
        """
        df = self.get_coverage_dataframe(sample_id, reference)

        if df.empty:
            logger.warning("No data available for recovery calculation: %s, %s", sample_id, reference)
            return 0.0

        if 'depth' not in df.columns:
            logger.warning("Depth column not found in DataFrame for %s, %s", sample_id, reference)
            return 0.0

        total_positions = len(df)
        recovered_positions = (df['depth'] >= depth_threshold).sum()

        if total_positions == 0:
            return 0.0

        return recovered_positions / total_positions

    def get_recovery_data(self, sample_ids: Optional[List[str]] = None, depth_threshold: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Calculate recovery data for all samples and references.

        Args:
            depth_threshold: Minimum depth for recovery calculation

        Returns:
            Nested dictionary with recovery percentages:
            {sample_id: {reference: recovery_percentage}}
        """
        recovery_data = {}

        if sample_ids is None:
            sample_ids = self.get_available_samples()

        for sample_id in sample_ids:
            recovery_data[sample_id] = {}
            for reference in self.get_available_references(sample_id):
                recovery_percentage = self.calculate_recovery(sample_id, reference, depth_threshold)
                recovery_data[sample_id][reference] = recovery_percentage

        return recovery_data

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the loaded coverage data.

        Returns:
            Dictionary with summary statistics about the dataset
        """
        total_samples = len(self.data)
        total_references = len(self.get_available_references())
        total_datasets = sum(len(references) for references in self.data.values())

        # Calculate total positions across all datasets
        total_positions = 0
        for sample_data in self.data.values():
            for df in sample_data.values():
                total_positions += len(df)

        summary = {
            'total_samples': total_samples,
            'total_references': total_references,
            'total_sample_reference_combinations': total_datasets,
            'total_positions': total_positions,
            'samples': self.get_available_samples(),
            'references': self.get_available_references()
        }

        return summary