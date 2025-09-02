"""
Coverage data management module.

This module provides data management functionality specifically for coverage analysis,
including loading coverage depth data and managing coverage-related datasets.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
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
        self.annotation = self._load_annotation_data()

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

    def _load_annotation_data(self) -> pd.DataFrame:
        """
        Load mapping annotation data from mapping.parquet.

        Returns:
            DataFrame with deduplicated mapping annotations containing species, segment, reference
        """
        mapping_file = self.data_path / "mapping.parquet"

        if not mapping_file.exists():
            logger.warning("Mapping annotation file not found: %s", mapping_file)
            return pd.DataFrame()

        try:
            mapping_df = pd.read_parquet(mapping_file, engine='pyarrow')

            # Select only the relevant columns and deduplicate
            annotation_cols = ['species', 'segment', 'cluster']

            # Check if all required columns exist
            missing_cols = [col for col in annotation_cols if col not in mapping_df.columns]
            if missing_cols:
                logger.warning("Missing required columns in mapping.parquet: %s", missing_cols)
                return pd.DataFrame()

            # Select relevant columns and deduplicate
            annotation_df = mapping_df[annotation_cols].drop_duplicates()

            annotation_df.rename(columns={'cluster': 'reference'}, inplace=True)

            logger.info("Loaded %d unique reference annotations from mapping data", len(annotation_df))
            return annotation_df

        except Exception as e:
            logger.warning("Could not load mapping annotation data: %s", e)
            return pd.DataFrame()

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

    def get_available_references(self, sample_id: Optional[Union[str, List[str]]] = None) -> List[str]:
        """
        Get available reference IDs from coverage data.

        Args:
            sample_id: Optional sample ID to get references for specific sample
            sample_ids: Optional list of sample IDs to get references for specific samples

        Returns:
            Sorted list of reference identifiers
        """
        if sample_id is not None and isinstance(sample_id, str):
            references = set()
            if sample_id in self.data:
                references.update(self.data[sample_id].keys())
            else:
                logger.warning("Sample ID %s not found in coverage data", sample_id)
            return sorted(list(references))

        elif sample_id is not None and isinstance(sample_id, list):
            references = set()
            for sid in sample_id:
                if sid in self.data:
                    references.update(self.data[sid].keys())
                else:
                    logger.warning("Sample ID %s not found in coverage data", sid)
            return sorted(list(references))

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
            logger.debug("Sample %s not found in coverage data", sample_id)
            return {}

        sample_data = self.data[sample_id]

        if reference is not None:
            if reference in sample_data:
                return {reference: sample_data[reference]}
            else:
                logger.debug("Reference %s not found for sample %s", reference, sample_id)
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

        # Get final POS value
        total_positions = df['POS'].iloc[-1]
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

    def get_references_for_segment(self, segment: str) -> List[str]:
        """
        Get all references that belong to a specific segment.

        Args:
            segment: Segment identifier (e.g., 'L', 'S')

        Returns:
            List of reference identifiers for the segment
        """
        if self.annotation.empty:
            logger.warning("No annotation data available")
            return []

        # Filter annotation data for the specific segment
        segment_refs = self.annotation[
            self.annotation['segment'].str.upper() == segment.upper()
        ]['reference'].unique().tolist()

        return segment_refs

    def get_references_for_species_segment(self, species: str, segment: str) -> List[str]:
        """
        Get all references that belong to a specific species and segment.

        Args:
            species: Species identifier
            segment: Segment identifier

        Returns:
            List of reference identifiers for the species-segment combination
        """
        if self.annotation.empty:
            logger.warning("No annotation data available")
            return []

        # Filter annotation data for the specific species and segment
        refs = self.annotation[
            (self.annotation['species'].str.upper() == species.upper()) &
            (self.annotation['segment'].str.upper() == segment.upper())
        ]['reference'].unique().tolist()

        return refs


    def get_references_for_species(self, species: str) -> List[str]:
        """
        Get all references that belong to a specific species.

        Args:
            species: Species identifier

        Returns:
            List of reference identifiers for the species
        """
        if self.annotation.empty:
            logger.warning("No annotation data available")
            return []

        # Filter annotation data for the specific species
        species_refs = self.annotation[
            self.annotation['species'].str.upper() == species.upper()
        ]['reference'].unique().tolist()

        return species_refs

    def get_species_segment_for_reference(self, reference: str) -> Optional[Dict[str, str]]:
        """
        Get species and segment information for a specific reference.

        Args:
            reference: Reference identifier

        Returns:
            Dictionary with 'species' and 'segment' keys, or None if not found
        """
        if self.annotation.empty:
            logger.warning("No annotation data available")
            return None

        # Find the annotation row for this reference
        ref_annotation = self.annotation[
            self.annotation['reference'] == reference
        ]

        if ref_annotation.empty:
            return None

        # Return the first match (should be unique after deduplication)
        row = ref_annotation.iloc[0]
        return {
            'species': row['species'],
            'segment': row['segment']
        }
    def get_species_for_reference(self, reference: str) -> Optional[str]:
        """
        Get species information for a specific reference.

        Args:
            reference: Reference identifier

        Returns:
            Species identifier for the reference, or None if not found
        """
        species_segment = self.get_species_segment_for_reference(reference)
        return species_segment['species'] if species_segment else None

    def get_segment_for_reference(self, reference: str) -> Optional[str]:
        """
        Get segment information for a specific reference.

        Args:
            reference: Reference identifier

        Returns:
            Segment identifier for the reference, or None if not found
        """
        species_segment = self.get_species_segment_for_reference(reference)
        return species_segment['segment'] if species_segment else None

    def get_samples_for_reference(self, reference: str) -> List[str]:
        """
        Get all samples that have coverage data for a specific reference.

        Args:
            reference: Reference identifier

        Returns:
            List of sample identifiers that have data for this reference
        """
        samples = []
        for sample_id, sample_data in self.data.items():
            if reference in sample_data:
                samples.append(sample_id)

        return samples

    def get_frequency_sd_data(self, sample_ids: List[str], depth_threshold: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Calculate frequency standard deviation data for specified samples.

        This function computes the standard deviation of nucleotide frequencies across samples
        for each genomic position, properly aligned by position coordinates.

        Args:
            sample_ids: List of sample identifiers to get frequency shift data for
            depth_threshold: Minimum depth threshold for including positions

        Returns:
            Dictionary mapping reference -> DataFrame with columns:
              - POS, sdA, sdC, sdG, sdT, sum
              - counts_A, counts_C, counts_G, counts_T (dicts: {sample_id: absolute count})
              - hover_text_A, hover_text_C, hover_text_G, hover_text_T (preformatted strings for hover)

        Raises:
            ValueError: If required columns are missing from the data
        """
        frequency_sd_data = {}
        references = self.get_available_references(sample_ids)

        for ref in references:
            try:
                # Collect per-sample frames for this reference
                dfs_to_concat: List[pd.DataFrame] = []
                valid_sample_ids: List[str] = []

                for sample_id in sample_ids:
                    sample_data = self.get_sample_data(sample_id, ref)
                    if ref in sample_data and not sample_data[ref].empty:
                        df = sample_data[ref].copy()

                        # Ensure required columns exist
                        required_cols = ['POS', 'depth', 'A', 'C', 'T', 'G', 'freqA', 'freqC', 'freqG', 'freqT']
                        if not all(col in df.columns for col in required_cols):
                            raise ValueError(f"Missing required columns in data for sample {sample_id} and reference {ref}")

                        df = df.set_index('POS')[['depth', 'A', 'C', 'T', 'G', 'freqA', 'freqC', 'freqG', 'freqT']]
                        # Apply depth filter first
                        df = df[df['depth'] >= depth_threshold]
                        # Suffix columns with sample id for outer join later
                        df.columns = [f'{col}_{sample_id}' for col in df.columns]
                        dfs_to_concat.append(df)
                        valid_sample_ids.append(sample_id)

                if not dfs_to_concat:
                    logging.warning("No valid data found for reference %s", ref)
                    frequency_sd_data[ref] = pd.DataFrame()
                    continue

                # Combine all samples (outer join to include all positions present in any sample)
                combined_df = pd.concat(dfs_to_concat, axis=1, join='outer')

                # Helper formatters
                def _fmt_num(x: float) -> str:
                    if pd.isna(x):
                        return '-'
                    xf = float(x)
                    return str(int(xf)) if xf.is_integer() else f"{xf:.2f}"

                def _counts_for(base: str) -> pd.Series:
                    # Build {sample: count} preserving sample order for each POS
                    cols = [f'{base}_{sid}' for sid in valid_sample_ids if f'{base}_{sid}' in combined_df.columns]
                    if not cols:
                        return pd.Series([{}] * len(combined_df), index=combined_df.index)
                    def _row_to_dict(row: pd.Series) -> Dict[str, float]:
                        return {sid: float(row[f'{base}_{sid}'])
                                for sid in valid_sample_ids
                                if f'{base}_{sid}' in row.index and pd.notna(row[f'{base}_{sid}'])}
                    return combined_df[cols].apply(_row_to_dict, axis=1)

                def _counts_to_hover(d: Dict[str, float]) -> str:
                    if not d:
                        return ''
                    parts = [f"{sid}: {_fmt_num(d[sid])}" for sid in valid_sample_ids if sid in d]
                    return '{ ' + ', '.join(parts) + ' }'

                # Build per-nucleotide counts dicts and hover strings (cleanly)
                counts_A = _counts_for('A')
                counts_C = _counts_for('C')
                counts_G = _counts_for('G')
                counts_T = _counts_for('T')
                hover_A = counts_A.apply(_counts_to_hover)
                hover_C = counts_C.apply(_counts_to_hover)
                hover_G = counts_G.apply(_counts_to_hover)
                hover_T = counts_T.apply(_counts_to_hover)

                # Compute SDs across samples per nucleotide using column filters
                def _std_for(prefix: str) -> pd.Series:
                    cols = [c for c in combined_df.columns if c.startswith(prefix + '_')]
                    if not cols:
                        return pd.Series([0.0] * len(combined_df), index=combined_df.index)
                    return combined_df[cols].std(axis=1, ddof=1)

                sd_df = pd.DataFrame(index=combined_df.index)
                sd_df['sdA'] = _std_for('freqA')
                sd_df['sdC'] = _std_for('freqC')
                sd_df['sdG'] = _std_for('freqG')
                sd_df['sdT'] = _std_for('freqT')
                sd_df = sd_df.fillna(0.0)
                sd_df['sum'] = sd_df[['sdA', 'sdC', 'sdG', 'sdT']].sum(axis=1)
                sd_df['POS'] = sd_df.index

                # Attach per-nucleotide dicts and hover strings
                sd_df['counts_A'] = counts_A.values
                sd_df['counts_C'] = counts_C.values
                sd_df['counts_G'] = counts_G.values
                sd_df['counts_T'] = counts_T.values
                sd_df['hover_text_A'] = hover_A.values
                sd_df['hover_text_C'] = hover_C.values
                sd_df['hover_text_G'] = hover_G.values
                sd_df['hover_text_T'] = hover_T.values
                sd_df.reset_index(drop=True, inplace=True)

                frequency_sd_data[ref] = sd_df
                logging.debug("Calculated frequency SD for reference %s with %d samples and %d positions",
                            ref, len(valid_sample_ids), len(sd_df))

            except (ValueError, KeyError, pd.errors.ParserError) as e:
                logging.error("Error calculating frequency SD for reference %s: %s", ref, e)
                frequency_sd_data[ref] = pd.DataFrame()

        return frequency_sd_data