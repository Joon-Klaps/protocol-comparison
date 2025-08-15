"""
Coverage summary statistics.

This module calculates statistics for coverage depth and genome recovery analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import logging

from ..base import DataManager

logger = logging.getLogger(__name__)


class CoverageDataManager(DataManager):
    """Data manager specifically for coverage analysis."""

    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.coverage_dir = self.data_path / "coverage"
        self.depth_dir = self.data_path / "depth"

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load coverage-related data files.

        Returns:
            Dictionary containing coverage data DataFrames
        """
        data = {}

        # Load coverage depth data
        depth_files = list(self.depth_dir.glob("*.depth"))
        for depth_file in depth_files:
            sample_id = depth_file.stem
            try:
                df = pd.read_csv(depth_file, sep='\t', names=['contig', 'position', 'depth'])
                df['sample_id'] = sample_id
                data[f'depth_{sample_id}'] = df
            except Exception as e:
                logger.warning(f"Could not load depth file {depth_file}: {e}")

        # Load summary coverage statistics
        coverage_summary = self.coverage_dir / "coverage_summary.tsv"
        if coverage_summary.exists():
            data['coverage_summary'] = pd.read_csv(coverage_summary, sep='\t')

        return data

    def get_available_samples(self) -> List[str]:
        """Get available sample IDs from coverage data."""
        samples = set()

        for key in self.load_data().keys():
            if key.startswith('depth_'):
                samples.add(key.replace('depth_', ''))

        return sorted(list(samples))


class CoverageSummaryStats:
    """
    Summary statistics calculator for coverage analysis.

    Handles coverage depth statistics, genome recovery calculations,
    and segment-specific coverage metrics.
    """

    def __init__(self, data_path: Path):
        """
        Initialize coverage summary statistics.

        Args:
            data_path: Path to data directory
        """
        self.data_manager = CoverageDataManager(data_path)
        self.data = self.data_manager.load_data()
        self.depth_threshold = 10  # Default minimum depth for recovery

    def set_depth_threshold(self, threshold: int) -> None:
        """Set the minimum depth threshold for genome recovery calculations."""
        self.depth_threshold = threshold

    def get_sample_coverage(self, sample_id: str, segment: Optional[str] = None) -> pd.DataFrame:
        """
        Get coverage data for a specific sample.

        Args:
            sample_id: Sample identifier
            segment: Optional segment filter (e.g., 'L', 'S')

        Returns:
            DataFrame with coverage data
        """
        key = f'depth_{sample_id}'
        if key not in self.data:
            logger.warning(f"No coverage data found for sample {sample_id}")
            return pd.DataFrame()

        df = self.data[key].copy()

        if segment:
            df = df[df['contig'].str.contains(segment, case=False)]

        return df

    def calculate_coverage_stats(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate coverage statistics for samples.

        Args:
            sample_ids: Optional list of specific samples to analyze

        Returns:
            DataFrame with coverage statistics
        """
        if not sample_ids:
            sample_ids = self.data_manager.get_available_samples()

        stats_list = []

        for sample_id in sample_ids:
            coverage_df = self.get_sample_coverage(sample_id)

            if coverage_df.empty:
                continue

            # Group by contig (segment)
            for contig in coverage_df['contig'].unique():
                contig_data = coverage_df[coverage_df['contig'] == contig]

                stats = {
                    'sample_id': sample_id,
                    'contig': contig,
                    'total_positions': len(contig_data),
                    'mean_depth': contig_data['depth'].mean(),
                    'median_depth': contig_data['depth'].median(),
                    'max_depth': contig_data['depth'].max(),
                    'min_depth': contig_data['depth'].min(),
                    'positions_above_threshold': (contig_data['depth'] >= self.depth_threshold).sum(),
                    'coverage_percentage': (contig_data['depth'] >= self.depth_threshold).mean() * 100,
                    'zero_coverage_positions': (contig_data['depth'] == 0).sum()
                }

                stats_list.append(stats)

        return pd.DataFrame(stats_list)

    def calculate_segment_specific_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate segment-specific coverage statistics.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with segment-specific statistics
        """
        coverage_stats = self.calculate_coverage_stats(sample_ids)

        if coverage_stats.empty:
            return {}

        segment_stats = {}
        for contig in coverage_stats['contig'].unique():
            contig_data = coverage_stats[coverage_stats['contig'] == contig]
            segment_stats[contig] = {
                'sample_count': len(contig_data),
                'mean_coverage_percentage': contig_data['coverage_percentage'].mean(),
                'median_coverage_percentage': contig_data['coverage_percentage'].median(),
                'min_coverage_percentage': contig_data['coverage_percentage'].min(),
                'max_coverage_percentage': contig_data['coverage_percentage'].max(),
                'std_coverage_percentage': contig_data['coverage_percentage'].std(),
                'mean_depth': contig_data['mean_depth'].mean(),
                'median_depth': contig_data['median_depth'].mean(),
                'max_depth': contig_data['max_depth'].max(),
                'min_depth': contig_data['min_depth'].min()
            }

        return segment_stats

    def calculate_overall_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate overall coverage statistics across all samples and segments.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with overall statistics
        """
        coverage_stats = self.calculate_coverage_stats(sample_ids)

        if coverage_stats.empty:
            return {}

        stats = {
            'sample_count': coverage_stats['sample_id'].nunique(),
            'segment_count': coverage_stats['contig'].nunique(),
            'total_comparisons': len(coverage_stats),
            'mean_coverage_percentage': coverage_stats['coverage_percentage'].mean(),
            'median_coverage_percentage': coverage_stats['coverage_percentage'].median(),
            'min_coverage_percentage': coverage_stats['coverage_percentage'].min(),
            'max_coverage_percentage': coverage_stats['coverage_percentage'].max(),
            'std_coverage_percentage': coverage_stats['coverage_percentage'].std(),
            'mean_depth': coverage_stats['mean_depth'].mean(),
            'median_depth': coverage_stats['median_depth'].mean(),
            'depth_threshold': self.depth_threshold
        }

        # Calculate percentage of samples with good coverage (>= 80%)
        good_coverage = (coverage_stats['coverage_percentage'] >= 80).sum()
        stats['samples_with_good_coverage_pct'] = (good_coverage / len(coverage_stats)) * 100

        return stats

    def export_results(self, output_path: Path, sample_ids: Optional[List[str]] = None) -> None:
        """
        Export coverage statistics to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Export detailed coverage statistics
        coverage_stats = self.calculate_coverage_stats(sample_ids)
        if not coverage_stats.empty:
            coverage_stats.to_csv(output_path / 'coverage_detailed_stats.csv', index=False)

        # Export segment-specific statistics
        segment_stats = self.calculate_segment_specific_stats(sample_ids)
        if segment_stats:
            segment_df = pd.DataFrame.from_dict(segment_stats, orient='index')
            segment_df.to_csv(output_path / 'coverage_segment_stats.csv')

        # Export overall statistics
        overall_stats = self.calculate_overall_stats(sample_ids)
        if overall_stats:
            overall_df = pd.DataFrame([overall_stats])
            overall_df.to_csv(output_path / 'coverage_overall_stats.csv', index=False)

            # Also save as JSON for easier reading
            import json
            with open(output_path / 'coverage_overall_stats.json', 'w', encoding='utf-8') as f:
                json.dump(overall_stats, f, indent=2, default=str)

        logger.info("Coverage summary statistics exported to %s", output_path)
