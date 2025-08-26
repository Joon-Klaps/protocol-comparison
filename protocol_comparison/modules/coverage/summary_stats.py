"""
Coverage summary statistics.

This module calculates statistics for coverage depth and genome recovery analysis.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import pandas as pd
import logging
import json

if TYPE_CHECKING:
    from .data import CoverageDataManager
else:
    from .data import CoverageDataManager

logger = logging.getLogger(__name__)


class CoverageSummaryStats:
    """
    Calculator for coverage analysis statistics.

    Handles:
    - Genome recovery calculations per species and segment
    - Coverage depth statistics
    - Reference-specific coverage metrics
    """

    def __init__(self, data_path: Path, data_manager: Optional['CoverageDataManager'] = None):
        """
        Initialize coverage summary stats calculator.

        Args:
            data_path: Path to data directory
            data_manager: Optional shared data manager instance to avoid duplicate loading
        """
        if data_manager is not None:
            self.data_manager = data_manager
            # Don't reload data if it's already loaded in the shared data manager
            self.data = self.data_manager.load_data() if not hasattr(data_manager, '_flat_data') or data_manager._flat_data is None else data_manager._flat_data
        else:
            self.data_manager = CoverageDataManager(data_path)
            self.data = self.data_manager.load_data()

        self.depth_threshold = 10  # Default minimum depth for recovery

    def set_depth_threshold(self, threshold: int) -> None:
        """Set the minimum depth threshold for genome recovery calculations."""
        self.depth_threshold = threshold

    def calculate_recovery_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate recovery statistics grouped by species and segment.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with recovery statistics per species per segment
        """
        if not sample_ids:
            sample_ids = self.data_manager.get_available_samples()

        # Get recovery data using the data manager
        recovery_data = self.data_manager.get_recovery_data(sample_ids, self.depth_threshold)

        # Convert to DataFrame for easier analysis
        recovery_records = []
        for sample_id, ref_data in recovery_data.items():
            for reference, recovery_value in ref_data.items():
                # Parse reference to extract species and segment
                # Assuming reference format like "LASV_L" or "HAZV_S"
                ref_parts = reference.split('_')
                if len(ref_parts) >= 2:
                    species = ref_parts[0]
                    segment = ref_parts[1]
                else:
                    species = reference
                    segment = "unknown"

                recovery_records.append({
                    'sample_id': sample_id,
                    'reference': reference,
                    'species': species,
                    'segment': segment,
                    'recovery_value': recovery_value
                })

        if not recovery_records:
            return {}

        recovery_df = pd.DataFrame(recovery_records)

        # Group by species and segment
        grouped = recovery_df.groupby(['species', 'segment'])
        stats = {}

        for (species, segment), group in grouped:
            # Calculate statistics for this species-segment combination
            min_idx = group['recovery_value'].idxmin()
            max_idx = group['recovery_value'].idxmax()

            recovery_stats = {
                'mean_recovery_pct': f"{group['recovery_value'].mean() * 100:.1f}%",
                'min_recovery': f"{group['recovery_value'].min() * 100:.1f}%",
                'min_recovery_sample': group.loc[min_idx, 'sample_id'],
                'max_recovery': f"{group['recovery_value'].max() * 100:.1f}%",
                'max_recovery_sample': group.loc[max_idx, 'sample_id'],
                'sample_count': len(group),
                'depth_threshold': self.depth_threshold
            }

            # Initialize species in stats if not exists
            if species not in stats:
                stats[species] = {}

            # Add segment stats for this species
            stats[species][segment] = recovery_stats

        return stats

    def calculate_depth_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate depth statistics across samples and references.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with depth statistics
        """
        if not sample_ids:
            sample_ids = self.data_manager.get_available_samples()

        depth_records = []

        # Process all data to calculate depth statistics
        for sample_id in sample_ids:
            sample_data = self.data_manager.get_sample_data(sample_id)
            for reference, df in sample_data.items():
                if 'depth' in df.columns and not df.empty:
                    # Parse reference to extract species and segment
                    ref_parts = reference.split('_')
                    if len(ref_parts) >= 2:
                        species = ref_parts[0]
                        segment = ref_parts[1]
                    else:
                        species = reference
                        segment = "unknown"

                    depth_stats = {
                        'sample_id': sample_id,
                        'reference': reference,
                        'species': species,
                        'segment': segment,
                        'mean_depth': df['depth'].mean(),
                        'median_depth': df['depth'].median(),
                        'max_depth': df['depth'].max(),
                        'positions_covered': (df['depth'] > 0).sum(),
                        'total_positions': len(df),
                        'positions_at_threshold': (df['depth'] >= self.depth_threshold).sum()
                    }

                    depth_records.append(depth_stats)

        if not depth_records:
            return {}

        depth_df = pd.DataFrame(depth_records)

        # Group by species and segment
        grouped = depth_df.groupby(['species', 'segment'])
        stats = {}

        for (species, segment), group in grouped:
            depth_summary = {
                'mean_avg_depth': group['mean_depth'].mean(),
                'mean_median_depth': group['median_depth'].mean(),
                'total_samples': len(group),
                'avg_positions_covered': group['positions_covered'].mean(),
                'avg_total_positions': group['total_positions'].mean(),
                'avg_positions_at_threshold': group['positions_at_threshold'].mean(),
                'depth_threshold': self.depth_threshold
            }

            # Initialize species in stats if not exists
            if species not in stats:
                stats[species] = {}

            # Add segment stats for this species
            stats[species][segment] = depth_summary

        return stats

    def export_results(self, output_path: Path, sample_ids: Optional[List[str]] = None) -> None:
        """
        Export coverage statistics to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Export recovery statistics
        recovery_stats = self.calculate_recovery_stats(sample_ids)
        if recovery_stats:
            # Flatten nested dictionary for CSV export
            flat_recovery_stats = []
            for species, segments in recovery_stats.items():
                for segment, stats in segments.items():
                    row = {'species': species, 'segment': segment}
                    row.update(stats)
                    flat_recovery_stats.append(row)

            if flat_recovery_stats:
                recovery_df = pd.DataFrame(flat_recovery_stats)
                recovery_df.to_csv(output_path / 'coverage_recovery_summary.csv', index=False)

        # Export depth statistics
        depth_stats = self.calculate_depth_stats(sample_ids)
        if depth_stats:
            # Flatten nested dictionary for CSV export
            flat_depth_stats = []
            for species, segments in depth_stats.items():
                for segment, stats in segments.items():
                    row = {'species': species, 'segment': segment}
                    row.update(stats)
                    flat_depth_stats.append(row)

            if flat_depth_stats:
                depth_df = pd.DataFrame(flat_depth_stats)
                depth_df.to_csv(output_path / 'coverage_depth_summary.csv', index=False)

        # Also save as JSON for programmatic access
        with open(output_path / 'coverage_recovery_stats.json', 'w', encoding='utf-8') as f:
            json.dump(recovery_stats, f, indent=2, default=str)

        with open(output_path / 'coverage_depth_stats.json', 'w', encoding='utf-8') as f:
            json.dump(depth_stats, f, indent=2, default=str)

        logger.info("Coverage summary statistics exported to %s", output_path)
