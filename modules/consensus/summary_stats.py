"""
Consensus summary statistics.

This module calculates statistics for consensus sequence analysis, genome recovery,
and average nucleotide identity (ANI) comparisons.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import logging

from ..base import DataManager

logger = logging.getLogger(__name__)


class ConsensusDataManager(DataManager):
    """Data manager specifically for consensus sequence analysis."""

    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.consensus_dir = self.data_path / "consensus"
        self.reference_dir = self.data_path / "references"

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load consensus-related data files.

        Returns:
            Dictionary containing consensus data DataFrames
        """
        data = {}

        # Load ANI comparison data if available
        ani_file = self.consensus_dir / "ani_comparison.tsv"
        if ani_file.exists():
            data['ani'] = pd.read_csv(ani_file, sep='\t')

        # Load genome recovery data
        recovery_file = self.consensus_dir / "genome_recovery.tsv"
        if recovery_file.exists():
            data['recovery'] = pd.read_csv(recovery_file, sep='\t')

        # Load reference mapping data
        mapping_file = self.reference_dir / "reference_mapping.tsv"
        if mapping_file.exists():
            data['mapping'] = pd.read_csv(mapping_file, sep='\t')

        return data

    def get_available_samples(self) -> List[str]:
        """Get available sample IDs from consensus data."""
        samples = set()

        for data_type, df in self.load_data().items():
            if 'sample_id' in df.columns:
                samples.update(df['sample_id'].unique())

        return sorted(list(samples))


class ConsensusSummaryStats:
    """
    Summary statistics calculator for consensus sequence analysis.

    Handles genome recovery statistics, ANI calculations,
    and consensus sequence quality metrics.
    """

    def __init__(self, data_path: Path):
        """
        Initialize consensus summary statistics.

        Args:
            data_path: Path to data directory
        """
        self.data_manager = ConsensusDataManager(data_path)
        self.data = self.data_manager.load_data()

    def calculate_genome_recovery_stats(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate genome recovery statistics for samples.

        Args:
            sample_ids: Optional list of specific samples to analyze

        Returns:
            DataFrame with genome recovery statistics
        """
        if 'recovery' not in self.data:
            logger.warning("No genome recovery data available")
            return pd.DataFrame()

        recovery_df = self.data['recovery'].copy()

        if sample_ids:
            recovery_df = recovery_df[recovery_df['sample_id'].isin(sample_ids)]

        # Calculate recovery percentages if not already present
        if 'recovery_percentage' not in recovery_df.columns:
            if 'covered_bases' in recovery_df.columns and 'total_bases' in recovery_df.columns:
                recovery_df['recovery_percentage'] = (
                    recovery_df['covered_bases'] / recovery_df['total_bases'] * 100
                )

        return recovery_df

    def calculate_ani_matrix(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate Average Nucleotide Identity (ANI) matrix between samples.

        Args:
            sample_ids: Optional list of specific samples to analyze

        Returns:
            DataFrame with ANI comparison matrix
        """
        if 'ani' not in self.data:
            logger.warning("No ANI data available")
            return pd.DataFrame()

        ani_df = self.data['ani'].copy()

        if sample_ids:
            ani_df = ani_df[
                (ani_df['sample1'].isin(sample_ids)) &
                (ani_df['sample2'].isin(sample_ids))
            ]

        # Create matrix format
        if not ani_df.empty:
            matrix = ani_df.pivot(index='sample1', columns='sample2', values='ani_value')
            return matrix.fillna(100.0)  # Self-comparison is 100%

        return pd.DataFrame()

    def calculate_recovery_summary(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate summary statistics for genome recovery.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with recovery summary statistics
        """
        recovery_df = self.calculate_genome_recovery_stats(sample_ids)

        if recovery_df.empty:
            return {}

        stats = {
            'sample_count': len(recovery_df),
            'mean_recovery_pct': recovery_df['recovery_percentage'].mean(),
            'median_recovery_pct': recovery_df['recovery_percentage'].median(),
            'min_recovery_pct': recovery_df['recovery_percentage'].min(),
            'max_recovery_pct': recovery_df['recovery_percentage'].max(),
            'std_recovery_pct': recovery_df['recovery_percentage'].std()
        }

        # Add percentage of samples with good recovery (>= 80%)
        good_recovery = (recovery_df['recovery_percentage'] >= 80).sum()
        stats['samples_with_good_recovery_pct'] = (good_recovery / len(recovery_df)) * 100

        # Add percentage of samples with excellent recovery (>= 95%)
        excellent_recovery = (recovery_df['recovery_percentage'] >= 95).sum()
        stats['samples_with_excellent_recovery_pct'] = (excellent_recovery / len(recovery_df)) * 100

        return stats

    def calculate_ani_summary(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate summary statistics for ANI comparisons.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with ANI summary statistics
        """
        ani_matrix = self.calculate_ani_matrix(sample_ids)

        if ani_matrix.empty:
            return {}

        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu(ani_matrix.values, k=1)
        ani_values = upper_triangle[upper_triangle > 0]

        if len(ani_values) == 0:
            return {}

        stats = {
            'comparison_count': len(ani_values),
            'mean_ani': float(np.mean(ani_values)),
            'median_ani': float(np.median(ani_values)),
            'min_ani': float(np.min(ani_values)),
            'max_ani': float(np.max(ani_values)),
            'std_ani': float(np.std(ani_values))
        }

        # Add percentage of comparisons with high similarity (>= 95%)
        high_similarity = (ani_values >= 95).sum()
        stats['high_similarity_comparisons_pct'] = (high_similarity / len(ani_values)) * 100

        # Add percentage of comparisons with very high similarity (>= 99%)
        very_high_similarity = (ani_values >= 99).sum()
        stats['very_high_similarity_comparisons_pct'] = (very_high_similarity / len(ani_values)) * 100

        return stats

    def calculate_segment_specific_recovery(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """
        Calculate segment-specific recovery statistics.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with segment-specific recovery statistics
        """
        recovery_df = self.calculate_genome_recovery_stats(sample_ids)

        if recovery_df.empty or 'segment' not in recovery_df.columns:
            return {}

        segment_stats = {}
        for segment in recovery_df['segment'].unique():
            segment_data = recovery_df[recovery_df['segment'] == segment]
            segment_stats[segment] = {
                'sample_count': len(segment_data),
                'mean_recovery_pct': segment_data['recovery_percentage'].mean(),
                'median_recovery_pct': segment_data['recovery_percentage'].median(),
                'min_recovery_pct': segment_data['recovery_percentage'].min(),
                'max_recovery_pct': segment_data['recovery_percentage'].max(),
                'std_recovery_pct': segment_data['recovery_percentage'].std()
            }

        return segment_stats

    def calculate_overall_summary(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate overall summary statistics for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary with overall summary statistics
        """
        summary = {}

        # Genome recovery statistics
        recovery_stats = self.calculate_recovery_summary(sample_ids)
        if recovery_stats:
            summary['genome_recovery'] = recovery_stats

        # ANI statistics
        ani_stats = self.calculate_ani_summary(sample_ids)
        if ani_stats:
            summary['ani'] = ani_stats

        # Segment-specific recovery
        segment_stats = self.calculate_segment_specific_recovery(sample_ids)
        if segment_stats:
            summary['by_segment'] = segment_stats

        return summary

    def export_results(self, output_path: Path, sample_ids: Optional[List[str]] = None) -> None:
        """
        Export consensus statistics to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Export detailed genome recovery data
        recovery_df = self.calculate_genome_recovery_stats(sample_ids)
        if not recovery_df.empty:
            recovery_df.to_csv(output_path / 'genome_recovery_detailed.csv', index=False)

        # Export ANI matrix
        ani_matrix = self.calculate_ani_matrix(sample_ids)
        if not ani_matrix.empty:
            ani_matrix.to_csv(output_path / 'ani_matrix.csv')

        # Export overall summary statistics
        overall_stats = self.calculate_overall_summary(sample_ids)
        if overall_stats:
            overall_df = pd.DataFrame.from_dict(overall_stats, orient='index')
            overall_df.to_csv(output_path / 'consensus_overall_stats.csv')

            # Also save as JSON for easier reading
            import json
            with open(output_path / 'consensus_overall_stats.json', 'w', encoding='utf-8') as f:
                json.dump(overall_stats, f, indent=2, default=str)

        # Export segment-specific statistics if available
        segment_stats = self.calculate_segment_specific_recovery(sample_ids)
        if segment_stats:
            segment_df = pd.DataFrame.from_dict(segment_stats, orient='index')
            segment_df.to_csv(output_path / 'consensus_segment_stats.csv')

        logger.info("Consensus summary statistics exported to %s", output_path)
