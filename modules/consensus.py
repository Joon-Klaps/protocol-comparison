"""
Consensus analysis module for viral genomics data.

This module handles consensus sequence comparison, ANI calculations,
and nucleotide-level comparisons between different samples.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from .base import DataManager, BaseAnalyzer

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


class ConsensusAnalyzer(BaseAnalyzer):
    """
    Analyzer for consensus sequence comparisons and genome recovery analysis.

    Handles:
    - Consensus comparison (how much recovered in each)
    - Nucleotide-level comparisons (ANI to other sequences)
    - Genome recovery statistics
    """

    def __init__(self, data_path: Path):
        """
        Initialize consensus analyzer.

        Args:
            data_path: Path to data directory
        """
        data_manager = ConsensusDataManager(data_path)
        super().__init__(data_manager)

    def calculate_genome_recovery(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
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

        # Calculate recovery percentages
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
        matrix = ani_df.pivot(index='sample1', columns='sample2', values='ani_value')
        return matrix.fillna(100.0)  # Self-comparison is 100%

    def generate_summary_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate summary statistics for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary of summary statistics
        """
        stats = {}

        # Genome recovery statistics
        recovery_df = self.calculate_genome_recovery(sample_ids)
        if not recovery_df.empty:
            stats['genome_recovery'] = {
                'mean_recovery': recovery_df['recovery_percentage'].mean(),
                'median_recovery': recovery_df['recovery_percentage'].median(),
                'min_recovery': recovery_df['recovery_percentage'].min(),
                'max_recovery': recovery_df['recovery_percentage'].max(),
                'std_recovery': recovery_df['recovery_percentage'].std(),
                'sample_count': len(recovery_df)
            }

        # ANI statistics
        ani_matrix = self.calculate_ani_matrix(sample_ids)
        if not ani_matrix.empty:
            # Get upper triangle (excluding diagonal)
            upper_triangle = np.triu(ani_matrix.values, k=1)
            ani_values = upper_triangle[upper_triangle > 0]

            if len(ani_values) > 0:
                stats['ani'] = {
                    'mean_ani': float(np.mean(ani_values)),
                    'median_ani': float(np.median(ani_values)),
                    'min_ani': float(np.min(ani_values)),
                    'max_ani': float(np.max(ani_values)),
                    'std_ani': float(np.std(ani_values)),
                    'comparison_count': len(ani_values)
                }

        return stats

    def create_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create visualizations for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        # Genome recovery plot
        recovery_df = self.calculate_genome_recovery(sample_ids)
        if not recovery_df.empty:
            fig_recovery = px.bar(
                recovery_df,
                x='sample_id',
                y='recovery_percentage',
                title='Genome Recovery Percentage by Sample',
                labels={'recovery_percentage': 'Recovery (%)', 'sample_id': 'Sample ID'}
            )
            fig_recovery.update_layout(
                xaxis_tickangle=-45,
                height=500
            )
            figures['genome_recovery'] = fig_recovery

        # ANI heatmap
        ani_matrix = self.calculate_ani_matrix(sample_ids)
        if not ani_matrix.empty and ani_matrix.shape[0] > 1:
            fig_ani = px.imshow(
                ani_matrix,
                title='Average Nucleotide Identity (ANI) Matrix',
                labels=dict(x='Sample', y='Sample', color='ANI (%)'),
                color_continuous_scale='viridis'
            )
            fig_ani.update_layout(height=600)
            figures['ani_matrix'] = fig_ani

        # Recovery distribution
        if not recovery_df.empty:
            fig_dist = px.histogram(
                recovery_df,
                x='recovery_percentage',
                nbins=20,
                title='Distribution of Genome Recovery Percentages',
                labels={'recovery_percentage': 'Recovery (%)', 'count': 'Frequency'}
            )
            figures['recovery_distribution'] = fig_dist

        return figures

    def export_results(self, output_path: Path, sample_ids: Optional[List[str]] = None) -> None:
        """
        Export analysis results to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        output_path.mkdir(parents=True, exist_ok=True)

        # Export summary statistics
        stats = self.generate_summary_stats(sample_ids)
        stats_df = pd.DataFrame.from_dict(stats, orient='index')
        stats_df.to_csv(output_path / 'consensus_summary_stats.csv')

        # Export detailed data
        recovery_df = self.calculate_genome_recovery(sample_ids)
        if not recovery_df.empty:
            recovery_df.to_csv(output_path / 'genome_recovery_detailed.csv', index=False)

        ani_matrix = self.calculate_ani_matrix(sample_ids)
        if not ani_matrix.empty:
            ani_matrix.to_csv(output_path / 'ani_matrix.csv')

        logger.info(f"Consensus analysis results exported to {output_path}")
