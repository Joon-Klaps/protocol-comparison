"""
Read statistics analysis module for viral genomics data.

This module handles read mapping statistics, UMI analysis,
and contamination checks between different samples.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from .base import DataManager, BaseAnalyzer

logger = logging.getLogger(__name__)


class ReadStatsDataManager(DataManager):
    """Data manager specifically for read statistics analysis."""

    def __init__(self, data_path: Path):
        super().__init__(data_path)
        self.stats_dir = self.data_path / "read_stats"
        self.mapping_dir = self.data_path / "mapping"
        self.contamination_dir = self.data_path / "contamination"

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load read statistics related data files.

        Returns:
            Dictionary containing read statistics DataFrames
        """
        data = {}

        # Load mapping statistics
        mapping_stats = self.mapping_dir / "mapping_stats.tsv"
        if mapping_stats.exists():
            data['mapping_stats'] = pd.read_csv(mapping_stats, sep='\t')

        # Load UMI statistics
        umi_stats = self.stats_dir / "umi_stats.tsv"
        if umi_stats.exists():
            data['umi_stats'] = pd.read_csv(umi_stats, sep='\t')

        # Load contamination check results
        contamination_lasv = self.contamination_dir / "lasv_contamination.tsv"
        if contamination_lasv.exists():
            data['contamination_lasv'] = pd.read_csv(contamination_lasv, sep='\t')

        contamination_hazv = self.contamination_dir / "hazv_contamination.tsv"
        if contamination_hazv.exists():
            data['contamination_hazv'] = pd.read_csv(contamination_hazv, sep='\t')

        # Load read count summaries
        read_counts = self.stats_dir / "read_counts.tsv"
        if read_counts.exists():
            data['read_counts'] = pd.read_csv(read_counts, sep='\t')

        return data

    def get_available_samples(self) -> List[str]:
        """Get available sample IDs from read statistics data."""
        samples = set()

        for data_type, df in self.load_data().items():
            if 'sample_id' in df.columns:
                samples.update(df['sample_id'].unique())

        return sorted(list(samples))


class ReadStatsAnalyzer(BaseAnalyzer):
    """
    Analyzer for read mapping and contamination statistics.

    Handles:
    - Total number of reads recovered for target vs total reads (per segment)
    - UMI number of reads recovered for target vs total reads (per segment)
    - Contamination check (HAZV and LASV presence)
    - Read mapping efficiency analysis
    """

    def __init__(self, data_path: Path):
        """
        Initialize read statistics analyzer.

        Args:
            data_path: Path to data directory
        """
        data_manager = ReadStatsDataManager(data_path)
        super().__init__(data_manager)

    def calculate_mapping_efficiency(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate mapping efficiency statistics for samples.

        Args:
            sample_ids: Optional list of specific samples to analyze

        Returns:
            DataFrame with mapping efficiency statistics
        """
        if 'mapping_stats' not in self.data:
            logger.warning("No mapping statistics data available")
            return pd.DataFrame()

        mapping_df = self.data['mapping_stats'].copy()

        if sample_ids:
            mapping_df = mapping_df[mapping_df['sample_id'].isin(sample_ids)]

        # Calculate efficiency percentages
        mapping_df['mapping_efficiency'] = (
            mapping_df['mapped_reads'] / mapping_df['total_reads'] * 100
        )
        mapping_df['target_efficiency'] = (
            mapping_df['target_mapped_reads'] / mapping_df['total_reads'] * 100
        )

        return mapping_df

    def calculate_umi_statistics(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate UMI-based statistics for samples.

        Args:
            sample_ids: Optional list of specific samples to analyze

        Returns:
            DataFrame with UMI statistics
        """
        if 'umi_stats' not in self.data:
            logger.warning("No UMI statistics data available")
            return pd.DataFrame()

        umi_df = self.data['umi_stats'].copy()

        if sample_ids:
            umi_df = umi_df[umi_df['sample_id'].isin(sample_ids)]

        # Calculate UMI efficiency percentages
        umi_df['umi_efficiency'] = (
            umi_df['target_umis'] / umi_df['total_umis'] * 100
        )

        return umi_df

    def analyze_contamination(self, sample_ids: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
        """
        Analyze contamination levels for LASV and HAZV.

        Args:
            sample_ids: Optional list of specific samples to analyze

        Returns:
            Dictionary with contamination analysis results
        """
        contamination_results = {}

        # LASV contamination
        if 'contamination_lasv' in self.data:
            lasv_df = self.data['contamination_lasv'].copy()
            if sample_ids:
                lasv_df = lasv_df[lasv_df['sample_id'].isin(sample_ids)]

            # Calculate relative contamination
            lasv_df['contamination_percentage'] = (
                lasv_df['contaminant_reads'] / lasv_df['total_reads'] * 100
            )
            contamination_results['lasv'] = lasv_df

        # HAZV contamination
        if 'contamination_hazv' in self.data:
            hazv_df = self.data['contamination_hazv'].copy()
            if sample_ids:
                hazv_df = hazv_df[hazv_df['sample_id'].isin(sample_ids)]

            # Calculate relative contamination
            hazv_df['contamination_percentage'] = (
                hazv_df['contaminant_reads'] / hazv_df['total_reads'] * 100
            )
            contamination_results['hazv'] = hazv_df

        return contamination_results

    def calculate_segment_statistics(self, sample_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate per-segment read statistics.

        Args:
            sample_ids: Optional list of specific samples to analyze

        Returns:
            DataFrame with per-segment statistics
        """
        if 'read_counts' not in self.data:
            logger.warning("No read count data available")
            return pd.DataFrame()

        read_counts_df = self.data['read_counts'].copy()

        if sample_ids:
            read_counts_df = read_counts_df[read_counts_df['sample_id'].isin(sample_ids)]

        # Calculate percentage of reads per segment
        read_counts_df['segment_percentage'] = (
            read_counts_df['segment_reads'] / read_counts_df['total_reads'] * 100
        )

        return read_counts_df

    def generate_summary_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate summary statistics for read statistics analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary of summary statistics
        """
        stats = {}

        # Mapping efficiency statistics
        mapping_df = self.calculate_mapping_efficiency(sample_ids)
        if not mapping_df.empty:
            stats['mapping_efficiency'] = {
                'mean_mapping_efficiency': mapping_df['mapping_efficiency'].mean(),
                'median_mapping_efficiency': mapping_df['mapping_efficiency'].median(),
                'min_mapping_efficiency': mapping_df['mapping_efficiency'].min(),
                'max_mapping_efficiency': mapping_df['mapping_efficiency'].max(),
                'mean_target_efficiency': mapping_df['target_efficiency'].mean(),
                'sample_count': len(mapping_df)
            }

        # UMI statistics
        umi_df = self.calculate_umi_statistics(sample_ids)
        if not umi_df.empty:
            stats['umi_efficiency'] = {
                'mean_umi_efficiency': umi_df['umi_efficiency'].mean(),
                'median_umi_efficiency': umi_df['umi_efficiency'].median(),
                'min_umi_efficiency': umi_df['umi_efficiency'].min(),
                'max_umi_efficiency': umi_df['umi_efficiency'].max(),
                'sample_count': len(umi_df)
            }

        # Contamination statistics
        contamination_data = self.analyze_contamination(sample_ids)
        contamination_stats = {}

        for virus_type, df in contamination_data.items():
            if not df.empty:
                contamination_stats[virus_type] = {
                    'mean_contamination': df['contamination_percentage'].mean(),
                    'max_contamination': df['contamination_percentage'].max(),
                    'samples_with_contamination': (df['contamination_percentage'] > 1.0).sum(),
                    'sample_count': len(df)
                }

        if contamination_stats:
            stats['contamination'] = contamination_stats

        return stats

    def create_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create visualizations for read statistics analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        # Mapping efficiency plot
        mapping_df = self.calculate_mapping_efficiency(sample_ids)
        if not mapping_df.empty:
            fig_mapping = px.bar(
                mapping_df,
                x='sample_id',
                y=['mapping_efficiency', 'target_efficiency'],
                title='Mapping Efficiency by Sample',
                labels={'value': 'Efficiency (%)', 'sample_id': 'Sample ID'},
                barmode='group'
            )
            fig_mapping.update_layout(
                xaxis_tickangle=-45,
                height=500
            )
            figures['mapping_efficiency'] = fig_mapping

        # UMI efficiency plot
        umi_df = self.calculate_umi_statistics(sample_ids)
        if not umi_df.empty:
            fig_umi = px.scatter(
                umi_df,
                x='total_umis',
                y='umi_efficiency',
                color='sample_id',
                title='UMI Efficiency vs Total UMIs',
                labels={'total_umis': 'Total UMIs', 'umi_efficiency': 'UMI Efficiency (%)'}
            )
            figures['umi_efficiency'] = fig_umi

        # Contamination analysis plots
        contamination_data = self.analyze_contamination(sample_ids)

        if contamination_data:
            # Create subplot for contamination comparison
            fig_contamination = make_subplots(
                rows=1, cols=len(contamination_data),
                subplot_titles=list(contamination_data.keys())
            )

            for i, (virus_type, df) in enumerate(contamination_data.items(), 1):
                fig_contamination.add_trace(
                    go.Bar(
                        x=df['sample_id'],
                        y=df['contamination_percentage'],
                        name=f'{virus_type.upper()} Contamination',
                        showlegend=(i == 1)
                    ),
                    row=1, col=i
                )

            fig_contamination.update_layout(
                title='Contamination Analysis by Virus Type',
                height=500
            )
            figures['contamination'] = fig_contamination

        # Segment statistics
        segment_df = self.calculate_segment_statistics(sample_ids)
        if not segment_df.empty:
            fig_segments = px.bar(
                segment_df,
                x='sample_id',
                y='segment_percentage',
                color='segment',
                title='Read Distribution by Segment',
                labels={'segment_percentage': 'Reads (%)', 'sample_id': 'Sample ID'}
            )
            fig_segments.update_layout(
                xaxis_tickangle=-45,
                height=500
            )
            figures['segment_distribution'] = fig_segments

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
        stats_df.to_csv(output_path / 'read_stats_summary.csv')

        # Export detailed data
        mapping_df = self.calculate_mapping_efficiency(sample_ids)
        if not mapping_df.empty:
            mapping_df.to_csv(output_path / 'mapping_efficiency_detailed.csv', index=False)

        umi_df = self.calculate_umi_statistics(sample_ids)
        if not umi_df.empty:
            umi_df.to_csv(output_path / 'umi_statistics_detailed.csv', index=False)

        contamination_data = self.analyze_contamination(sample_ids)
        for virus_type, df in contamination_data.items():
            df.to_csv(output_path / f'contamination_{virus_type}_detailed.csv', index=False)

        logger.info("Read statistics analysis results exported to %s", output_path)
