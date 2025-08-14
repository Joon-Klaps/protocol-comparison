"""
Coverage analysis module for viral genomics data.

This module handles coverage plot analysis, depth calculations,
and overlay comparisons between different samples.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from .base import DataManager, BaseAnalyzer

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


class CoverageAnalyzer(BaseAnalyzer):
    """
    Analyzer for coverage and depth analysis.

    Handles:
    - Coverage recovery comparison with overlay plots
    - Coverage plots for each segment (LASV and HAZV)
    - Depth distribution analysis
    - Genome recovery statistics based on depth thresholds
    """

    def __init__(self, data_path: Path):
        """
        Initialize coverage analyzer.

        Args:
            data_path: Path to data directory
        """
        data_manager = CoverageDataManager(data_path)
        super().__init__(data_manager)
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
            sample_ids = self.get_samples()

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

    def generate_summary_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate summary statistics for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary of summary statistics
        """
        coverage_stats = self.calculate_coverage_stats(sample_ids)

        if coverage_stats.empty:
            return {}

        stats = {
            'overall': {
                'sample_count': coverage_stats['sample_id'].nunique(),
                'segment_count': coverage_stats['contig'].nunique(),
                'mean_coverage_percentage': coverage_stats['coverage_percentage'].mean(),
                'median_coverage_percentage': coverage_stats['coverage_percentage'].median(),
                'min_coverage_percentage': coverage_stats['coverage_percentage'].min(),
                'max_coverage_percentage': coverage_stats['coverage_percentage'].max()
            }
        }

        # Statistics by segment
        segment_stats = {}
        for contig in coverage_stats['contig'].unique():
            contig_data = coverage_stats[coverage_stats['contig'] == contig]
            segment_stats[contig] = {
                'sample_count': len(contig_data),
                'mean_coverage': contig_data['coverage_percentage'].mean(),
                'mean_depth': contig_data['mean_depth'].mean(),
                'median_depth': contig_data['median_depth'].mean()
            }

        stats['by_segment'] = segment_stats

        return stats

    def create_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create visualizations for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        if not sample_ids:
            sample_ids = self.get_samples()

        # Coverage overlay plot
        fig_overlay = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Coverage Overlay Plot']
        )

        colors = px.colors.qualitative.Set1

        for i, sample_id in enumerate(sample_ids[:10]):  # Limit to 10 samples for readability
            coverage_df = self.get_sample_coverage(sample_id)

            if not coverage_df.empty:
                color = colors[i % len(colors)]

                for contig in coverage_df['contig'].unique():
                    contig_data = coverage_df[coverage_df['contig'] == contig]

                    fig_overlay.add_trace(
                        go.Scatter(
                            x=contig_data['position'],
                            y=contig_data['depth'],
                            mode='lines',
                            name=f'{sample_id}_{contig}',
                            line=dict(color=color),
                            opacity=0.7
                        )
                    )

        fig_overlay.update_layout(
            title='Coverage Overlay Plot',
            xaxis_title='Position',
            yaxis_title='Depth',
            height=600
        )

        figures['coverage_overlay'] = fig_overlay

        # Coverage statistics bar plot
        coverage_stats = self.calculate_coverage_stats(sample_ids)

        if not coverage_stats.empty:
            fig_stats = px.bar(
                coverage_stats,
                x='sample_id',
                y='coverage_percentage',
                color='contig',
                title='Coverage Percentage by Sample and Segment',
                labels={'coverage_percentage': 'Coverage (%)', 'sample_id': 'Sample ID'}
            )
            fig_stats.update_layout(
                xaxis_tickangle=-45,
                height=500
            )
            figures['coverage_stats'] = fig_stats

            # Depth distribution histogram
            fig_depth_dist = px.histogram(
                coverage_stats,
                x='mean_depth',
                nbins=30,
                title='Distribution of Mean Depth Across Samples',
                labels={'mean_depth': 'Mean Depth', 'count': 'Frequency'}
            )
            figures['depth_distribution'] = fig_depth_dist

        return figures

    def create_segment_specific_plots(self, segment: str, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create segment-specific coverage plots.

        Args:
            segment: Segment identifier (e.g., 'L', 'S')
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary of plotly figures for the segment
        """
        if not sample_ids:
            sample_ids = self.get_samples()

        figures = {}

        # Individual coverage plots for each sample
        fig_segment = make_subplots(
            rows=len(sample_ids), cols=1,
            subplot_titles=[f'{sample_id} - {segment} segment' for sample_id in sample_ids],
            shared_xaxes=True
        )

        for i, sample_id in enumerate(sample_ids):
            coverage_df = self.get_sample_coverage(sample_id, segment)

            if not coverage_df.empty:
                fig_segment.add_trace(
                    go.Scatter(
                        x=coverage_df['position'],
                        y=coverage_df['depth'],
                        mode='lines',
                        name=f'{sample_id}',
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )

        fig_segment.update_layout(
            title=f'Coverage Plots - {segment} Segment',
            height=200 * len(sample_ids)
        )

        figures[f'{segment}_segment_coverage'] = fig_segment

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
        stats_df.to_csv(output_path / 'coverage_summary_stats.csv')

        # Export detailed coverage statistics
        coverage_stats = self.calculate_coverage_stats(sample_ids)
        if not coverage_stats.empty:
            coverage_stats.to_csv(output_path / 'coverage_detailed_stats.csv', index=False)

        logger.info("Coverage analysis results exported to %s", output_path)
