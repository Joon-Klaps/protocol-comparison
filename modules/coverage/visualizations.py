"""
Coverage visualizations.

This module creates visualizations for coverage depth analysis and genome recovery.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from .summary_stats import CoverageSummaryStats

logger = logging.getLogger(__name__)


class CoverageVisualizations:
    """
    Visualization generator for coverage analysis.

    Creates various plots for coverage depth analysis:
    - Coverage overlay plots
    - Coverage statistics bar plots
    - Depth distribution plots
    - Segment-specific coverage plots
    """

    def __init__(self, data_path: Path):
        """
        Initialize coverage visualizations.

        Args:
            data_path: Path to data directory
        """
        self.stats = CoverageSummaryStats(data_path)

    def create_coverage_overlay_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create coverage overlay plot showing all samples on one plot.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with coverage overlay
        """
        if not sample_ids:
            sample_ids = self.stats.data_manager.get_available_samples()

        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Coverage Overlay Plot']
        )

        colors = px.colors.qualitative.Set1

        for i, sample_id in enumerate(sample_ids[:10]):  # Limit to 10 samples for readability
            coverage_df = self.stats.get_sample_coverage(sample_id)

            if not coverage_df.empty:
                color = colors[i % len(colors)]

                for contig in coverage_df['contig'].unique():
                    contig_data = coverage_df[coverage_df['contig'] == contig]

                    fig.add_trace(
                        go.Scatter(
                            x=contig_data['position'],
                            y=contig_data['depth'],
                            mode='lines',
                            name=f'{sample_id}_{contig}',
                            line=dict(color=color),
                            opacity=0.7
                        )
                    )

        fig.update_layout(
            title='Coverage Overlay Plot',
            xaxis_title='Position',
            yaxis_title='Depth',
            height=600,
            showlegend=True
        )

        return fig

    def create_coverage_stats_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create coverage statistics bar plot.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with coverage statistics
        """
        coverage_stats = self.stats.calculate_coverage_stats(sample_ids)

        if coverage_stats.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No coverage data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.bar(
            coverage_stats,
            x='sample_id',
            y='coverage_percentage',
            color='contig',
            title='Coverage Percentage by Sample and Segment',
            labels={'coverage_percentage': 'Coverage (%)', 'sample_id': 'Sample ID'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500
        )

        return fig

    def create_depth_distribution_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create depth distribution histogram.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with depth distribution
        """
        coverage_stats = self.stats.calculate_coverage_stats(sample_ids)

        if coverage_stats.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No coverage data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.histogram(
            coverage_stats,
            x='mean_depth',
            nbins=30,
            title='Distribution of Mean Depth Across Samples',
            labels={'mean_depth': 'Mean Depth', 'count': 'Frequency'}
        )

        return fig

    def create_segment_specific_plots(self, segment: str, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create segment-specific coverage plots.

        Args:
            segment: Segment identifier (e.g., 'L', 'S')
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Plotly figure for the segment
        """
        if not sample_ids:
            sample_ids = self.stats.data_manager.get_available_samples()

        # Filter samples that have data for this segment
        valid_samples = []
        for sample_id in sample_ids:
            coverage_df = self.stats.get_sample_coverage(sample_id, segment)
            if not coverage_df.empty:
                valid_samples.append(sample_id)

        if not valid_samples:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No coverage data available for segment {segment}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Create subplot for each sample
        fig = make_subplots(
            rows=len(valid_samples), cols=1,
            subplot_titles=[f'{sample_id} - {segment} segment' for sample_id in valid_samples],
            shared_xaxes=True
        )

        for i, sample_id in enumerate(valid_samples):
            coverage_df = self.stats.get_sample_coverage(sample_id, segment)

            fig.add_trace(
                go.Scatter(
                    x=coverage_df['position'],
                    y=coverage_df['depth'],
                    mode='lines',
                    name=f'{sample_id}',
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )

        fig.update_layout(
            title=f'Coverage Plots - {segment} Segment',
            height=200 * len(valid_samples),
            showlegend=True
        )

        return fig

    def create_coverage_heatmap(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create coverage percentage heatmap.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with coverage heatmap
        """
        coverage_stats = self.stats.calculate_coverage_stats(sample_ids)

        if coverage_stats.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No coverage data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Pivot to create matrix format
        heatmap_data = coverage_stats.pivot(
            index='sample_id',
            columns='contig',
            values='coverage_percentage'
        )

        fig = px.imshow(
            heatmap_data,
            title='Coverage Percentage Heatmap',
            labels=dict(x='Segment', y='Sample', color='Coverage (%)'),
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)

        return fig

    def create_all_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create all coverage visualizations.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        # Coverage overlay plot
        fig_overlay = self.create_coverage_overlay_plot(sample_ids)
        if fig_overlay.data:  # Only add if figure has data
            figures['coverage_overlay'] = fig_overlay

        # Coverage statistics bar plot
        fig_stats = self.create_coverage_stats_plot(sample_ids)
        if fig_stats.data:
            figures['coverage_stats'] = fig_stats

        # Depth distribution
        fig_depth_dist = self.create_depth_distribution_plot(sample_ids)
        if fig_depth_dist.data:
            figures['depth_distribution'] = fig_depth_dist

        # Coverage heatmap
        fig_heatmap = self.create_coverage_heatmap(sample_ids)
        if fig_heatmap.data:
            figures['coverage_heatmap'] = fig_heatmap

        # Segment-specific plots (try common segments)
        for segment in ['L', 'S']:
            fig_segment = self.create_segment_specific_plots(segment, sample_ids)
            if fig_segment.data:
                figures[f'{segment.lower()}_segment_coverage'] = fig_segment

        return figures
