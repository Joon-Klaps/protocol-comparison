"""
Consensus visualizations.

This module creates visualizations for consensus sequence analysis, genome recovery,
and average nucleotide identity (ANI) comparisons.
"""

from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from .summary_stats import ConsensusSummaryStats

logger = logging.getLogger(__name__)


class ConsensusVisualizations:
    """
    Visualization generator for consensus sequence analysis.

    Creates various plots for consensus analysis:
    - Genome recovery plots
    - ANI heatmaps
    - Recovery distribution plots
    - Segment-specific analysis plots
    """

    def __init__(self, data_path: Path):
        """
        Initialize consensus visualizations.

        Args:
            data_path: Path to data directory
        """
        self.stats = ConsensusSummaryStats(data_path)

    def create_genome_recovery_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create genome recovery bar plot.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with genome recovery percentages
        """
        recovery_df = self.stats.calculate_genome_recovery_stats(sample_ids)

        if recovery_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No genome recovery data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Create color scale based on recovery percentage
        colors = []
        for pct in recovery_df['recovery_percentage']:
            if pct >= 95:
                colors.append('green')
            elif pct >= 80:
                colors.append('orange')
            else:
                colors.append('red')

        fig = go.Figure(data=[
            go.Bar(
                x=recovery_df['sample_id'],
                y=recovery_df['recovery_percentage'],
                marker_color=colors,
                text=[f"{pct:.1f}%" for pct in recovery_df['recovery_percentage']],
                textposition='auto'
            )
        ])

        fig.update_layout(
            title='Genome Recovery Percentage by Sample',
            xaxis_title='Sample ID',
            yaxis_title='Recovery (%)',
            xaxis_tickangle=-45,
            height=500
        )

        # Add horizontal lines for thresholds
        fig.add_hline(y=95, line_dash="dash", line_color="green",
                      annotation_text="Excellent (95%)")
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                      annotation_text="Good (80%)")

        return fig

    def create_ani_heatmap(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create ANI comparison heatmap.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with ANI heatmap
        """
        ani_matrix = self.stats.calculate_ani_matrix(sample_ids)

        if ani_matrix.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No ANI data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.imshow(
            ani_matrix,
            title='Average Nucleotide Identity (ANI) Matrix',
            labels=dict(x='Sample', y='Sample', color='ANI (%)'),
            color_continuous_scale='viridis',
            aspect='auto'
        )
        fig.update_layout(height=600)

        return fig

    def create_recovery_distribution_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create genome recovery distribution histogram.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with recovery distribution
        """
        recovery_df = self.stats.calculate_genome_recovery_stats(sample_ids)

        if recovery_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No genome recovery data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.histogram(
            recovery_df,
            x='recovery_percentage',
            nbins=20,
            title='Distribution of Genome Recovery Percentages',
            labels={'recovery_percentage': 'Recovery (%)', 'count': 'Frequency'}
        )

        # Add vertical lines for thresholds
        fig.add_vline(x=95, line_dash="dash", line_color="green",
                      annotation_text="Excellent")
        fig.add_vline(x=80, line_dash="dash", line_color="orange",
                      annotation_text="Good")

        return fig

    def create_segment_recovery_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create segment-specific recovery comparison plot.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with segment recovery comparison
        """
        recovery_df = self.stats.calculate_genome_recovery_stats(sample_ids)

        if recovery_df.empty or 'segment' not in recovery_df.columns:
            fig = go.Figure()
            fig.add_annotation(
                text="No segment-specific recovery data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.box(
            recovery_df,
            x='segment',
            y='recovery_percentage',
            title='Genome Recovery by Segment',
            labels={'recovery_percentage': 'Recovery (%)', 'segment': 'Segment'}
        )

        # Add horizontal lines for thresholds
        fig.add_hline(y=95, line_dash="dash", line_color="green",
                      annotation_text="Excellent (95%)")
        fig.add_hline(y=80, line_dash="dash", line_color="orange",
                      annotation_text="Good (80%)")

        return fig

    def create_ani_distribution_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create ANI values distribution plot.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with ANI distribution
        """
        ani_matrix = self.stats.calculate_ani_matrix(sample_ids)

        if ani_matrix.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No ANI data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Extract upper triangle values (excluding diagonal)
        import numpy as np
        upper_triangle = np.triu(ani_matrix.values, k=1)
        ani_values = upper_triangle[upper_triangle > 0]

        if len(ani_values) == 0:
            fig = go.Figure()
            fig.add_annotation(
                text="No ANI comparison values available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.histogram(
            x=ani_values,
            nbins=20,
            title='Distribution of ANI Values',
            labels={'x': 'ANI (%)', 'count': 'Frequency'}
        )

        # Add vertical lines for similarity thresholds
        fig.add_vline(x=99, line_dash="dash", line_color="green",
                      annotation_text="Very High Similarity")
        fig.add_vline(x=95, line_dash="dash", line_color="orange",
                      annotation_text="High Similarity")

        return fig

    def create_recovery_vs_ani_scatter(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create scatter plot of recovery vs mean ANI for each sample.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with recovery vs ANI scatter plot
        """
        recovery_df = self.stats.calculate_genome_recovery_stats(sample_ids)
        ani_matrix = self.stats.calculate_ani_matrix(sample_ids)

        if recovery_df.empty or ani_matrix.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for recovery vs ANI comparison",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Calculate mean ANI for each sample
        import numpy as np
        sample_ani_means = []
        sample_recoveries = []
        sample_names = []

        for sample in recovery_df['sample_id'].unique():
            if sample in ani_matrix.index:
                # Get ANI values for this sample (excluding self-comparison)
                ani_values = ani_matrix.loc[sample][ani_matrix.loc[sample] < 100]
                if len(ani_values) > 0:
                    mean_ani = ani_values.mean()
                    recovery = recovery_df[recovery_df['sample_id'] == sample]['recovery_percentage'].iloc[0]

                    sample_ani_means.append(mean_ani)
                    sample_recoveries.append(recovery)
                    sample_names.append(sample)

        if not sample_ani_means:
            fig = go.Figure()
            fig.add_annotation(
                text="No paired recovery/ANI data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = px.scatter(
            x=sample_ani_means,
            y=sample_recoveries,
            text=sample_names,
            title='Genome Recovery vs Mean ANI',
            labels={'x': 'Mean ANI (%)', 'y': 'Recovery (%)'}
        )

        fig.update_traces(textposition="top center")

        return fig

    def create_all_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create all consensus visualizations.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        # Genome recovery plot
        fig_recovery = self.create_genome_recovery_plot(sample_ids)
        if fig_recovery.data:
            figures['genome_recovery'] = fig_recovery

        # ANI heatmap
        fig_ani = self.create_ani_heatmap(sample_ids)
        if fig_ani.data:
            figures['ani_heatmap'] = fig_ani

        # Recovery distribution
        fig_recovery_dist = self.create_recovery_distribution_plot(sample_ids)
        if fig_recovery_dist.data:
            figures['recovery_distribution'] = fig_recovery_dist

        # ANI distribution
        fig_ani_dist = self.create_ani_distribution_plot(sample_ids)
        if fig_ani_dist.data:
            figures['ani_distribution'] = fig_ani_dist

        # Segment-specific recovery
        fig_segment = self.create_segment_recovery_plot(sample_ids)
        if fig_segment.data:
            figures['segment_recovery'] = fig_segment

        # Recovery vs ANI scatter
        fig_scatter = self.create_recovery_vs_ani_scatter(sample_ids)
        if fig_scatter.data:
            figures['recovery_vs_ani'] = fig_scatter

        return figures
