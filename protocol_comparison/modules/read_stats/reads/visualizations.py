"""
Read processing visualizations.

This module creates visualizations for read count changes through the processing pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import logging

from .summary_stats import ReadProcessingDataManager

logger = logging.getLogger(__name__)


class ReadProcessingVisualizations:
    """
    Visualization creator for read processing pipeline statistics.

    Creates line plots showing read count changes through processing steps.
    """

    def __init__(self, data_path: Path):
        """
        Initialize read processing visualizations.

        Args:
            data_path: Path to data directory
        """
        self.data_manager = ReadProcessingDataManager(data_path)
        self.data = self.data_manager.load_data()

    def create_processing_timeline(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create read processing timeline visualization.
        X-axis: processing steps, Y-axis: counts, each sample as a colored line.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure for read processing timeline
        """
        if "reads" not in self.data or self.data["reads"].empty:
            return go.Figure()

        reads_df = self.data["reads"].copy()
        if sample_ids:
            reads_df = reads_df[reads_df['sample'].isin(sample_ids)]

        fig = go.Figure()

        # Processing steps
        steps = ['Raw', 'Post-trimming', 'Post-host removal']

        # Add a trace for each sample
        for _, row in reads_df.iterrows():
            sample_id = row['sample']
            counts = [row['raw_reads'], row['post_trimming_reads'], row['post_host_removal_reads']]

            fig.add_trace(go.Scatter(
                x=steps,
                y=counts,
                mode='lines+markers',
                name=sample_id,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{sample_id}</b><br>' +
                             'Step: %{x}<br>' +
                             'Reads: %{y:,}<extra></extra>'
            ))

        fig.update_layout(
            title='Read Processing Pipeline - Count Changes by Sample',
            xaxis_title='Processing Step',
            yaxis_title='Number of Reads',
            height=600,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )

        return fig

    def create_efficiency_overview(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create efficiency overview showing retention percentages.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure for efficiency overview
        """
        if "reads" not in self.data or self.data["reads"].empty:
            return go.Figure()

        reads_df = self.data["reads"].copy()
        if sample_ids:
            reads_df = reads_df[reads_df['sample'].isin(sample_ids)]

        # Calculate efficiency metrics
        reads_df['trimming_retention'] = reads_df['post_trimming_reads'] / reads_df['raw_reads'] * 100
        reads_df['host_removal_retention'] = reads_df['post_host_removal_reads'] / reads_df['post_trimming_reads'] * 100
        reads_df['overall_retention'] = reads_df['post_host_removal_reads'] / reads_df['raw_reads'] * 100

        fig = go.Figure()

        # Add traces for each efficiency metric
        fig.add_trace(go.Bar(
            x=reads_df['sample'],
            y=reads_df['trimming_retention'],
            name='Trimming Retention %',
            marker_color='lightblue'
        ))

        fig.add_trace(go.Bar(
            x=reads_df['sample'],
            y=reads_df['host_removal_retention'],
            name='Host Removal Retention %',
            marker_color='lightcoral'
        ))

        fig.add_trace(go.Bar(
            x=reads_df['sample'],
            y=reads_df['overall_retention'],
            name='Overall Retention %',
            marker_color='lightgreen'
        ))

        fig.update_layout(
            title='Read Processing Efficiency by Sample',
            xaxis_title='Sample ID',
            yaxis_title='Retention Percentage (%)',
            xaxis_tickangle=-45,
            height=600,
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        return fig

    def create_all_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create all read processing visualizations.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        # Timeline visualization
        timeline_fig = self.create_processing_timeline(sample_ids)
        if timeline_fig.data or timeline_fig.layout.annotations:
            figures['processing_timeline'] = timeline_fig

        # Efficiency overview
        efficiency_fig = self.create_efficiency_overview(sample_ids)
        if efficiency_fig.data or efficiency_fig.layout.annotations:
            figures['efficiency_overview'] = efficiency_fig

        return figures
