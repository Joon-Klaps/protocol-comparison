"""
Coverage visualizations.

This module creates visualizations for coverage depth analysis and genome recovery.
"""

from pathlib import Path
from typing import Dict, List, Optional, TYPE_CHECKING
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

if TYPE_CHECKING:
    from .data import CoverageDataManager
    from .summary_stats import CoverageSummaryStats
else:
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

    def __init__(self, data_path: Path, data_manager: Optional['CoverageDataManager'] = None):
        """
        Initialize coverage visualizations.

        Args:
            data_path: Path to data directory
            data_manager: Optional shared data manager instance to avoid duplicate loading
        """
        if data_manager is not None:
            self.stats = CoverageSummaryStats(data_path, data_manager=data_manager)
            self.data_manager = data_manager
        else:
            self.stats = CoverageSummaryStats(data_path)
            self.data_manager = self.stats.data_manager

        self.depth_threshold = 10  # Default minimum depth for visualizations

    def set_depth_threshold(self, threshold: int) -> None:
        """Set the minimum depth threshold for visualizations."""
        self.depth_threshold = threshold
        self.stats.set_depth_threshold(threshold)

    def create_coverage_overlay_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create coverage overlay plot showing all samples on one plot.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with coverage overlay
        """
        if not sample_ids:
            sample_ids = self.data_manager.get_available_samples()

        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=['Coverage Overlay Plot']
        )

        colors = px.colors.qualitative.Set1

        for i, sample_id in enumerate(sample_ids[:10]):  # Limit to 10 samples for readability
            sample_data = self.data_manager.get_sample_data(sample_id)

            if sample_data:
                color = colors[i % len(colors)]

                for reference, coverage_df in sample_data.items():
                    if 'depth' in coverage_df.columns and 'position' in coverage_df.columns:
                        fig.add_trace(
                            go.Scatter(
                                x=coverage_df['position'],
                                y=coverage_df['depth'],
                                mode='lines',
                                name=f'{sample_id}_{reference}',
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

    def create_recovery_stats_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create recovery statistics bar plot.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with recovery statistics
        """
        # Set depth threshold for analysis
        self.stats.set_depth_threshold(self.depth_threshold)

        # Get recovery data from data manager
        recovery_data = self.data_manager.get_recovery_data(sample_ids, self.depth_threshold)

        if not recovery_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No recovery data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Convert to DataFrame for plotting
        plot_data = []
        for sample_id, ref_data in recovery_data.items():
            for reference, recovery_value in ref_data.items():
                plot_data.append({
                    'sample_id': sample_id,
                    'reference': reference,
                    'recovery_percentage': recovery_value * 100
                })

        if not plot_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid recovery data for plotting",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        recovery_df = pd.DataFrame(plot_data)

        fig = px.bar(
            recovery_df,
            x='sample_id',
            y='recovery_percentage',
            color='reference',
            barmode='group',  # This creates dodged/grouped bars instead of stacked
            title=f'Genome Recovery Percentage by Sample and Reference (min depth: {self.depth_threshold}x)',
            labels={'recovery_percentage': 'Recovery (%)', 'sample_id': 'Sample ID'}
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            bargap=0.1,  # Gap between groups of bars
            bargroupgap=0.05  # Gap between bars within a group
        )

        return fig

    def create_depth_profile(self, sample_id: str, reference: Optional[str] = None) -> go.Figure:
        """
        Create depth profile for a specific sample.

        Args:
            sample_id: Sample identifier
            reference: Optional specific reference to plot

        Returns:
            Plotly figure with depth profile
        """
        sample_data = self.data_manager.get_sample_data(sample_id, reference)

        if not sample_data:
            fig = go.Figure()
            fig.add_annotation(
                text=f"No depth data available for sample {sample_id}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = make_subplots(
            rows=len(sample_data), cols=1,
            subplot_titles=[f'{sample_id} - {ref}' for ref in sample_data.keys()],
            shared_xaxes=True
        )

        for i, (ref, df) in enumerate(sample_data.items()):
            if 'depth' in df.columns and 'position' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['position'],
                        y=df['depth'],
                        mode='lines',
                        name=f'{ref}',
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )

        fig.update_layout(
            title=f'Depth Profile - {sample_id}',
            height=300 * len(sample_data),
            showlegend=True
        )

        return fig

    def create_freq_shift_individual_plot(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create frequency shift line plots showing individual nucleotide standard deviations.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Plotly figure with individual nucleotide frequency shift plots
        """
        if sample_ids is None:
            sample_ids = self.data_manager.get_available_samples()

        # Get frequency shift data
        freq_shift_data = self.data_manager.get_frequency_sd_data(sample_ids)

        if not freq_shift_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No frequency shift data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Create subplots for each reference
        references = list(freq_shift_data.keys())
        if not references:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid frequency shift data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        fig = make_subplots(
            rows=len(references), cols=1,
            subplot_titles=[f'{ref} - Individual Nucleotide Frequency Shifts' for ref in references],
            shared_xaxes=False,  # Independent x-axes
            vertical_spacing=0.20
        )

        nucleotide_colors = {
            'sdA': '#FF6B6B',  # Red for A
            'sdC': '#4ECDC4',  # Teal for C
            'sdG': '#45B7D1',  # Blue for G
            'sdT': '#FFA07A'   # Orange for T
        }

        for row_idx, (reference, sd_df) in enumerate(freq_shift_data.items()):
            if sd_df.empty:
                continue

            # Ensure we have the required columns
            required_cols = ['Position', 'sdA', 'sdC', 'sdG', 'sdT']
            missing_cols = [col for col in required_cols if col not in sd_df.columns]
            if missing_cols:
                logger.warning("Missing required columns %s in frequency shift data for reference %s",
                             missing_cols, reference)
                continue

            # Add traces for each nucleotide
            for nucleotide, color in nucleotide_colors.items():
                fig.add_trace(
                    go.Scatter(
                        x=sd_df['Position'],
                        y=sd_df[nucleotide],
                        mode='lines',
                        name=f'{nucleotide[-1]} ({reference})',  # Extract nucleotide letter
                        line=dict(color=color, width=2),
                        hovertemplate=f'<b>{nucleotide[-1]} - {reference}</b><br>' +
                                    'Position: %{x}<br>' +
                                    'Frequency SD: %{y:.4f}<br>' +
                                    '<extra></extra>',
                        showlegend=(row_idx == 0)  # Only show legend for first reference
                    ),
                    row=row_idx + 1, col=1
                )

        fig.update_layout(
            title='Individual Nucleotide Frequency Standard Deviations',
            height=300 * len(references),
            showlegend=True,
            hovermode='x unified'
        )

        # Update x-axis and y-axis labels for all subplots
        for i in range(len(references)):
            fig.update_xaxes(
                title_text='Genomic Position',
                range=[0, None],  # Start at 0, auto-scale max
                row=i + 1, col=1
            )
            fig.update_yaxes(title_text='Frequency SD', row=i + 1, col=1)

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
            return go.Figure()

        # Filter samples that have data for this segment
        valid_samples = []
        for sample_id in sample_ids:
            sample_data = self.data_manager.get_sample_data(sample_id)
            # Look for references containing the segment
            for reference in sample_data.keys():
                if segment.upper() in reference.upper():
                    valid_samples.append(sample_id)
                    break

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
            sample_data = self.data_manager.get_sample_data(sample_id)

            # Find the reference for this segment
            for reference, df in sample_data.items():
                if segment.upper() in reference.upper() and 'depth' in df.columns and 'position' in df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=df['position'],
                            y=df['depth'],
                            mode='lines',
                            name=f'{sample_id}',
                            line=dict(width=2)
                        ),
                        row=i+1, col=1
                    )
                    break

        fig.update_layout(
            title=f'Coverage Plots - {segment} Segment',
            height=200 * len(valid_samples),
            showlegend=True
        )

        return fig

    def create_coverage_heatmap(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create coverage recovery percentage heatmap.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with coverage heatmap
        """
        # Get recovery data
        recovery_data = self.data_manager.get_recovery_data(sample_ids, self.depth_threshold)

        if not recovery_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No recovery data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Convert to DataFrame and pivot for heatmap
        plot_data = []
        for sample_id, ref_data in recovery_data.items():
            for reference, recovery_value in ref_data.items():
                plot_data.append({
                    'sample_id': sample_id,
                    'reference': reference,
                    'recovery_percentage': recovery_value * 100
                })

        if not plot_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No valid recovery data for heatmap",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        recovery_df = pd.DataFrame(plot_data)

        # Pivot to create matrix format
        heatmap_data = recovery_df.pivot(
            index='sample_id',
            columns='reference',
            values='recovery_percentage'
        )

        fig = px.imshow(
            heatmap_data,
            title=f'Genome Recovery Percentage Heatmap (min depth: {self.depth_threshold}x)',
            labels=dict(x='Reference', y='Sample', color='Recovery (%)'),
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=600)

        return fig

    def create_depth_plots_by_reference(self, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create depth plots with one subplot per reference, showing all samples on log scale.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Plotly figure with depth plots organized by reference
        """
        if sample_ids is None:
            sample_ids = self.data_manager.get_available_samples()

        # Get all available references
        references = self.data_manager.get_available_references(sample_ids)

        if not references:
            fig = go.Figure()
            fig.add_annotation(
                text="No depth data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Create subplots for each reference
        fig = make_subplots(
            rows=len(references), cols=1,
            subplot_titles=[f'{ref} - Depth Coverage' for ref in references],
            shared_xaxes=False,  # Independent x-axes
            vertical_spacing=0.20
        )

        # Generate colors for samples
        colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Set1

        for row_idx, reference in enumerate(references):
            sample_count = 0

            for sample_idx, sample_id in enumerate(sample_ids):
                sample_data = self.data_manager.get_sample_data(sample_id, reference)

                if reference in sample_data and not sample_data[reference].empty:
                    df = sample_data[reference]

                    # Check if required columns exist
                    if 'Position' not in df.columns or 'depth' not in df.columns:
                        logger.warning("Missing Position or depth columns for sample %s, reference %s",
                                     sample_id, reference)
                        continue

                    # Filter out zero depth positions for log scale
                    df_filtered = df[df['depth'] > 0].copy()

                    if df_filtered.empty:
                        logger.warning("No positions with depth > 0 for sample %s, reference %s",
                                     sample_id, reference)
                        continue

                    color = colors[sample_idx % len(colors)]

                    fig.add_trace(
                        go.Scatter(
                            x=df_filtered['Position'],
                            y=df_filtered['depth'],
                            mode='lines',
                            name=f'{sample_id}',
                            line=dict(color=color, width=1.5),
                            hovertemplate=f'<b>{sample_id}</b><br>' +
                                        'Position: %{x}<br>' +
                                        'Depth: %{y}<br>' +
                                        '<extra></extra>',
                            showlegend=(row_idx == 0)  # Only show legend for first reference
                        ),
                        row=row_idx + 1, col=1
                    )
                    sample_count += 1

            # Add depth threshold line for each subplot
            if sample_count > 0:  # Only add threshold line if we have data
                fig.add_hline(
                    y=self.depth_threshold,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                    annotation_text=f"Depth Threshold: {self.depth_threshold}x",
                    annotation_position="top right",
                    row=row_idx + 1, col=1
                )

        # Update layout
        fig.update_layout(
            title='Depth Coverage by Reference (Log Scale)',
            height=300 * len(references),
            showlegend=True,
            hovermode='x unified'
        )

        # Update x-axis and y-axis labels for all subplots
        for i in range(len(references)):
            fig.update_xaxes(
                title_text='Genomic Position',
                range=[0, None],  # Start at 0, auto-scale max
                row=i + 1, col=1
            )
            fig.update_yaxes(
                title_text='Depth Coverage (log scale)',
                type='log',
                range= [0, 5],
                row=i + 1, col=1
            )

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
        try:
            fig_overlay = self.create_coverage_overlay_plot(sample_ids)
            if fig_overlay.data:  # Only add if figure has data
                figures['Coverage Overlay Plot'] = fig_overlay
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to create coverage overlay plot: %s", e)

        # Recovery statistics bar plot
        try:
            fig_recovery = self.create_recovery_stats_plot(sample_ids)
            if fig_recovery.data:
                figures['Genome Recovery Statistics'] = fig_recovery
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to create recovery stats plot: %s", e)

        # Frequency shift plots - Individual nucleotides
        try:
            fig_freq_shift_individual = self.create_freq_shift_individual_plot(sample_ids)
            if fig_freq_shift_individual.data:
                figures['Individual Nucleotide Frequency Shifts'] = fig_freq_shift_individual
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to create individual frequency shift plot: %s", e)

        # Depth plots by reference
        try:
            fig_depth_by_ref = self.create_depth_plots_by_reference(sample_ids)
            if fig_depth_by_ref.data:
                figures['Depth Coverage by Reference'] = fig_depth_by_ref
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to create depth plots by reference: %s", e)

        # Segment-specific plots (try common segments)
        for segment in ['L', 'S']:
            try:
                fig_segment = self.create_segment_specific_plots(segment, sample_ids)
                if fig_segment.data:
                    figures[f'{segment} Segment Coverage'] = fig_segment
            except (ValueError, KeyError, AttributeError) as e:
                logger.warning("Failed to create %s segment plot: %s", segment, e)

        return figures
