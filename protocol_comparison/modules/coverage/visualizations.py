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
import math
from ...sample_selection import (
    get_current_sample_order,
    label_for_sample,
)

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

    def _format_reference_label(self, reference: str) -> str:
        """Return a human-friendly label for a reference.

        Falls back to the raw reference if metadata is missing.
        """
        meta = self.data_manager.get_species_segment_for_reference(reference)
        if isinstance(meta, dict) and meta:
            vals = [str(v) for v in meta.values() if v is not None]
            if vals:
                return f"{reference} - {' '.join(vals)}"
        return str(reference)

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
                    'reference': self._format_reference_label(reference),
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

        # Apply desired sample order from session state if present, then derive display labels
        sample_order: List[str] = get_current_sample_order([])
        if sample_order:
            recovery_df['sample_id'] = pd.Categorical(recovery_df['sample_id'], categories=sample_order, ordered=True)
            recovery_df = recovery_df.sort_values('sample_id')
        # Alias labels for display and custom hover
        recovery_df['alias_label'] = recovery_df['sample_id'].astype(str).map(label_for_sample)

        fig = px.bar(
            recovery_df,
            x='sample_id',
            y='recovery_percentage',
            color='reference',
            barmode='group',
            title=f'Genome Recovery Percentage by Sample and Reference (min depth: {self.depth_threshold}x)',
            labels={'recovery_percentage': 'Recovery (%)', 'sample_id': 'Sample'},
            custom_data=['alias_label', 'sample_id']
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=500,
            bargap=0.1,  # Gap between groups of bars
            bargroupgap=0.05  # Gap between bars within a group
        )

        # Ensure x-axis respects raw sample order, but display alias labels
        sample_order = get_current_sample_order([])
        if sample_order:
            fig.update_xaxes(
                categoryorder='array',
                categoryarray=sample_order,
                tickmode='array',
                tickvals=sample_order,
                ticktext=[label_for_sample(s) for s in sample_order]
            )

        # Add hover info with both alias and raw ID per bar
        fig.update_traces(hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>'
                                         'Reference: %{fullData.name}<br>'
                                         'Recovery: %{y:.2f}%<extra></extra>')

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
            subplot_titles=[f'{label_for_sample(sample_id)} - {ref}' for ref in sample_data.keys()],
            shared_xaxes=True
        )

        for i, (ref, df) in enumerate(sample_data.items()):
            if 'depth' in df.columns and 'POS' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['POS'],
                        y=df['depth'],
                        mode='lines',
                        name=f'{ref}',
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )

        fig.update_layout(
            title=f'Depth Profile - {label_for_sample(sample_id)}',
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
            fig = go.Figure()
            fig.add_annotation(
                text="No samples selected, please select samples",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

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
            subplot_titles=[self._format_reference_label(ref) for ref in references],
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
            required_cols = ['POS', 'sdA', 'sdC', 'sdG', 'sdT']
            missing_cols = [col for col in required_cols if col not in sd_df.columns]
            if missing_cols:
                logger.warning("Missing required columns %s in frequency shift data for reference %s",
                             missing_cols, reference)
                continue

            # Add traces for each nucleotide
            for nucleotide, color in nucleotide_colors.items():
                fig.add_trace(
                    go.Scatter(
                        x=sd_df['POS'],
                        y=sd_df[nucleotide],
                        mode='lines',
                        name=f'{nucleotide[-1]}',  # Extract nucleotide letter
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

    def create_species_segment_specific_plots(self, species:str, segment: str, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create segment-specific coverage plots.

        Args:
            segment: Segment identifier (e.g., 'L', 'S')
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Plotly figure for the segment
        """
        if not sample_ids:
            fig = go.Figure()
            fig.add_annotation(
                text="No samples selected, please select samples",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Get references for this segment using annotation data
        references = self.data_manager.get_references_for_species_segment(species, segment)


        if not references:
            fig = go.Figure()
            return fig

        # Filter samples that have data for any of the segment's references
        valid_samples = []
        sample_reference_map = {}
        for sample_id in sample_ids:
            sample_data = self.data_manager.get_sample_data(sample_id)
            for ref in references:
                if ref in sample_data:
                    valid_samples.append(sample_id)
                    sample_reference_map[sample_id] = ref
                    break

        if not valid_samples:
            fig = go.Figure()
            return fig

        # Create subplot for each sample
        fig = make_subplots(
            rows=len(valid_samples), cols=1,
            subplot_titles=[f'{label_for_sample(sample_id)}, {species} - {segment} ({sample_reference_map[sample_id]})' for sample_id in valid_samples],
            shared_xaxes=True
        )

        # Track maximum depth across all included samples to enable standardized Y-axis scaling
        max_depth_value = 0

        for i, sample_id in enumerate(valid_samples):
            reference = sample_reference_map[sample_id]
            sample_data = self.data_manager.get_sample_data(sample_id, reference)
            df = sample_data[reference]

            # Update global max depth
            if 'depth' in df.columns and not df['depth'].empty:
                depth_series = pd.to_numeric(df['depth'], errors='coerce')
                if depth_series.notna().any():
                    max_depth_value = max(max_depth_value, float(depth_series.max()))
                else:
                    logger.warning("Depth column contains no numeric values for sample %s, reference %s", sample_id, reference)

            fig.add_trace(
                go.Scatter(
                    x=df['POS'],
                    y=df['depth'],
                    mode='lines',
                    name=f'{label_for_sample(sample_id)}',
                    line=dict(width=2)
                ),
                row=i+1, col=1
            )

        fig.update_layout(
            title=f'Coverage Plots - {segment} Segment',
            height=200 * len(valid_samples),
            showlegend=True
        )

        # Add buttons to toggle Y-axis scaling across subplots: Auto vs Standardized (0 .. max)
        if max_depth_value > 0 and len(valid_samples) > 0:
            # Build relayout payloads for all subplot y-axes using per-axis keys
            auto_axes_updates = {}
            std_axes_updates = {}
            for idx in range(1, len(valid_samples) + 1):
                axis_key = 'yaxis' if idx == 1 else f'yaxis{idx}'
                auto_axes_updates[f'{axis_key}.autorange'] = True
                # Remove any fixed range if present when switching back to autorange
                auto_axes_updates[f'{axis_key}.range'] = None
                std_axes_updates[f'{axis_key}.autorange'] = False
                std_axes_updates[f'{axis_key}.range'] = [0, max_depth_value]

            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        # Default to 'Auto Y' (index 0) since autorange is the initial state
                        active=0,
                        buttons=[
                            dict(
                                label="Auto Y",
                                method="relayout",
                                args=[auto_axes_updates]
                            ),
                            dict(
                                label="Standardize Y",
                                method="relayout",
                                args=[std_axes_updates]
                            ),
                        ],
                        x=-0.05,
                        xanchor="left",
                        y=1.3,
                        yanchor="top",
                        pad=dict(t=2, r=2, b=2, l=2)
                    )
                ]
            )

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
            fig = go.Figure()
            fig.add_annotation(
                text="No samples selected, please select samples",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

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
            subplot_titles=[self._format_reference_label(ref) for ref in references],
            shared_xaxes=False,  # Independent x-axes
            vertical_spacing=0.20
        )

        # Generate colors for samples
        colors = px.colors.qualitative.Set1
        # Track global max depth for standardized scaling across subplots
        max_depth_value = 0.0

        for row_idx, reference in enumerate(references):
            sample_count = 0

            for sample_idx, sample_id in enumerate(sample_ids):
                sample_data = self.data_manager.get_sample_data(sample_id, reference)

                if reference in sample_data and not sample_data[reference].empty:
                    df = sample_data[reference]

                    # Check if required columns exist
                    if 'POS' not in df.columns:
                        logger.warning(
                            "Missing POS column for sample %s, reference %s",
                            sample_id, reference,
                        )
                        continue

                    if 'depth' not in df.columns:
                        logger.warning(
                            "Missing depth columns for sample %s, reference %s",
                            sample_id, reference,
                        )
                        continue

                    # Filter out zero depth positions for log scale and hover sanity
                    df_filtered = df[df['depth'] > 0].copy()

                    if df_filtered.empty:
                        logger.warning(
                            "No positions with depth > 0 for sample %s, reference %s",
                            sample_id, reference,
                        )
                        continue

                    # Update global max depth from the plotted data
                    try:
                        max_depth_value = max(
                            max_depth_value, float(df_filtered['depth'].max())
                        )
                    except (TypeError, ValueError):
                        logger.warning(
                            "Non-numeric depths encountered for sample %s, reference %s",
                            sample_id, reference,
                        )

                    color = colors[sample_idx % len(colors)]

                    fig.add_trace(
                        go.Scatter(
                            x=df_filtered['POS'],
                            y=df_filtered['depth'],
                            mode='lines',
                            name=f'{label_for_sample(sample_id)}',
                            line=dict(color=color, width=1.5),
                            hovertemplate=(
                                f'<b>{label_for_sample(sample_id)}</b><br>'
                                'Position: %{x}<br>'
                                'Depth: %{y}<br>'
                                '<extra></extra>'
                            ),
                            showlegend=(row_idx == 0),  # Only show legend for first reference
                        ),
                        row=row_idx + 1, col=1,
                    )
                    sample_count += 1

            # Add depth threshold line for each subplot where we have any data
            if sample_count > 0:
                fig.add_hline(
                    y=self.depth_threshold,
                    line_dash="dash",
                    line_color="red",
                    line_width=2,
                )

        # Update layout
        fig.update_layout(
            title='Depth Coverage by Reference (Log Scale)',
            height=300 * len(references),
            showlegend=True,
            hovermode='x unified',
        )

        # Initial axis setup (log scale)
        log_upper = math.log10(max_depth_value) if max_depth_value > 0 else 0
        for i in range(len(references)):
            fig.update_xaxes(
                title_text='Genomic Position',
                range=[0, None],  # Start at 0, auto-scale max
                row=i + 1, col=1,
            )
            fig.update_yaxes(
                title_text='Depth Coverage (log scale)',
                type='log',
                range=[0, math.ceil(log_upper)] if max_depth_value > 0 else [0, 1],
                row=i + 1, col=1,
            )

        # Buttons to toggle Y-axis type with standardized max across subplots
        if len(references) > 0 and max_depth_value > 0:
            linear_updates = {}
            log_updates = {}
            log_upper_ceiled = math.ceil(log_upper)
            for idx in range(1, len(references) + 1):
                axis_key = 'yaxis' if idx == 1 else f'yaxis{idx}'
                # Linear: fixed 0..max range
                linear_updates[f'{axis_key}.type'] = 'linear'
                linear_updates[f'{axis_key}.autorange'] = False
                linear_updates[f'{axis_key}.range'] = [0, max_depth_value]
                # Log: fixed 10^0..10^ceil(log10(max))
                log_updates[f'{axis_key}.type'] = 'log'
                log_updates[f'{axis_key}.autorange'] = False
                log_updates[f'{axis_key}.range'] = [0, log_upper_ceiled]

            fig.update_layout(
                updatemenus=[
                    dict(
                        type='buttons',
                        direction='right',
                        # Default to 'Log Y' (index 1) because axes are initialized as log
                        active=1,
                        buttons=[
                            dict(
                                label='Linear Y',
                                method='relayout',
                                args=[linear_updates],
                            ),
                            dict(
                                label='Log Y',
                                method='relayout',
                                args=[log_updates],
                            ),
                        ],
                        x=-0.05,
                        xanchor='left',
                        y=1.3,
                        yanchor='top',
                        pad=dict(t=2, r=2, b=2, l=2),
                    )
                ]
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

        # Coverage overlay plot - method not implemented yet
        # try:
        #     fig_overlay = self.create_coverage_overlay_plot(sample_ids)
        #     if fig_overlay.data:  # Only add if figure has data
        #         figures['Coverage Overlay Plot'] = fig_overlay
        # except (ValueError, KeyError, AttributeError) as e:
        #     logger.warning("Failed to create coverage overlay plot: %s", e)

        # Recovery statistics bar plot
        try:
            fig_recovery = self.create_recovery_stats_plot(sample_ids)
            if fig_recovery.data or fig_recovery.layout.annotations:
                figures['Genome Recovery Statistics'] = fig_recovery
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to create recovery stats plot: %s", e)

        # Frequency shift plots - Individual nucleotides
        try:
            fig_freq_shift_individual = self.create_freq_shift_individual_plot(sample_ids)
            if fig_freq_shift_individual.data or fig_freq_shift_individual.layout.annotations:
                figures['Individual Nucleotide Frequency Shifts'] = fig_freq_shift_individual
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to create individual frequency shift plot: %s", e)

        # Depth plots by reference
        try:
            fig_depth_by_ref = self.create_depth_plots_by_reference(sample_ids)
            if fig_depth_by_ref.data or fig_depth_by_ref.layout.annotations:
                figures['Depth Coverage by Reference'] = fig_depth_by_ref
        except (ValueError, KeyError, AttributeError) as e:
            logger.warning("Failed to create depth plots by reference: %s", e)

        # Segment-specific plots (try common segments)
        for species in ['LASV', 'HAZV']:
            for segment in ['L', 'S', 'M']:
                try:
                    fig_segment = self.create_species_segment_specific_plots(species,segment, sample_ids)
                    if fig_segment.data or fig_segment.layout.annotations:
                        figures[f'{species} {segment} Segment Coverage'] = fig_segment
                except (ValueError, KeyError, AttributeError) as e:
                    logger.warning("Failed to create %s %s segment plot: %s", species, segment, e)

        return figures
