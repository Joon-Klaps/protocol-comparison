"""
Mapping visualizations.

This module creates interactive visualizations for mapping statistics.
"""

from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import logging

from .summary_stats import MappingDataManager

logger = logging.getLogger(__name__)


class MappingVisualizations:
    """
    Visualization creator for mapping statistics.

    Creates interactive plots with 2x2 button controls for data exploration.
    """

    def __init__(self, data_path: Path):
        """
        Initialize mapping visualizations.

        Args:
            data_path: Path to data directory
        """
        self.data_manager = MappingDataManager(data_path)
        self.data = self.data_manager.load_data()

    def create_interactive_mapping_plot(self, species: str, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive mapping plot for a specific species with 2x2 button controls.

        Args:
            species: Species to visualize
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with interactive controls
        """
        if "mapping" not in self.data or self.data["mapping"].empty:
            return go.Figure()

        mapping_df = self.data["mapping"].copy()
        if sample_ids:
            mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]

        # Filter for specific species
        species_data = mapping_df[mapping_df['species'] == species]

        if species_data.empty:
            return go.Figure()

        fig = go.Figure()

        # Get unique segments for this species
        segments = species_data['segment'].unique()

        # Define color palette for segments - shades of blue and red for 3 segments
        segment_colors = {
            'mapped': ['#84b7f3', '#547eb4', '#2e4667'],  # Light Blue, Medium Blue, Dark Blue
            'unmapped': ['#f7e4ab', '#f2bf8b', '#e5806a']  # Light Red, Medium Red, Dark Red
        }

        # Create color mappings for each segment
        mapped_colors = {segment: segment_colors['mapped'][i % len(segment_colors['mapped'])]
                        for i, segment in enumerate(segments)}
        unmapped_colors = {segment: segment_colors['unmapped'][i % len(segment_colors['unmapped'])]
                          for i, segment in enumerate(segments)}

        # Add all 4 trace types for each segment (Raw Mapped, Raw Unmapped, % Mapped, % Unmapped)
        for segment in segments:
            segment_data = species_data[species_data['segment'] == segment]

            # Raw Mapped (initially visible)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_mapped'],
                name=f'Mapped - {segment}',
                visible=True,
                legendgroup=f'mapped_{segment}',
                marker_color=mapped_colors[segment]
            ))

            # Raw Unmapped (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_unmapped'],
                name=f'Unmapped - {segment}',
                visible=False,
                legendgroup=f'unmapped_{segment}',
                marker_color=unmapped_colors[segment]
            ))

            # % Mapped (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_mapped_pct'],
                name=f'Mapped % - {segment}',
                visible=False,
                legendgroup=f'mapped_pct_{segment}',
                marker_color=mapped_colors[segment]
            ))

            # % Unmapped (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_unmapped_pct'],
                name=f'Unmapped % - {segment}',
                visible=False,
                legendgroup=f'unmapped_pct_{segment}',
                marker_color=unmapped_colors[segment]
            ))

        n_segments = len(segments)

        # Create visibility patterns for all 4 combinations
        # [Raw Mapped, Raw Unmapped, % Mapped, % Unmapped] per segment
        visibility_patterns = {
            'raw_mapped': [True, False, False, False] * n_segments,
            'raw_unmapped': [False, True, False, False] * n_segments,
            'pct_mapped': [False, False, True, False] * n_segments,
            'pct_unmapped': [False, False, False, True] * n_segments
        }

        # Create single row with all 4 buttons
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=-0.05,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="Raw Mapped",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['raw_mapped']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Number of Reads (log scale)",
                                'yaxis.type': "log"
                            }
                        ]
                    ),
                    dict(
                        label="Raw Unmapped",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['raw_unmapped']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Number of Reads (log scale)",
                                'yaxis.type': "log"
                            }
                        ]
                    ),
                    dict(
                        label="% Mapped",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['pct_mapped']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Percentage (%)",
                                'yaxis.type': "linear"
                            }
                        ]
                    ),
                    dict(
                        label="% Unmapped",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['pct_unmapped']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Percentage (%)",
                                'yaxis.type': "linear"
                            }
                        ]
                    )
                ]
            )
        ]

        fig.update_layout(
            xaxis_title='',
            yaxis=dict(
                title='Number of Reads (log scale)',
                type='log'
            ),
            xaxis_tickangle=-45,
            height=600,
            updatemenus=updatemenus,
            showlegend=True,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        # Set initial visibility to Raw Mapped
        for i, trace in enumerate(fig.data):
            trace.visible = visibility_patterns['raw_mapped'][i]

        return fig

    def create_interactive_mapping_umi_plot(self, species: str, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """
        Create interactive UMI mapping plot for a specific species with button controls for UMI metrics.

        Displays UMI-related statistics including deduplicated reads, total UMIs, unique UMIs,
        and estimated PCR cycles. PCR cycles are calculated as: Total UMIs / Unique UMIs.

        Args:
            species: Species to visualize
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Plotly figure with interactive controls for UMI statistics
        """
        if "mapping" not in self.data or self.data["mapping"].empty:
            return go.Figure()

        mapping_df = self.data["mapping"].copy()
        if sample_ids:
            mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]

        # Filter for specific species
        species_data = mapping_df[mapping_df['species'] == species]

        if species_data.empty:
            return go.Figure()

        fig = go.Figure()

        # Get unique segments for this species
        segments = species_data['segment'].unique()

        # Define color palette for segments - different shades for UMI metrics
        segment_colors = {
            'umi_reads': ['#84b7f3', '#547eb4', '#2e4667'],
            'total_umis': ['#84b7f3', '#547eb4', '#2e4667'],
            'unique_umis': ['#84b7f3', '#547eb4', '#2e4667'],
            'pcr_cycles': ['#84b7f3', '#547eb4', '#2e4667'],
        }

        # Create color mappings for each segment
        umi_reads_colors = {segment: segment_colors['umi_reads'][i % len(segment_colors['umi_reads'])]
                           for i, segment in enumerate(segments)}
        total_umis_colors = {segment: segment_colors['total_umis'][i % len(segment_colors['total_umis'])]
                            for i, segment in enumerate(segments)}
        unique_umis_colors = {segment: segment_colors['unique_umis'][i % len(segment_colors['unique_umis'])]
                             for i, segment in enumerate(segments)}
        pcr_cycles_colors = {segment: segment_colors['pcr_cycles'][i % len(segment_colors['pcr_cycles'])]
                            for i, segment in enumerate(segments)}

        # Add all 4 UMI trace types for each segment
        for segment in segments:
            segment_data = species_data[species_data['segment'] == segment]

            # UMI Deduplicated Reads (initially visible)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['umi_mapping_reads'],
                name=f'UMI Reads - {segment}',
                visible=True,
                legendgroup=f'umi_reads_{segment}',
                marker_color=umi_reads_colors[segment]
            ))

            # Total UMIs (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['total_UMIs'],
                name=f'Total UMIs - {segment}',
                visible=False,
                legendgroup=f'total_umis_{segment}',
                marker_color=total_umis_colors[segment]
            ))

            # Unique UMIs (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['unique_UMIs'],
                name=f'Unique UMIs - {segment}',
                visible=False,
                legendgroup=f'unique_umis_{segment}',
                marker_color=unique_umis_colors[segment]
            ))

            # Estimated PCR Cycles (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['estimated_PCR_cycles'],
                name=f'PCR Cycles - {segment}',
                visible=False,
                legendgroup=f'pcr_cycles_{segment}',
                marker_color=pcr_cycles_colors[segment]
            ))

        n_segments = len(segments)

        # Create visibility patterns for all 4 UMI metrics
        # [UMI Reads, Total UMIs, Unique UMIs, PCR Cycles] per segment
        visibility_patterns = {
            'umi_reads': [True, False, False, False] * n_segments,
            'total_umis': [False, True, False, False] * n_segments,
            'unique_umis': [False, False, True, False] * n_segments,
            'pcr_cycles': [False, False, False, True] * n_segments
        }

        # Create single row with all 4 UMI metric buttons
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=-0.05,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="UMI Reads",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['umi_reads']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Number of Reads (log scale)",
                                'yaxis.type': "log"
                            }
                        ]
                    ),
                    dict(
                        label="Total UMIs",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['total_umis']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Number of UMIs (log scale)",
                                'yaxis.type': "log"
                            }
                        ]
                    ),
                    dict(
                        label="Unique UMIs",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['unique_umis']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Number of Unique UMIs (log scale)",
                                'yaxis.type': "log"
                            }
                        ]
                    ),
                    dict(
                        label="PCR Cycles",
                        method="update",
                        args=[
                            {"visible": visibility_patterns['pcr_cycles']},  # Data updates
                            {  # Layout updates
                                'yaxis.title': "Estimated PCR Cycles",
                                'yaxis.type': "linear"
                            }
                        ]
                    )
                ]
            )
        ]

        fig.update_layout(
            xaxis_title='',
            yaxis=dict(
                title='Number of Reads (log scale)',
                type='log'
            ),
            xaxis_tickangle=-45,
            height=600,
            updatemenus=updatemenus,
            showlegend=True,
            barmode='group',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            )
        )

        # Set initial visibility to UMI Reads
        for i, trace in enumerate(fig.data):
            trace.visible = visibility_patterns['umi_reads'][i]

        return fig

    def create_all_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, go.Figure]:
        """
        Create all mapping visualizations.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        if "mapping" not in self.data or self.data["mapping"].empty:
            return figures

        mapping_df = self.data["mapping"].copy()
        if sample_ids:
            mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]

        # Create interactive plots for each species
        species_list = mapping_df['species'].unique()
        for species in species_list:
            mapping_fig = self.create_interactive_mapping_plot(species, sample_ids)
            if mapping_fig.data:
                figures[f'Mapping Statistics - {species}'] = mapping_fig
            umi_fig = self.create_interactive_mapping_umi_plot(species, sample_ids)
            if umi_fig.data:
                figures[f'UMI Statistics - {species}'] = umi_fig

        return figures
