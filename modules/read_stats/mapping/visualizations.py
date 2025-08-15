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

        # Add all 4 trace types for each segment (Raw Mapped, Raw Unmapped, % Mapped, % Unmapped)
        for segment in segments:
            segment_data = species_data[species_data['segment'] == segment]

            # Raw Mapped (initially visible)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_mapped'],
                name=f'Mapped - {segment}',
                log_y=True,
                visible=True,
                legendgroup=f'mapped_{segment}',
                marker_color='lightblue' if segment == segments[0] else 'darkblue'
            ))

            # Raw Unmapped (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_unmapped'],
                name=f'Unmapped - {segment}',
                log_y=True,
                visible=False,
                legendgroup=f'unmapped_{segment}',
                marker_color='lightcoral' if segment == segments[0] else 'darkred'
            ))

            # % Mapped (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_mapped_pct'],
                name=f'Mapped % - {segment}',
                visible=False,
                legendgroup=f'mapped_pct_{segment}',
                marker_color='lightblue' if segment == segments[0] else 'darkblue'
            ))

            # % Unmapped (initially hidden)
            fig.add_trace(go.Bar(
                x=segment_data['sample'],
                y=segment_data['reads_unmapped_pct'],
                name=f'Unmapped % - {segment}',
                visible=False,
                legendgroup=f'unmapped_pct_{segment}',
                marker_color='lightcoral' if segment == segments[0] else 'darkred'
            ))

        n_segments = len(segments)

        # Create 2x2 button layout: Data Type (Raw/%) x Read Type (Mapped/Unmapped)
        updatemenus = [
            # Data Type buttons (Raw vs Percentage)
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="Raw Counts",
                        method="update",
                        args=[{
                            "visible": [True, False, False, False] * n_segments,
                            "yaxis.title": "Number of Reads"
                        }]
                    ),
                    dict(
                        label="Percentages",
                        method="update",
                        args=[{
                            "visible": [False, False, True, False] * n_segments,
                            "yaxis.title": "Percentage (%)"
                        }]
                    )
                ]
            ),
            # Read Type buttons (Mapped vs Unmapped)
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=0.6,
                xanchor="left",
                y=1.15,
                yanchor="top",
                buttons=[
                    dict(
                        label="Mapped Reads",
                        method="update",
                        args=[{
                            "visible": [True, False, True, False] * n_segments
                        }]
                    ),
                    dict(
                        label="Unmapped Reads",
                        method="update",
                        args=[{
                            "visible": [False, True, False, True] * n_segments
                        }]
                    )
                ]
            )
        ]

        fig.update_layout(
            title=f'Mapping Statistics - {species}',
            xaxis_title='Sample ID',
            yaxis_title='Number of Reads',
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
            species_fig = self.create_interactive_mapping_plot(species, sample_ids)
            if species_fig.data:
                figures[f'mapping_{species.lower()}'] = species_fig

        return figures
