"""
Mapping visualizations.

This module creates interactive visualizations for mapping statistics.
"""

from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import logging
import streamlit as st
from ....sample_selection import label_for_sample

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
        """Create interactive mapping plot with toggle buttons.

        - Uses raw sample IDs for x categories to ensure uniqueness.
        - Displays alias labels as tick text and in hover (alias (raw ID)).
        """
        if "mapping" not in self.data or self.data["mapping"].empty:
            return go.Figure()

        mapping_df = self.data["mapping"].copy()
        if sample_ids:
            mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]

        species_data = mapping_df[mapping_df['species'] == species]
        if species_data.empty:
            return go.Figure()

        order = st.session_state.get("sample_order", [])
        if isinstance(order, list) and order:
            species_data['sample'] = species_data['sample'].astype('category')
            species_data['sample'] = species_data['sample'].cat.set_categories(order, ordered=True)
            species_data = species_data.sort_values('sample')

        species_data = species_data.copy()
        species_data['label'] = species_data['sample'].astype(str).map(label_for_sample)

        fig = go.Figure()

        segments = species_data['segment'].unique()
        segment_colors = {
            'mapped': ['#84b7f3', '#547eb4', '#2e4667'],
            'unmapped': ['#f7e4ab', '#f2bf8b', '#e5806a']
        }
        mapped_colors = {segment: segment_colors['mapped'][i % len(segment_colors['mapped'])]
                         for i, segment in enumerate(segments)}
        unmapped_colors = {segment: segment_colors['unmapped'][i % len(segment_colors['unmapped'])]
                           for i, segment in enumerate(segments)}

        for segment in segments:
            segment_data = species_data[species_data['segment'] == segment]
            custom = segment_data[['label', 'sample']].astype(str).to_numpy()

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['reads_mapped'],
                name=f'Mapped - {segment}',
                visible=True,
                legendgroup=f'mapped_{segment}',
                marker_color=mapped_colors[segment],
                customdata=custom
            ))

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['reads_unmapped'],
                name=f'Unmapped - {segment}',
                visible=False,
                legendgroup=f'unmapped_{segment}',
                marker_color=unmapped_colors[segment],
                customdata=custom
            ))

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['reads_mapped_pct'],
                name=f'Mapped % - {segment}',
                visible=False,
                legendgroup=f'mapped_pct_{segment}',
                marker_color=mapped_colors[segment],
                customdata=custom
            ))

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['reads_unmapped_pct'],
                name=f'Unmapped % - {segment}',
                visible=False,
                legendgroup=f'unmapped_pct_{segment}',
                marker_color=unmapped_colors[segment],
                customdata=custom
            ))

        n_segments = len(segments)
        visibility_patterns = {
            'raw_mapped': [True, False, False, False] * n_segments,
            'raw_unmapped': [False, True, False, False] * n_segments,
            'pct_mapped': [False, False, True, False] * n_segments,
            'pct_unmapped': [False, False, False, True] * n_segments
        }

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
                    dict(label="Raw Mapped", method="update", args=[
                        {"visible": visibility_patterns['raw_mapped']},
                        {'yaxis.title': "Number of Reads (log scale)", 'yaxis.type': "log"}
                    ]),
                    dict(label="Raw Unmapped", method="update", args=[
                        {"visible": visibility_patterns['raw_unmapped']},
                        {'yaxis.title': "Number of Reads (log scale)", 'yaxis.type': "log"}
                    ]),
                    dict(label="% Mapped", method="update", args=[
                        {"visible": visibility_patterns['pct_mapped']},
                        {'yaxis.title': "Percentage (%)", 'yaxis.type': "linear"}
                    ]),
                    dict(label="% Unmapped", method="update", args=[
                        {"visible": visibility_patterns['pct_unmapped']},
                        {'yaxis.title': "Percentage (%)", 'yaxis.type': "linear"}
                    ])
                ]
            )
        ]

        fig.update_layout(
            xaxis_title='',
            yaxis=dict(title='Number of Reads (log scale)', type='log'),
            xaxis_tickangle=-45,
            height=600,
            updatemenus=updatemenus,
            showlegend=True,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )

        sample_order = st.session_state.get("sample_order", list(species_data['sample'].astype(str).unique()))
        if isinstance(sample_order, list) and sample_order:
            present = [s for s in sample_order if s in set(species_data['sample'].astype(str))]
            if present:
                fig.update_xaxes(
                    categoryorder='array', categoryarray=present,
                    tickmode='array', tickvals=present,
                    ticktext=[label_for_sample(s) for s in present], tickangle=-45
                )

        fig.update_traces(selector=dict(type='bar'),
                          hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>%{fullData.name}: %{y}<extra></extra>')

        fig.update_traces(selector=dict(), visible=False)
        for i, vis in enumerate(visibility_patterns['raw_mapped']):
            fig.data[i].visible = vis
        return fig

    def create_interactive_mapping_umi_plot(self, species: str, sample_ids: Optional[List[str]] = None) -> go.Figure:
        """Create interactive UMI mapping plot.

        Uses raw sample IDs on x with alias tick text and dual-name hover.
        """
        if "mapping" not in self.data or self.data["mapping"].empty:
            return go.Figure()

        mapping_df = self.data["mapping"].copy()
        if sample_ids:
            mapping_df = mapping_df[mapping_df['sample'].isin(sample_ids)]

        species_data = mapping_df[mapping_df['species'] == species]
        if species_data.empty:
            return go.Figure()

        order = st.session_state.get("sample_order", [])
        if isinstance(order, list) and order:
            species_data['sample'] = species_data['sample'].astype('category')
            species_data['sample'] = species_data['sample'].cat.set_categories(order, ordered=True)
            species_data = species_data.sort_values('sample')

        species_data = species_data.copy()
        species_data['label'] = species_data['sample'].astype(str).map(label_for_sample)

        fig = go.Figure()
        segments = species_data['segment'].unique()
        segment_colors = {
            'umi_reads': ['#84b7f3', '#547eb4', '#2e4667'],
            'total_umis': ['#84b7f3', '#547eb4', '#2e4667'],
            'unique_umis': ['#84b7f3', '#547eb4', '#2e4667'],
            'pcr_cycles': ['#84b7f3', '#547eb4', '#2e4667'],
        }
        umi_reads_colors = {segment: segment_colors['umi_reads'][i % len(segment_colors['umi_reads'])]
                            for i, segment in enumerate(segments)}
        total_umis_colors = {segment: segment_colors['total_umis'][i % len(segment_colors['total_umis'])]
                             for i, segment in enumerate(segments)}
        unique_umis_colors = {segment: segment_colors['unique_umis'][i % len(segment_colors['unique_umis'])]
                              for i, segment in enumerate(segments)}
        pcr_cycles_colors = {segment: segment_colors['pcr_cycles'][i % len(segment_colors['pcr_cycles'])]
                             for i, segment in enumerate(segments)}

        for segment in segments:
            segment_data = species_data[species_data['segment'] == segment]
            custom = segment_data[['label', 'sample']].astype(str).to_numpy()

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['umi_mapping_reads'],
                name=f'UMI Reads - {segment}',
                visible=True,
                legendgroup=f'umi_reads_{segment}',
                marker_color=umi_reads_colors[segment],
                customdata=custom
            ))

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['total_UMIs'],
                name=f'Total UMIs - {segment}',
                visible=False,
                legendgroup=f'total_umis_{segment}',
                marker_color=total_umis_colors[segment],
                customdata=custom
            ))

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['unique_UMIs'],
                name=f'Unique UMIs - {segment}',
                visible=False,
                legendgroup=f'unique_umis_{segment}',
                marker_color=unique_umis_colors[segment],
                customdata=custom
            ))

            fig.add_trace(go.Bar(
                x=segment_data['sample'].astype(str),
                y=segment_data['estimated_PCR_cycles'],
                name=f'PCR Cycles - {segment}',
                visible=False,
                legendgroup=f'pcr_cycles_{segment}',
                marker_color=pcr_cycles_colors[segment],
                customdata=custom
            ))

        n_segments = len(segments)
        visibility_patterns = {
            'umi_reads': [True, False, False, False] * n_segments,
            'total_umis': [False, True, False, False] * n_segments,
            'unique_umis': [False, False, True, False] * n_segments,
            'pcr_cycles': [False, False, False, True] * n_segments
        }

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
                    dict(label="UMI Reads", method="update", args=[
                        {"visible": visibility_patterns['umi_reads']},
                        {'yaxis.title': "Number of Reads (log scale)", 'yaxis.type': "log"}
                    ]),
                    dict(label="Total UMIs", method="update", args=[
                        {"visible": visibility_patterns['total_umis']},
                        {'yaxis.title': "Number of UMIs (log scale)", 'yaxis.type': "log"}
                    ]),
                    dict(label="Unique UMIs", method="update", args=[
                        {"visible": visibility_patterns['unique_umis']},
                        {'yaxis.title': "Number of Unique UMIs (log scale)", 'yaxis.type': "log"}
                    ]),
                    dict(label="PCR Cycles", method="update", args=[
                        {"visible": visibility_patterns['pcr_cycles']},
                        {'yaxis.title': "Estimated PCR Cycles", 'yaxis.type': "linear"}
                    ])
                ]
            )
        ]

        fig.update_layout(
            xaxis_title='',
            yaxis=dict(title='Number of Reads (log scale)', type='log'),
            xaxis_tickangle=-45,
            height=600,
            updatemenus=updatemenus,
            showlegend=True,
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )

        sample_order = st.session_state.get("sample_order", list(species_data['sample'].astype(str).unique()))
        if isinstance(sample_order, list) and sample_order:
            present = [s for s in sample_order if s in set(species_data['sample'].astype(str))]
            if present:
                fig.update_xaxes(
                    categoryorder='array', categoryarray=present,
                    tickmode='array', tickvals=present,
                    ticktext=[label_for_sample(s) for s in present], tickangle=-45
                )

        fig.update_traces(selector=dict(type='bar'),
                          hovertemplate='<b>%{customdata[0]} (%{customdata[1]})</b><br>%{fullData.name}: %{y}<extra></extra>')

        fig.update_traces(selector=dict(), visible=False)
        for i, vis in enumerate(visibility_patterns['umi_reads']):
            fig.data[i].visible = vis
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
            if mapping_fig.data or mapping_fig.layout.annotations:
                figures[f'Mapping Statistics - {species}'] = mapping_fig
            umi_fig = self.create_interactive_mapping_umi_plot(species, sample_ids)
            if umi_fig.data or umi_fig.layout.annotations:
                figures[f'UMI Statistics - {species}'] = umi_fig

        return figures
