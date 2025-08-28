"""
Read processing visualizations.

This module creates visualizations for read count changes through the processing pipeline.
"""

from pathlib import Path
from typing import Dict, List, Optional
import plotly.graph_objects as go
import logging
import streamlit as st
import pandas as pd

from .summary_stats import ReadProcessingDataManager
from ....sample_selection import (
    label_for_sample,
)

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

        # Legend order for line chart can follow sample_order
        order = st.session_state.get("sample_order", [])
        if isinstance(order, list) and order:
            # Reorder DataFrame rows to emit traces in desired legend order
            # Keep those in order first then append the rest
            in_order = reads_df[reads_df['sample'].isin(order)]
            in_order['sample'] = in_order['sample'].astype('category').cat.set_categories(order, ordered=True)
            in_order = in_order.sort_values('sample')
            rest = reads_df[~reads_df['sample'].isin(order)]
            reads_df = pd.concat([in_order, rest], ignore_index=True)

        fig = go.Figure()

        # Processing steps
        steps = ['Raw', 'Post-trimming', 'Post-host removal']

        # Add a trace for each sample
        for _, row in reads_df.iterrows():
            sample_id = row['sample']
            display_label = label_for_sample(str(sample_id))
            counts = [row['raw_reads'], row['post_trimming_reads'], row['post_host_removal_reads']]

            fig.add_trace(go.Scatter(
                x=steps,
                y=counts,
                mode='lines+markers',
                name=display_label,
                line=dict(width=2),
                marker=dict(size=6),
                hovertemplate=f'<b>{display_label}</b><br>' +
                             'Step: %{x}<br>' +
                             'Reads: %{y:,}<extra></extra>'
            ))

        # Add buttons for log/linear y-axis and update title
        updatemenus = [
            dict(
                type="buttons",
                direction="right",
                showactive=True,
                x=-0.05,
                xanchor="left",
                y=1.5,
                yanchor="top",
                buttons=[
                    dict(label="Log Scale", method="relayout", args=[
                        {"yaxis.type": "log", "yaxis.title": "Number of Reads (log scale)", "title.text": "Read Processing Pipeline - Count Changes by Sample (Log Scale)"}
                    ]),
                    dict(label="Linear Scale", method="relayout", args=[
                        {"yaxis.type": "linear", "yaxis.title": "Number of Reads", "title.text": "Read Processing Pipeline - Count Changes by Sample (Linear Scale)"}
                    ])
                ]
            )
        ]

        fig.update_layout(
            title='Read Processing Pipeline - Count Changes by Sample (Log Scale)',
            xaxis_title='Processing Step',
            yaxis_title='Number of Reads (log scale)',
            height=600,
            hovermode='closest',
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1.3,
                xanchor="left",
                x=1.02
            ),
            updatemenus=updatemenus
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

        return figures
