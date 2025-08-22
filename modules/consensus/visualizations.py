"""
Consensus visualizations.

This module creates visualizations for consensus sequence analysis and alignment visualization
using dash_bio components for interactive multiple sequence alignment display and clustering.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

from .data import ConsensusDataManager

logger = logging.getLogger(__name__)


class ConsensusVisualizations:
    """
    Visualization generator for consensus sequence analysis and alignment visualization.

    Creates various plots for consensus analysis:
    - Multiple sequence alignment visualization using dash_bio
    - Pairwise identity clustergram using dash_bio
    - Alignment statistics plots
    """

    def __init__(self, data_path: Path):
        """
        Initialize consensus visualizations.

        Args:
            data_path: Path to data directory
        """
        self.data_path = data_path
        self.data_manager = ConsensusDataManager(data_path)

    def create_alignment_visualization(
        self,
        selected_key: Tuple[str, str, str],
        sample_ids: Optional[List[str]] = None,
        max_sequences: int = 50
    ) -> Optional[Dict[str, Any]]:
        """
        Create interactive multiple sequence alignment visualization using dash_bio.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs to visualize
            max_sequences: Maximum number of sequences to display (for performance)

        Returns:
            Dictionary containing dash_bio AlignmentChart component data or None
        """
                # Get alignment data for the selected key
        alignment_data = self.data_manager.alignment_data.get(selected_key, {})

        if not alignment_data:
            logger.warning("No alignment data available for key: %s", selected_key)
            return None

        # Filter by sample IDs if provided
        if sample_ids:
            alignment_data = {
                sample_id: seq_record
                for sample_id, seq_record in alignment_data.items()
                if sample_id in sample_ids
            }

        if not alignment_data:
            logger.warning("No sequences found for selected samples")
            return None

        # Limit number of sequences for performance
        if len(alignment_data) > max_sequences:
            # Take first N sequences (could be randomized or prioritized)
            sample_items = list(alignment_data.items())[:max_sequences]
            alignment_data = dict(sample_items)

        # Format for dash_bio using data manager helper
        try:
            dash_bio_data = self.data_manager.format_alignment_for_dash_bio(alignment_data)

            if not dash_bio_data:
                return None

            # Create AlignmentChart component configuration
            alignment_config = {
                'data': dash_bio_data,
                'height': 600,
                'tilewidth': 30,
                'showlabel': True,
                'showid': True,
                'colorscale': 'clustal2',  # Use clustal2 coloring scheme
                'textcolor': 'white',
                'textsize': 10,
                'showconservation': True,
                'conservationcolor': 'Blackbody',
                'conservationcolorscale': 'Viridis',
                'conservationopacity': 0.8,
                'conservationmethod': 'entropy'
            }

            return alignment_config

        except Exception as e:
            logger.error("Error creating alignment visualization: %s", e)
            return None

    def create_identity_clustergram(
        self,
        selected_key: Tuple[str, str, str],
        sample_ids: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Create pairwise identity clustergram using dash_bio.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs to include

        Returns:
            Dictionary containing dash_bio clustergram component data or None
        """
        try:
            # Get identity matrix using data manager helper
            selected_alignment = self.data_manager.alignment_data.get(selected_key, {})

            # Filter by sample IDs if provided
            if sample_ids:
                selected_alignment = {
                    sample_id: seq_record
                    for sample_id, seq_record in selected_alignment.items()
                    if sample_id in sample_ids
                }

            if not selected_alignment or len(selected_alignment) < 2:
                logger.warning("Insufficient data for identity clustergram")
                return None

            identity_matrix = self.data_manager.compute_pairwise_identity_matrix(selected_alignment)
            sample_labels = list(selected_alignment.keys())

            # Create clustergram configuration for dash_bio
            clustergram_config = {
                'data': identity_matrix.tolist(),  # Convert numpy array to list
                'row_labels': sample_labels,
                'column_labels': sample_labels,
                'height': 600,
                'width': 600,
                'display_ratio': [0.1, 0.7, 0.2],  # [dendro, heatmap, dendro]
                'cluster': 'all',  # Cluster both rows and columns
                'line_width': 2,
                'color_threshold': {
                    'row': 0.7,
                    'col': 0.7
                },
                'hidden_labels': 'none',
                'colorscale': [
                    [0, '#440154'],    # Low identity - purple
                    [0.5, '#21908c'],  # Medium identity - teal
                    [1, '#fde725']     # High identity - yellow
                ],
                'center_values': False,
                'standardize': 'none',
                'row_sort_values': True,
                'column_sort_values': True
            }

            return clustergram_config

        except Exception as e:
            logger.error("Error creating identity clustergram: %s", e)
            return None

    def create_alignment_statistics_plot(
        self,
        selected_key: Tuple[str, str, str],
        sample_ids: Optional[List[str]] = None
    ) -> go.Figure:
        """
        Create alignment statistics visualization.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Plotly figure with alignment statistics
        """
        # Get alignment data
        alignment_data = self.data_manager.alignment_data.get(selected_key, {})

        if not alignment_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No alignment data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Filter by sample IDs if provided
        if sample_ids:
            alignment_data = {
                sample_id: seq_record
                for sample_id, seq_record in alignment_data.items()
                if sample_id in sample_ids
            }

        if not alignment_data:
            fig = go.Figure()
            fig.add_annotation(
                text="No sequences found for selected samples",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

        # Calculate statistics for each sequence
        stats_data = []
        for sample_id, seq_record in alignment_data.items():
            seq_str = str(seq_record.seq)
            stats_data.append({
                'sample_id': sample_id,
                'length': len(seq_str),
                'gaps': seq_str.count('-'),
                'n_count': seq_str.upper().count('N'),
                'gc_content': (seq_str.upper().count('G') + seq_str.upper().count('C')) /
                             (len(seq_str) - seq_str.count('-') - seq_str.upper().count('N')) * 100
                             if (len(seq_str) - seq_str.count('-') - seq_str.upper().count('N')) > 0 else 0
            })

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sequence Length Distribution', 'Gap Content',
                          'N Content', 'GC Content Distribution'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Length distribution
        lengths = [s['length'] for s in stats_data]
        fig.add_trace(
            go.Histogram(x=lengths, name='Length', nbinsx=20),
            row=1, col=1
        )

        # Gap content
        sample_ids_list = [s['sample_id'] for s in stats_data]
        gaps = [s['gaps'] for s in stats_data]
        fig.add_trace(
            go.Scatter(x=sample_ids_list, y=gaps,
                      mode='markers', name='Gaps'),
            row=1, col=2
        )

        # N content
        n_counts = [s['n_count'] for s in stats_data]
        fig.add_trace(
            go.Scatter(x=sample_ids_list, y=n_counts,
                      mode='markers', name='N Count'),
            row=2, col=1
        )

        # GC content distribution
        gc_contents = [s['gc_content'] for s in stats_data]
        fig.add_trace(
            go.Histogram(x=gc_contents, name='GC Content', nbinsx=20),
            row=2, col=2
        )

        fig.update_layout(
            height=800,
            title_text=f"Alignment Statistics - {selected_key[0]} {selected_key[1]} {selected_key[2]}",
            showlegend=False
        )

        # Update x-axis for sample plots to be readable
        fig.update_xaxes(tickangle=45, row=1, col=2)
        fig.update_xaxes(tickangle=45, row=2, col=1)

        return fig

    def get_missing_samples_warning(
        self,
        selected_key: Tuple[str, str, str],
        requested_samples: List[str]
    ) -> Optional[str]:
        """
        Check for missing samples and return warning message.

        Args:
            selected_key: Tuple of (method, species, segment)
            requested_samples: List of requested sample IDs

        Returns:
            Warning message if samples are missing, None otherwise
        """
        alignment_data = self.data_manager.alignment_data.get(selected_key, {})
        available_samples = set(alignment_data.keys())
        requested_set = set(requested_samples)

        missing_samples = requested_set - available_samples

        if missing_samples:
            missing_list = ', '.join(sorted(missing_samples))
            method, species, segment = selected_key
            return (f"⚠️ Warning: The following samples are not available for "
                   f"{method} {species} {segment}: {missing_list}")

        return None

    def create_all_visualizations(
        self,
        selected_key: Tuple[str, str, str],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create all consensus visualizations for a specific alignment.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of visualizations and components
        """
        results = {}

        # Check for missing samples
        if sample_ids:
            warning = self.get_missing_samples_warning(selected_key, sample_ids)
            if warning:
                results['missing_samples_warning'] = warning

        # Alignment visualization (dash_bio component)
        alignment_data = self.create_alignment_visualization(selected_key, sample_ids)
        if alignment_data:
            results['alignment_viewer'] = alignment_data

        # Identity clustergram (dash_bio component)
        clustergram_data = self.create_identity_clustergram(selected_key, sample_ids)
        if clustergram_data:
            results['identity_clustergram'] = clustergram_data

        # Statistics plot
        stats_fig = self.create_alignment_statistics_plot(selected_key, sample_ids)
        if stats_fig.data:
            results['alignment_statistics'] = stats_fig

        return results
