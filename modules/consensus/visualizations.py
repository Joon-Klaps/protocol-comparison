"""
Consensus visualizations.

This module creates visualizations for consensus sequence analysis and alignment visualization
using dash_bio components for interactive multiple sequence alignment display and clustering.
"""

import numpy as np
import plotly.graph_objects as go
import logging
import dash_bio

from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from AlignmentViewer import AlignmentViewer
from plotly.subplots import make_subplots


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

    def __init__(self, data_manager: ConsensusDataManager):
        """
        Initialize consensus visualizations.

        Args:
            data_manager: ConsensusDataManager instance
        """
        self.data_manager = data_manager

    def create_alignment_visualization(
        self,
        key: Tuple[str, str, str],
        sample_ids: List[str],
        max_sequences: int = 50
    ) -> Any:
        """
        Create interactive multiple sequence alignment visualization using dash_bio.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs to visualize
            max_sequences: Maximum number of sequences to display (for performance)

        Returns:
            Plotly figure containing dash_bio AlignmentChart component or None
        """
        method, species, segment = key
        alignment_data = self.data_manager.filter_alignment_by_samples(method,species,segment, sample_ids)

        # Get alignment data for the selected key
        if not alignment_data:
            logger.warning("No alignment found for %s with samples %s", key, sample_ids)
            return None

        # Limit number of sequences for performance
        if len(alignment_data) > max_sequences:
            # Take first N sequences (could be randomized or prioritized)
            sample_items = list(alignment_data.items())[:max_sequences]
            alignment_data = dict(sample_items)

        # Format for dash_bio using data manager helper
        try:
            sequences = [seq for seq in alignment_data.values()]
            # Create AlignmentChart component
            fig = AlignmentViewer.get_alignment_plotly(
                sequences,
                show_consensus=True,
            )


            if fig:
                logger.info("Created alignment visualization for %d sequences", len(alignment_data))
            return fig

        except Exception as e:
            logger.error("Error creating alignment visualization: %s", e)
            return None

    def create_identity_clustergram(
        self,
        key: Tuple[str, str, str],
        sample_ids: List[str],
    ):
        """
        Create pairwise identity clustergram using dash_bio.

        Args:
            alignment_data: Dictionary of alignment data for visualization

        Returns:
            Dictionary for dash_bio clustergram component or None
        """

        method, species, segment = key
        alignment_data = self.data_manager.filter_alignment_by_samples(
            method,
            species,
            segment,
            sample_ids
        )

        # Get alignment data for the selected key
        if not alignment_data:
            logger.warning("No alignment found for %s with samples %s", key, sample_ids)
            return None

        try:
            identity_matrix = self.data_manager.compute_pairwise_identity_matrix(alignment_data)
            if identity_matrix is None or identity_matrix.size == 0:
                logger.warning("Could not generate identity matrix for %s", key)
                return None

            aligned_sample_ids = list(alignment_data.keys())

            # Prepare data for dash_bio.Clustergram
            return dash_bio.Clustergram(
                row_labels=aligned_sample_ids,
                column_labels=aligned_sample_ids,
                data=identity_matrix,
                center_values=False,
            )
        except (ValueError, AttributeError, KeyError) as e:
            logger.error("Error creating identity clustergram for %s: %s", key, e)
            return None

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
        sample_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Create all consensus visualizations for a specific alignment.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of visualizations and components
        """
        results = {"figures": [], "warnings": []}

        # Check for missing samples
        if sample_ids:
            warning = self.get_missing_samples_warning(selected_key, sample_ids)
            if warning:
                results['warnings'].append(warning)

        if not sample_ids:
            return results

        # Alignment visualization
        alignment_fig = self.create_alignment_visualization(selected_key, sample_ids)
        if alignment_fig:
            results['figures'].append({
                'title': f"Multiple Sequence Alignment - {selected_key[0]} {selected_key[1]} {selected_key[2]}",
                'description': 'Interactive alignment of consensus sequences.',
                'figure': alignment_fig,
                'type': 'plotly'
            })

        # Identity clustergram
        clustergram_fig = self.create_identity_clustergram(selected_key, sample_ids)
        if clustergram_fig:
            results['figures'].append({
                'title': f"Pairwise Identity Clustergram - {selected_key[0]} {selected_key[1]} {selected_key[2]}",
                'description': 'Hierarchical clustering of sequences based on pairwise identity.',
                'figure': clustergram_fig,
                'type': 'plotly'
            })

        return results