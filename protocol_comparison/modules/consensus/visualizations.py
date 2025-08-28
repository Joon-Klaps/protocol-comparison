"""
Consensus visualizations.

This module creates visualizations for consensus sequence analysis and alignment visualization
using dash_bio components for interactive multiple sequence alignment display and clustering.
"""

import plotly.graph_objects as go
import logging
import streamlit as st
import numpy as np

from typing import Dict, List, Optional, Any, Tuple, TYPE_CHECKING
if TYPE_CHECKING:
    from AlignmentViewer import AlignmentViewer  # type: ignore
try:
    from AlignmentViewer import AlignmentViewer  # type: ignore
except ImportError:
    AlignmentViewer = None  # type: ignore


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
            return None

        # Limit number of sequences for performance
        if len(alignment_data) > max_sequences:
            # Take first N sequences (could be randomized or prioritized)
            sample_items = list(alignment_data.items())[:max_sequences]
            alignment_data = dict(sample_items)

        # Format for dash_bio using data manager helper
        try:
            if AlignmentViewer is None:
                logger.warning("AlignmentViewer is not available; skipping alignment visualization")
                return None

            sequences = [seq for seq in alignment_data.values()]
            fig = AlignmentViewer.get_alignment_html(
                sequences,
                color_snps_only=True,
            )
            if fig:
                logger.info("Created alignment visualization for %d sequences", len(alignment_data))
            return fig

        except (ImportError, ModuleNotFoundError, AttributeError) as e:
            logger.error("Error creating alignment visualization: %s", e)
            return None

    def create_identity_clustergram(
        self,
        key: Tuple[str, str, str],
        sample_ids: List[str]
    ):
        """
        Create pairwise identity heatmap with buttons to switch between Global and Local PID.

        Args:
            key: Tuple of (method, species, segment)
            sample_ids: List of sample IDs to include

        Returns:
            Tuple of (Plotly figure, raw_data_dict) containing heatmaps with interactive switching or None
        """

        method_name, species, segment = key
        alignment_data = self.data_manager.filter_alignment_by_samples(
            method_name,
            species,
            segment,
            sample_ids
        )

        # Get alignment data for the selected key
        if not alignment_data:
            logger.warning("No alignment found for %s with samples %s", key, sample_ids)
            return None, None

        try:
            # Compute both global and local identity matrices
            global_identity_matrix = self.data_manager.compute_pairwise_identity_matrix(alignment_data)
            local_identity_matrix = self.data_manager.compute_pairwise_identity_matrix_local(alignment_data)

            # Validate matrices
            if not self._validate_identity_matrix(global_identity_matrix, "global", key):
                return None, None
            if not self._validate_identity_matrix(local_identity_matrix, "local", key):
                return None, None

            aligned_sample_ids = list(alignment_data.keys())

            # Reorder labels to match the desired sample order from the UI, if present
            aligned_sample_ids = self._apply_sample_order_to_labels(aligned_sample_ids)

            # Reorder matrices to match the label order
            local_identity_matrix = self._reindex_matrix(local_identity_matrix, aligned_sample_ids, list(alignment_data.keys()))
            global_identity_matrix = self._reindex_matrix(global_identity_matrix, aligned_sample_ids, list(alignment_data.keys()))

            # Create combined heatmap with both datasets
            combined_fig = self._create_dual_heatmap(
                local_identity_matrix, global_identity_matrix,
                aligned_sample_ids, method_name, species, segment
            )

            if combined_fig is None:
                logger.warning("Failed to create combined heatmap for %s", key)
                return None, None

            # Prepare raw data for export
            raw_data = {
                'global_pid_distance_matrix': {
                    'distance_matrix': global_identity_matrix,
                    'sample_labels': aligned_sample_ids
                },
                'local_pid_distance_matrix': {
                    'distance_matrix': local_identity_matrix,
                    'sample_labels': aligned_sample_ids
                }
            }

            return combined_fig, raw_data

        except (ValueError, AttributeError, KeyError) as e:
            logger.error("Error creating identity heatmap for %s: %s", key, e)
            return None, None

    def _apply_sample_order_to_labels(self, labels: List[str]) -> List[str]:
        """Return labels ordered by session state's sample_order when available."""
        order = st.session_state.get("sample_order", [])
        if not isinstance(order, list) or not order:
            return labels
        # first keep those in desired order that are present, then append any remaining
        desired_first = [s for s in order if s in labels]
        remaining = [s for s in labels if s not in desired_first]
        return desired_first + remaining

    def _reindex_matrix(self, matrix: np.ndarray, new_labels: List[str], old_labels: List[str]) -> np.ndarray:
        """Reindex a square matrix from old_labels order to new_labels order."""
        if matrix is None or not isinstance(matrix, np.ndarray):
            return matrix
        index_map = {label: i for i, label in enumerate(old_labels)}
        # ensure all new_labels exist in old_labels; if not, skip
        indices = [index_map[label] for label in new_labels if label in index_map]
        if not indices:
            return matrix
        return matrix[np.ix_(indices, indices)]

    def _validate_identity_matrix(self, matrix, matrix_type: str, key: Tuple[str, str, str]) -> bool:
        """
        Validate an identity matrix.

        Args:
            matrix: Identity matrix to validate
            matrix_type: Type of matrix for logging (e.g., "global", "local")
            key: Key tuple for logging

        Returns:
            True if matrix is valid, False otherwise
        """
        if matrix is None or matrix.shape[0] == 0:
            logger.warning("Could not generate %s identity matrix for %s", matrix_type, key)
            return False
        return True

    def _create_dual_heatmap(
        self,
        local_matrix,
        global_matrix,
        labels: List[str],
        method_name: str,
        species: str,
        segment: str
    ) -> Optional[go.Figure]:
        """
        Create a single figure with both local and global heatmaps and buttons to switch between them.

        Args:
            local_matrix: Local PID identity matrix
            global_matrix: Global PID identity matrix
            labels: Sample labels
            method_name: Analysis method name
            species: Species name
            segment: Segment name

        Returns:
            Plotly figure with interactive buttons
        """
        # Validate inputs
        if not self._validate_heatmap_inputs(local_matrix, global_matrix, labels):
            return None

        # Create base figure
        fig = go.Figure()

        # Base title components
        base_title = f"Pairwise Identity Heatmap - {method_name} {species} {segment}"

        # Add local PID heatmap (initially visible)
        local_heatmap = self._create_heatmap_trace(
            local_matrix, labels, visible=True
        )
        fig.add_trace(local_heatmap)

        # Add global PID heatmap (initially hidden)
        global_heatmap = self._create_heatmap_trace(
            global_matrix, labels, visible=False
        )
        fig.add_trace(global_heatmap)

        # Add interactive buttons with proper title updates
        buttons = self._create_toggle_buttons(base_title)

        # Configure layout
        fig.update_layout(
            title={
                'text': f"{base_title} (Local PID)",
                'x': 0.5,
                'xanchor': 'center'
            },
            width=800,
            height=800,
            showlegend=False,
            hovermode='closest',
            updatemenus=[{
                "type": "buttons",
                "direction": "right",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.06,
                "yanchor": "top",
                "buttons": buttons
            }],
            xaxis={
                'ticktext': labels,
                'tickvals': list(range(len(labels))),
                'tickangle': -45,
                'side': 'bottom'
            },
            yaxis={
                'ticktext': labels,
                'tickvals': list(range(len(labels))),
                'autorange': 'reversed'
            }
        )

        return fig

    def _validate_heatmap_inputs(self, local_matrix, global_matrix, labels: List[str]) -> bool:
        """Validate inputs for heatmap creation."""
        if local_matrix is None or global_matrix is None or len(labels) == 0:
            logger.error("Invalid inputs for heatmap creation")
            return False

        if (local_matrix.shape[0] != len(labels) or local_matrix.shape[1] != len(labels) or
            global_matrix.shape[0] != len(labels) or global_matrix.shape[1] != len(labels)):
            logger.error("Matrix dimensions don't match number of labels")
            return False

        return True

    def _create_heatmap_trace(self, matrix, labels: List[str], visible: bool = True) -> go.Heatmap:
        """
        Create a single heatmap trace.

        Args:
            matrix: Identity matrix
            labels: Sample labels
            visible: Whether trace should be initially visible

        Returns:
            Plotly Heatmap trace
        """
        return go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale='brBG',
            zmin=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate='%{y} vs %{x}<br>Identity: %{z:.3f}<extra></extra>',
            text=np.round(matrix, 3),
            texttemplate="%{text}",
            textfont={"size": 15},
            showscale=True,
            visible=visible
        )

    def _create_toggle_buttons(self, base_title: str) -> List[Dict]:
        """
        Create toggle buttons for switching between Local and Global PID.

        Args:
            base_title: Base title for the plot

        Returns:
            List of button configurations
        """
        return [
            {
                "label": "Local PID",
                "method": "update",
                "args": [
                    {"visible": [True, False]},  # Show local, hide global
                    {
                        "title.text": f"<b>{base_title} (Local PID)</b>",
                        "title.x": 0.5,
                        "title.xanchor": "center"
                    }
                ]
            },
            {
                "label": "Global PID",
                "method": "update",
                "args": [
                    {"visible": [False, True]},  # Hide local, show global
                    {
                        "title.text": f"<b>{base_title} (Global PID)</b>",
                        "title.x": 0.5,
                        "title.xanchor": "center"
                    }
                ]
            }
        ]

    def _create_simple_heatmap(self, identity_matrix, labels, title):
        """
        Create a simple heatmap without dendrograms.

        Args:
            identity_matrix: Identity matrix for heatmap (similarity values 0-1)
            labels: Sample labels
            title: Plot title

        Returns:
            Plotly figure with heatmap
        """
        # Validate inputs
        if identity_matrix is None or len(labels) == 0:
            logger.error("Invalid inputs to _create_simple_heatmap")
            return None

        if identity_matrix.shape[0] != len(labels) or identity_matrix.shape[1] != len(labels):
            logger.error("Identity matrix dimensions don't match number of labels")
            return None

        # Create figure
        fig = go.Figure()

        # Create heatmap with brBG colorscale and text annotations
        heatmap = go.Heatmap(
            z=identity_matrix,
            x=labels,
            y=labels,
            colorscale='brBG',
            zmin=0,
            zmax=1,
            hoverongaps=False,
            hovertemplate='%{y} vs %{x}<br>Identity: %{z:.3f}<extra></extra>',
            text=np.round(identity_matrix, 3),  # Add raw values as text
            texttemplate="%{text}",  # Display the text values
            textfont={"size": 15},  # Small font for readability
            showscale=True
        )

        fig.add_trace(heatmap)

        # Update layout
        fig.update_layout(
            title=title,
            width=800,
            height=800,
            showlegend=False,
            hovermode='closest',
            xaxis={
                'ticktext': labels,
                'tickvals': list(range(len(labels))),
                'tickangle': -45,
                'side': 'bottom'
            },
            yaxis={
                'ticktext': labels,
                'tickvals': list(range(len(labels))),
                'autorange': 'reversed'
            }
        )

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
        results = {"figures": [], "warnings": [], "raw_data": {}}

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
                'type': 'html'
            })

        # Identity clustergram
        clustergram_result = self.create_identity_clustergram(selected_key, sample_ids)
        if clustergram_result and clustergram_result[0]:  # Check if both figure and raw_data are returned
            clustergram_fig, raw_data = clustergram_result
            results['figures'].append({
                'title': f"Pairwise Identity Heatmap - {selected_key[0]} {selected_key[1]} {selected_key[2]}",
                'description': 'Interactive heatmap showing pairwise identity between samples.',
                'figure': clustergram_fig,
                'type': 'plotly'
            })
            # Add raw data to results
            results['raw_data'] = raw_data
        else:
            logger.warning("No clustergram generated for %s", selected_key)

        return results