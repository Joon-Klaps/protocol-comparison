"""
Consensus analysis tab component.

This module collects and organizes consensus analysis components for display.
Contains no Streamlit code - returns pure data, visualizations, and custom HTML.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import pandas as pd

from .data import ConsensusDataManager
from .visualizations import ConsensusVisualizations

logger = logging.getLogger(__name__)


class ConsensusTab:
    """
    Consensus analysis tab component that collects all consensus-related analysis.
    Framework-agnostic - returns data structures, figures, and custom HTML.
    """

    def __init__(self, data_path: Path):
        """
        Initialize consensus tab.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.components = {}
        self._cached_alignments = None
        self._cached_samples = None
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all consensus analysis components."""
        try:
            # Consensus components
            data_manager = ConsensusDataManager(self.data_path)
            self.components['consensus'] = {
                'data_manager': data_manager,
                'viz': ConsensusVisualizations(data_manager)
            }

        except Exception as e:
            logger.error("Error initializing consensus components: %s", e)
            self.components = {}

    def get_available_alignments(self) -> List[Tuple[str, str, str]]:
        """Get list of available (method, species, segment) combinations with caching."""
        if self._cached_alignments is None:
            if 'consensus' in self.components:
                try:
                    data_manager = self.components['consensus']['data_manager']
                    self._cached_alignments = list(data_manager.alignment_data.keys())
                except Exception as e:
                    logger.warning("Error getting alignments: %s", e)
                    self._cached_alignments = []
            else:
                self._cached_alignments = []
        return self._cached_alignments

    def get_available_samples(self) -> List[str]:
        """Get list of available samples with caching."""
        if self._cached_samples is None:
            if 'consensus' in self.components:
                try:
                    data_manager = self.components['consensus']['data_manager']
                    self._cached_samples = data_manager.get_available_samples()
                except Exception as e:
                    logger.warning("Error getting samples: %s", e)
                    self._cached_samples = []
            else:
                self._cached_samples = []
        return self._cached_samples

    def get_summary_stats(
        self,
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis',
            'description': 'Analysis of alignment statistics and method comparisons',
            'sections': []
        }

        if 'consensus' not in self.components:
            return summary

        data_manager = self.components['consensus']['data_manager']

        # Use cached alignments instead of calling keys() multiple times
        keys = self.get_available_alignments()

        # Calculate alignment summaries using data manager methods
        for selected_key in keys:
            try:
                # Check if alignment data exists first (avoid unnecessary loading)
                if selected_key not in data_manager.alignment_data:
                    continue

                # Add key information
                stats = data_manager.get_alignment_summary_stats(selected_key, sample_ids)
                if not stats:
                    continue

                summary['sections'].append({
                    'title': f"{selected_key[0]} - {selected_key[1]} {selected_key[2]}",
                    'type': 'metrics',
                    'data': stats
                })
            except Exception as e:
                logger.warning("Error getting stats for %s: %s", selected_key, e)

        return summary

    def get_visualizations(
        self,
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get visualizations for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary containing plotly figures and dash_bio components
        """
        figures = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis Visualizations',
            'figures': [],
            'warnings': []
        }

        if 'consensus' not in self.components:
            return figures

        viz_component = self.components['consensus']['viz']

        # Use cached alignments
        keys = self.get_available_alignments()

        for selected_key in keys:
            try:
                # Iterate through each selected alignment combination
                all_visualizations = viz_component.create_all_visualizations(selected_key, sample_ids)

                if all_visualizations:
                    figures['figures'].extend(all_visualizations.get('figures', []))
                    if 'warnings' in all_visualizations:
                        figures['warnings'].extend(all_visualizations['warnings'])
            except Exception as e:
                logger.warning("Failed to create visualizations for key %s: %s", selected_key, e)

        return figures

    def get_raw_data(self,sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get raw data tables for consensus analysis.

        Args:
            selected_keys: List of (method, species, segment) tuples
            sample_ids: Optional list of sample IDs

        Returns:
            Dictionary containing raw data tables
        """
        data = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis Raw Data',
            'tables': []
        }

        if 'consensus' not in self.components:
            return data

        try:
            data_manager = self.components['consensus']['data_manager']
            viz_component = self.components['consensus']['viz']

            # Iterate through each selected alignment combination
            for selected_key in self.get_available_alignments():
                try:
                    method, species, segment = selected_key

                    # Build a realigned filtered alignment for this combination (ensures consistent MSA)
                    if sample_ids and len(sample_ids) > 0:
                        filtered_data = data_manager.filter_alignment_by_samples(
                            method, species, segment, sample_ids, remove_gap_columns=True, realign=True
                        ) or {}
                    else:
                        # Use all available samples for this key
                        alignment_data = data_manager.alignment_data.get(selected_key, {})
                        if not alignment_data:
                            continue
                        all_ids = list(alignment_data.keys())
                        filtered_data = data_manager.filter_alignment_by_samples(
                            method, species, segment, all_ids, remove_gap_columns=True, realign=True
                        ) or {}

                    if filtered_data:
                        # Convert to FASTA format for raw data display
                        fasta_lines = []
                        # Keep order: if sample_ids provided, respect that order; else keep dict order
                        ordered_ids = sample_ids if sample_ids else list(filtered_data.keys())
                        for sample_id in ordered_ids:
                            if sample_id in filtered_data:
                                seq_record = filtered_data[sample_id]
                                fasta_lines.append(f">{sample_id}")
                                fasta_lines.append(str(seq_record.seq))

                        fasta_content = "\n".join(fasta_lines)

                        data['tables'].append({
                            'title': f'Alignment Data - {method} {species} {segment}',
                            'data': fasta_content,
                            'type': 'fasta',
                            'sample_count': len(filtered_data),
                            'alignment_length': len(str(list(filtered_data.values())[0].seq)) if filtered_data else 0
                        })

                        # Get distance matrices from visualizations
                        try:
                            viz_result = viz_component.create_all_visualizations(selected_key, list(filtered_data.keys()))
                            if viz_result and 'raw_data' in viz_result and viz_result['raw_data']:
                                raw_data = viz_result['raw_data']

                                # Add Global PID distance matrix
                                if 'global_pid_distance_matrix' in raw_data:
                                    global_data = raw_data['global_pid_distance_matrix']
                                    if 'distance_matrix' in global_data and 'sample_labels' in global_data:
                                        matrix = global_data['distance_matrix']
                                        labels = global_data['sample_labels']

                                        try:
                                            # Create pandas DataFrame for distance matrix
                                            df = pd.DataFrame(matrix, index=labels, columns=labels)
                                            df.index.name = 'Sample'

                                            data['tables'].append({
                                                'title': f'Global PID Distance Matrix - {method} {species} {segment}',
                                                'data': df,
                                                'type': 'dataframe',
                                                'description': 'Distance matrix (1 - identity) for Global PID method'
                                            })
                                        except Exception as csv_e:
                                            logger.warning("Failed to create DataFrame for global distance matrix: %s", csv_e)

                                # Add Local PID distance matrix
                                if 'local_pid_distance_matrix' in raw_data:
                                    local_data = raw_data['local_pid_distance_matrix']
                                    if 'distance_matrix' in local_data and 'sample_labels' in local_data:
                                        matrix = local_data['distance_matrix']
                                        labels = local_data['sample_labels']

                                        try:
                                            # Create pandas DataFrame for distance matrix
                                            df = pd.DataFrame(matrix, index=labels, columns=labels)
                                            df.index.name = 'Sample'

                                            data['tables'].append({
                                                'title': f'Local PID Distance Matrix - {method} {species} {segment}',
                                                'data': df,
                                                'type': 'dataframe',
                                                'description': 'Distance matrix (1 - identity) for Local PID method (ignoring gaps/Ns)'
                                            })
                                        except Exception as csv_e:
                                            logger.warning("Failed to create DataFrame for local distance matrix: %s", csv_e)

                        except Exception as viz_e:
                            logger.warning("Failed to get distance matrices for key %s: %s", selected_key, viz_e)

                except Exception as e:
                    logger.warning("Failed to get raw data for key %s: %s", selected_key, e)

        except Exception as e:
            logger.error("Error getting consensus raw data: %s", e)

        return data


def get_tab_info() -> Dict[str, Any]:
    """
    Get information about this tab component.

    Returns:
        Dictionary with tab metadata
    """
    return {
        'name': 'consensus',
        'title': 'Consensus Analysis',
        'icon': '🧬',
        'description': 'Analysis of genome recovery statistics and nucleotide identity comparisons',
        'order': 20,
        'requires_data': True,
        'data_subdirs': ['consensus']
    }


def create_tab(data_path: Path) -> ConsensusTab:
    """
    Factory function to create a consensus tab.

    Args:
        data_path: Path to data directory

    Returns:
        ConsensusTab instance
    """
    return ConsensusTab(data_path)
