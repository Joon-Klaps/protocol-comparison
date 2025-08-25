"""
Consensus analysis tab component.

This module collects and organizes consensus analysis components for display.
Contains no Streamlit code - returns pure data, visualizations, and custom HTML.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

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
        """Get list of available (method, species, segment) combinations."""
        if 'consensus' in self.components:
            try:
                data_manager = self.components['consensus']['data_manager']
                return list(data_manager.alignment_data.keys())
            except Exception as e:
                logger.warning("Error getting alignments: %s", e)
        return []

    def get_available_samples(self) -> List[str]:
        """Get list of available samples for a specific alignment."""
        if 'consensus' in self.components:
            try:
                data_manager = self.components['consensus']['data_manager']
                return data_manager.get_available_samples()
            except Exception as e:
                logger.warning("Error getting samples: %s", e)
        return []

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

        keys = data_manager.alignment_data.keys()

        # Calculate alignment summaries using data manager methods

        for selected_key in keys:
            alignment_data = data_manager.alignment_data.get(selected_key, {})

            if not alignment_data:
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
        return summary

    def get_visualizations(
        self,
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get visualizations for consensus analysis.

        Args:
            selected_keys: List of (method, species, segment) tuples
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

        data_manager = self.components['consensus']['data_manager']
        viz_component = self.components['consensus']['viz']

        keys = data_manager.alignment_data.keys()

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

            # Iterate through each selected alignment combination
            for selected_key in data_manager.alignment_data.keys():
                try:
                    method, species, segment = selected_key

                    # Get alignment data for this combination
                    alignment_data = data_manager.alignment_data.get(selected_key, {})

                    if not alignment_data:
                        continue

                    # Filter by sample_ids if provided
                    if sample_ids:
                        filtered_data = {
                            sample_id: seq_record
                            for sample_id, seq_record in alignment_data.items()
                            if sample_id in sample_ids
                        }
                    else:
                        filtered_data = alignment_data

                    if filtered_data:
                        # Convert to FASTA format for raw data display
                        fasta_lines = []
                        for sample_id, seq_record in filtered_data.items():
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
