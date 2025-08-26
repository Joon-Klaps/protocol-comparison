"""
Consensus analysis module.

This module handles consensus sequence analysis with tuple-based data structure:
- Alignment statistics and comparisons
- dash_bio visualization components
- Method comparison (mapping vs denovo)
- Streamlit integration support
"""

from .data import ConsensusDataManager
from .visualizations import ConsensusVisualizations
from .tab import ConsensusTab

# Main unified analyzer
class ConsensusAnalyzer:
    """
    Unified analyzer for consensus sequence analysis and genome recovery.

    This class provides a single interface to analyze:
    - Genome recovery statistics
    - ANI calculations
    - Consensus sequence comparisons
    - Segment-specific analysis

    Uses the hierarchical modular structure for better maintainability.
    """

    def __init__(self, data_path):
        """
        Initialize unified consensus analyzer.

        Args:
            data_path: Path to consensus analysis data directory
        """
        self.data_path = data_path

        # Initialize modular components with new structure
        self.data_manager = ConsensusDataManager(self.data_path)
        self.consensus_tab = ConsensusTab(self.data_path)
        self.consensus_viz = ConsensusVisualizations(self.data_manager)

    def get_available_alignments(self):
        """Get available alignment keys."""
        return self.consensus_tab.get_available_alignments()

    def get_available_samples(self):
        """Get list of available samples."""
        return self.consensus_tab.get_available_samples()

    def get_summary_stats(self, sample_ids=None):
        """Get summary statistics for consensus analysis."""
        return self.consensus_tab.get_summary_stats(sample_ids)

    def get_visualizations(self, sample_ids=None):
        """Get visualizations for consensus analysis."""
        return self.consensus_tab.get_visualizations(sample_ids)

    def run_analysis(self, sample_ids=None):
        """
        Run complete consensus analysis workflow.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing all analysis results
        """
        results = {
            'summary_stats': {},
            'visualizations': {},
            'available_data': {}
        }

        # Get summary statistics
        results['summary_stats'] = self.get_summary_stats(sample_ids)

        # Get visualizations
        results['visualizations'] = self.get_visualizations(sample_ids)

        # Get available data info
        results['available_data'] = {
            'alignments': self.get_available_alignments(),
            'samples': self.get_available_samples()
        }

        return results


__all__ = ['ConsensusDataManager', 'ConsensusVisualizations', 'ConsensusTab', 'ConsensusAnalyzer']
