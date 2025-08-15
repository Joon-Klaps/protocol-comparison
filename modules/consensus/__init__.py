"""
Consensus analysis module.

This module handles consensus sequence analysis:
- Genome recovery statistics
- ANI calculations
- Consensus sequence comparisons
"""

from .summary_stats import ConsensusSummaryStats, ConsensusDataManager
from .visualizations import ConsensusVisualizations

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
            data_path: Path to data directory
        """
        from pathlib import Path

        self.data_path = Path(data_path)

        # Initialize sub-modules
        self.consensus_stats = ConsensusSummaryStats(self.data_path)
        self.consensus_viz = ConsensusVisualizations(self.data_path)

        # Expose data for backward compatibility
        self.data = self.consensus_stats.data

    def get_available_samples(self):
        """Get available sample IDs from consensus data."""
        return self.consensus_stats.data_manager.get_available_samples()

    def calculate_genome_recovery(self, sample_ids=None):
        """Calculate genome recovery statistics for samples."""
        return self.consensus_stats.calculate_genome_recovery_stats(sample_ids)

    def calculate_ani_matrix(self, sample_ids=None):
        """Calculate Average Nucleotide Identity (ANI) matrix between samples."""
        return self.consensus_stats.calculate_ani_matrix(sample_ids)

    def generate_summary_stats(self, sample_ids=None):
        """
        Generate comprehensive summary statistics for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing consensus statistics
        """
        return self.consensus_stats.calculate_overall_summary(sample_ids)

    def create_visualizations(self, sample_ids=None):
        """
        Create all visualizations for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        return self.consensus_viz.create_all_visualizations(sample_ids)

    def export_results(self, output_path, sample_ids=None):
        """
        Export analysis results to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        from pathlib import Path
        import logging

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export consensus statistics
        self.consensus_stats.export_results(output_path, sample_ids)

        logging.getLogger(__name__).info("Consensus analysis results exported to %s", output_path)


__all__ = ['ConsensusSummaryStats', 'ConsensusVisualizations', 'ConsensusAnalyzer', 'ConsensusDataManager']
