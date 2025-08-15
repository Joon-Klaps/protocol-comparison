"""
Coverage analysis module.

This module handles coverage-related analysis:
- Coverage depth statistics
- Coverage visualizations
- Segment-specific coverage analysis
"""

from .summary_stats import CoverageSummaryStats, CoverageDataManager
from .visualizations import CoverageVisualizations

# Main unified analyzer
class CoverageAnalyzer:
    """
    Unified analyzer for coverage depth analysis and genome recovery.

    This class provides a single interface to analyze:
    - Coverage depth statistics
    - Coverage visualizations
    - Segment-specific coverage analysis
    - Genome recovery calculations

    Uses the hierarchical modular structure for better maintainability.
    """

    def __init__(self, data_path):
        """
        Initialize unified coverage analyzer.

        Args:
            data_path: Path to data directory
        """
        from pathlib import Path

        self.data_path = Path(data_path)
        self.depth_threshold = 10  # Default minimum depth for recovery

        # Initialize sub-modules
        self.coverage_stats = CoverageSummaryStats(self.data_path)
        self.coverage_viz = CoverageVisualizations(self.data_path)

        # Expose data for backward compatibility
        self.data = self.coverage_stats.data

    def set_depth_threshold(self, threshold: int) -> None:
        """Set the minimum depth threshold for genome recovery calculations."""
        self.depth_threshold = threshold
        self.coverage_stats.set_depth_threshold(threshold)

    def get_available_samples(self):
        """Get available sample IDs from coverage data."""
        return self.coverage_stats.data_manager.get_available_samples()

    def get_sample_coverage(self, sample_id: str, segment=None):
        """Get coverage data for a specific sample."""
        return self.coverage_stats.get_sample_coverage(sample_id, segment)

    def calculate_coverage_stats(self, sample_ids=None):
        """Calculate coverage statistics for samples."""
        return self.coverage_stats.calculate_coverage_stats(sample_ids)

    def generate_summary_stats(self, sample_ids=None):
        """
        Generate comprehensive summary statistics for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing coverage statistics
        """
        stats = {}

        # Get overall statistics
        overall_stats = self.coverage_stats.calculate_overall_stats(sample_ids)
        if overall_stats:
            stats['overall'] = overall_stats

        # Get segment-specific statistics
        segment_stats = self.coverage_stats.calculate_segment_specific_stats(sample_ids)
        if segment_stats:
            stats['by_segment'] = segment_stats

        return stats

    def create_visualizations(self, sample_ids=None):
        """
        Create all visualizations for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        return self.coverage_viz.create_all_visualizations(sample_ids)

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

        # Export coverage statistics
        self.coverage_stats.export_results(output_path, sample_ids)

        logging.getLogger(__name__).info("Coverage analysis results exported to %s", output_path)


__all__ = ['CoverageSummaryStats', 'CoverageVisualizations', 'CoverageAnalyzer', 'CoverageDataManager']
