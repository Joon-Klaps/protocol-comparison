"""
Read statistics analysis module.

This module provides unified access to read processing and mapping statistics
through a hierarchical modular structure. It also provides Streamlit page
components for web-based dashboards.
"""

from .reads import ReadProcessingSummaryStats, ReadProcessingVisualizations
from .mapping import MappingSummaryStats, MappingVisualizations

# Try to import streamlit components (optional)
try:
    from .streamlit_pages import get_read_stats_pages, ReadProcessingPage, MappingStatisticsPage
    _STREAMLIT_AVAILABLE = True
except ImportError:
    _STREAMLIT_AVAILABLE = False

__all__ = [
    'ReadProcessingSummaryStats',
    'ReadProcessingVisualizations',
    'MappingSummaryStats',
    'MappingVisualizations',
    'ReadStatsAnalyzer'
]

# Add streamlit components to __all__ if available
if _STREAMLIT_AVAILABLE:
    __all__.extend([
        'get_read_stats_pages',
        'ReadProcessingPage',
        'MappingStatisticsPage'
    ])

# Unified interface for backward compatibility
class ReadStatsAnalyzer:
    """
    Unified analyzer that combines read processing and mapping statistics.

    This class provides a single interface to analyze both:
    - Read processing pipeline statistics (trimming, host removal)
    - Mapping statistics (species, segments, efficiency)

    Uses the hierarchical modular structure for better maintainability.
    """

    def __init__(self, data_path):
        """
        Initialize unified read statistics analyzer.

        Args:
            data_path: Path to data directory
        """
        from pathlib import Path

        self.data_path = Path(data_path)

        # Initialize sub-modules
        self.reads_stats = ReadProcessingSummaryStats(self.data_path)
        self.reads_viz = ReadProcessingVisualizations(self.data_path)
        self.mapping_stats = MappingSummaryStats(self.data_path)
        self.mapping_viz = MappingVisualizations(self.data_path)

        # Expose data for backward compatibility
        self.data = {}
        self.data.update(self.reads_stats.data)
        self.data.update(self.mapping_stats.data)

    def get_available_samples(self):
        """Get available sample IDs from both read processing and mapping data."""
        read_samples = set(self.reads_stats.data_manager.get_available_samples())
        mapping_samples = set(self.mapping_stats.data_manager.get_available_samples())

        # Return intersection of both datasets
        common_samples = read_samples.intersection(mapping_samples)
        return sorted(list(common_samples))

    def create_mapping_overview_per_segment_species(self, df):
        """Create mapping statistics overview grouped by segment and species."""
        return self.mapping_stats.calculate_species_segment_stats()

    def generate_summary_stats(self, sample_ids=None):
        """
        Generate comprehensive summary statistics for both processing and mapping.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing both processing and mapping statistics
        """
        stats = {}

        # Get read processing statistics
        processing_stats = self.reads_stats.calculate_efficiency_stats(sample_ids)
        if processing_stats:
            stats['read_processing'] = processing_stats

        # Get mapping statistics
        mapping_stats = self.mapping_stats.calculate_species_segment_stats(sample_ids)
        if mapping_stats:
            stats['mapping'] = mapping_stats

        return stats

    def create_visualization_reads(self, sample_ids=None):
        """Create read processing visualization."""
        return self.reads_viz.create_processing_timeline(sample_ids)

    def create_visualization_mapping(self, sample_ids=None):
        """Create mapping statistics visualizations."""
        return self.mapping_viz.create_all_visualizations(sample_ids)

    def create_visualizations(self, sample_ids=None):
        """
        Create all visualizations for both read processing and mapping analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary of plotly figures
        """
        figures = {}

        # Add read processing visualization
        fig_reads = self.create_visualization_reads(sample_ids)
        if fig_reads.data:  # Only add if figure has data
            figures['read_processing'] = fig_reads

        # Add mapping visualizations
        mapping_figures = self.create_visualization_mapping(sample_ids)
        figures.update(mapping_figures)

        return figures

    def export_results(self, output_path, sample_ids=None):
        """
        Export analysis results for both processing and mapping to files.

        Args:
            output_path: Directory to save results
            sample_ids: Optional list of sample IDs to export
        """
        from pathlib import Path
        import logging

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Export read processing results
        processing_path = output_path / "read_processing"
        self.reads_stats.export_results(processing_path, sample_ids)

        # Export mapping results
        mapping_path = output_path / "mapping"
        self.mapping_stats.export_results(mapping_path, sample_ids)

        # Export unified summary
        unified_stats = self.generate_summary_stats(sample_ids)

        # Create a simplified unified summary
        unified_summary = {
            'analysis_type': 'unified_read_statistics',
            'sample_count': len(sample_ids) if sample_ids else len(self.get_available_samples()),
            'has_processing_data': 'read_processing' in unified_stats,
            'has_mapping_data': 'mapping' in unified_stats
        }

        # Add high-level metrics if available
        if 'read_processing' in unified_stats:
            unified_summary['overall_read_retention_pct'] = unified_stats['read_processing']['overall_efficiency']['mean_retention_pct']
            unified_summary['total_samples_processed'] = unified_stats['read_processing']['sample_count']
            unified_summary['total_raw_reads'] = unified_stats['read_processing']['total_raw_reads']
            unified_summary['total_final_reads'] = unified_stats['read_processing']['total_final_reads']

        if 'mapping' in unified_stats:
            species_count = len(unified_stats['mapping'])
            total_segments = sum(len(segments) for segments in unified_stats['mapping'].values())
            unified_summary['species_analyzed'] = species_count
            unified_summary['total_species_segment_combinations'] = total_segments

        import json
        with open(output_path / 'unified_summary.json', 'w', encoding='utf-8') as f:
            json.dump(unified_summary, f, indent=2, default=str)

        logging.getLogger(__name__).info("Unified read statistics analysis results exported to %s", output_path)
