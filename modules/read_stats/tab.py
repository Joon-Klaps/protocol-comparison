"""
Read statistics tab component.

This module collects and organizes read statistics components for display.
Contains no Streamlit code - returns pure data and visualizations.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .reads.summary_stats import ReadProcessingDataManager, ReadProcessingSummaryStats
from .reads.visualizations import ReadProcessingVisualizations
from .mapping.summary_stats import MappingDataManager, MappingSummaryStats
from .mapping.visualizations import MappingVisualizations

logger = logging.getLogger(__name__)


class ReadStatsTab:
    """
    Read statistics tab component that collects all read-related analysis.
    Framework-agnostic - returns data structures and figures.
    """

    def __init__(self, data_path: Path):
        """
        Initialize read stats tab.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.components = {}
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all read stats components."""
        try:
            # Read processing components
            read_data_manager = ReadProcessingDataManager(self.data_path)
            self.components['read_processing'] = {
                'stats': ReadProcessingSummaryStats(self.data_path),
                'viz': ReadProcessingVisualizations(self.data_path),
                'data_manager': read_data_manager
            }

            # Mapping components
            mapping_data_manager = MappingDataManager(self.data_path)
            self.components['mapping'] = {
                'stats': MappingSummaryStats(self.data_path),
                'viz': MappingVisualizations(self.data_path),
                'data_manager': mapping_data_manager
            }

        except Exception as e:
            logger.error("Error initializing read stats components: %s", e)
            self.components = {}

    def get_available_samples(self) -> List[str]:
        """Get list of available samples across all components."""
        all_samples = set()

        for component_type in self.components.values():
            if 'data_manager' in component_type:
                try:
                    samples = component_type['data_manager'].get_available_samples()
                    all_samples.update(samples)
                except Exception as e:
                    logger.warning("Error getting samples: %s", e)

        return sorted(list(all_samples))

    def get_summary_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get summary statistics for read processing and mapping.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'component_type': 'read_stats',
            'title': 'Read Statistics Analysis',
            'description': 'Analysis of read processing pipeline and mapping statistics',
            'sections': []
        }

        # Read processing summary
        if 'read_processing' in self.components:
            try:
                stats = self.components['read_processing']['stats'].calculate_efficiency_stats(sample_ids)
                if stats:
                    summary['sections'].append({
                        'title': 'Read Processing Efficiency',
                        'type': 'metrics',
                        'data': {
                            'Total Samples': stats.get('sample_count', 0),
                            'Average Retention': f"{stats.get('overall_efficiency', {}).get('mean_retention_pct', 0):.1f}%",
                            'Mean Raw Reads': f"{stats.get('mean_raw_reads', 0):,}",
                            'Mean Final Reads': f"{stats.get('mean_final_reads', 0):,}"
                        }
                    })

            except Exception as e:
                logger.error("Error generating read processing stats: %s", e)

        # Mapping summary
        if 'mapping' in self.components:
            try:
                mapping_stats = self.components['mapping']['stats'].calculate_species_segment_stats(sample_ids)
                if mapping_stats:
                    for species, segments in mapping_stats.items():
                        summary['sections'].append({
                            'title': f'Mapping Statistics - {species}',
                            'type': 'species_breakdown',
                            'data': segments
                        })

            except Exception as e:
                logger.error("Error generating mapping stats: %s", e)

        return summary

    def get_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get visualizations for read processing and mapping.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary containing plotly figures
        """
        figures = {
            'component_type': 'read_stats',
            'title': 'Read Statistics Visualizations',
            'figures': []
        }

        # Read processing visualization
        if 'read_processing' in self.components:
            try:
                fig = self.components['read_processing']['viz'].create_processing_timeline(sample_ids)
                if fig and fig.data:
                    figures['figures'].append({
                        'title': 'Read Processing Timeline',
                        'description': 'Timeline showing read count changes through processing pipeline',
                        'figure': fig,
                        'type': 'plotly'
                    })
            except Exception as e:
                logger.error("Error generating read processing visualization: %s", e)

        # Mapping visualizations
        if 'mapping' in self.components:
            try:
                mapping_figs = self.components['mapping']['viz'].create_all_visualizations(sample_ids)
                if mapping_figs:
                    for title, fig in mapping_figs.items():
                        if fig and fig.data:
                            figures['figures'].append({
                                'title': title,
                                'description': f'Mapping analysis: {title.lower()}',
                                'figure': fig,
                                'type': 'plotly'
                            })
            except Exception as e:
                logger.error("Error generating mapping visualizations: %s", e)

        return figures

    def get_raw_data(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get raw data tables for read processing and mapping.

        Args:
            sample_ids: Optional list of sample IDs

        Returns:
            Dictionary containing raw data tables
        """
        data = {
            'component_type': 'read_stats',
            'title': 'Read Statistics Raw Data',
            'tables': []
        }

        # Read processing raw data
        if 'read_processing' in self.components:
            try:
                component = self.components['read_processing']['stats']
                if hasattr(component, 'data') and component.data:
                    for data_name, df in component.data.items():
                        if not df.empty:
                            # Filter by sample IDs if provided
                            filtered_df = df
                            if sample_ids and 'sample' in df.columns:
                                filtered_df = df[df['sample'].isin(sample_ids)]
                            elif sample_ids and hasattr(df, 'index'):
                                # If sample IDs are in index
                                available_samples = [s for s in sample_ids if s in df.index]
                                if available_samples:
                                    filtered_df = df.loc[available_samples]

                            if not filtered_df.empty:
                                data['tables'].append({
                                    'title': f'Read Processing: {data_name.replace("_", " ").title()}',
                                    'data': filtered_df,
                                    'type': 'dataframe'
                                })
            except Exception as e:
                logger.error("Error getting read processing raw data: %s", e)

        # Mapping raw data
        if 'mapping' in self.components:
            try:
                component = self.components['mapping']['stats']
                if hasattr(component, 'data') and component.data:
                    for data_name, df in component.data.items():
                        if not df.empty:
                            # Filter by sample IDs if provided
                            filtered_df = df
                            if sample_ids and 'sample' in df.columns:
                                filtered_df = df[df['sample'].isin(sample_ids)]

                            if not filtered_df.empty:
                                data['tables'].append({
                                    'title': f'Mapping: {data_name.replace("_", " ").title()}',
                                    'data': filtered_df,
                                    'type': 'dataframe'
                                })
            except Exception as e:
                logger.error("Error getting mapping raw data: %s", e)

        return data


def get_tab_info() -> Dict[str, Any]:
    """
    Get information about this tab component.

    Returns:
        Dictionary with tab metadata
    """
    return {
        'name': 'read_stats',
        'title': 'Read Analysis',
        'icon': '📖',
        'description': 'Analysis of read processing pipeline and mapping statistics',
        'order': 10,
        'requires_data': True,
        'data_subdirs': ['read_stats', 'mapping']
    }


def create_tab(data_path: Path) -> ReadStatsTab:
    """
    Factory function to create a read stats tab.

    Args:
        data_path: Path to data directory

    Returns:
        ReadStatsTab instance
    """
    return ReadStatsTab(data_path)
