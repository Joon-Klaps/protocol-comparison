"""
Coverage analysis tab component.

This module collects and organizes coverage analysis components for display.
Contains no Streamlit code - returns pure data and visualizations.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .summary_stats import CoverageDataManager, CoverageSummaryStats
from .visualizations import CoverageVisualizations

logger = logging.getLogger(__name__)


class CoverageTab:
    """
    Coverage analysis tab component that collects all coverage-related analysis.
    Framework-agnostic - returns data structures and figures.
    """

    def __init__(self, data_path: Path):
        """
        Initialize coverage tab.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.components = {}
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all coverage analysis components."""
        try:
            # Coverage components
            data_manager = CoverageDataManager(self.data_path)
            self.components['coverage'] = {
                'stats': CoverageSummaryStats(self.data_path),
                'viz': CoverageVisualizations(self.data_path),
                'data_manager': data_manager
            }

        except Exception as e:
            logger.error("Error initializing coverage components: %s", e)
            self.components = {}

    def get_available_samples(self) -> List[str]:
        """Get list of available samples."""
        if 'coverage' in self.components:
            try:
                return self.components['coverage']['data_manager'].get_available_samples()
            except Exception as e:
                logger.warning("Error getting samples: %s", e)
        return []

    def get_summary_stats(self, sample_ids: Optional[List[str]] = None, depth_threshold: int = 10) -> Dict[str, Any]:
        """
        Get summary statistics for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze
            depth_threshold: Minimum depth threshold for calculations

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'component_type': 'coverage',
            'title': 'Coverage Analysis',
            'description': f'Analysis of coverage depth and genome recovery (min depth: {depth_threshold}x)',
            'sections': []
        }

        if 'coverage' not in self.components:
            return summary

        try:
            # Set depth threshold
            self.components['coverage']['stats'].set_depth_threshold(depth_threshold)

            stats = self.components['coverage']['stats'].calculate_coverage_stats(sample_ids)

            if stats:
                # Coverage overview metrics
                overview = stats.get('coverage_overview', {})
                if overview:
                    summary['sections'].append({
                        'title': 'Coverage Overview',
                        'type': 'metrics',
                        'data': {
                            'Total Samples': overview.get('total_samples', 0),
                            'Average Coverage': f"{overview.get('mean_coverage', 0):.1f}x",
                            'Median Coverage': f"{overview.get('median_coverage', 0):.1f}x",
                            f'High Coverage (>{depth_threshold*2}x)': overview.get('high_coverage_count', 0)
                        }
                    })

                # Genome recovery metrics
                recovery = stats.get('genome_recovery', {})
                if recovery:
                    summary['sections'].append({
                        'title': f'Genome Recovery (≥{depth_threshold}x)',
                        'type': 'metrics',
                        'data': {
                            'Average Recovery': f"{recovery.get('mean_recovery_pct', 0):.1f}%",
                            'Median Recovery': f"{recovery.get('median_recovery_pct', 0):.1f}%",
                            'Complete Genomes (>95%)': recovery.get('complete_genomes', 0),
                            'Partial Genomes (50-95%)': recovery.get('partial_genomes', 0)
                        }
                    })

                # Species coverage breakdown
                species_data = stats.get('species_coverage', {})
                if species_data:
                    summary['sections'].append({
                        'title': 'Species Coverage Breakdown',
                        'type': 'species_coverage',
                        'data': species_data
                    })

                # Coverage distribution
                distribution = stats.get('coverage_distribution', {})
                if distribution:
                    summary['sections'].append({
                        'title': 'Coverage Distribution',
                        'type': 'coverage_distribution',
                        'data': distribution
                    })

        except Exception as e:
            logger.error("Error generating coverage stats: %s", e)

        return summary

    def get_visualizations(self, sample_ids: Optional[List[str]] = None, depth_threshold: int = 10) -> Dict[str, Any]:
        """
        Get visualizations for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize
            depth_threshold: Minimum depth threshold for visualization

        Returns:
            Dictionary containing plotly figures
        """
        figures = {
            'component_type': 'coverage',
            'title': 'Coverage Analysis Visualizations',
            'figures': []
        }

        if 'coverage' not in self.components:
            return figures

        try:
            # Set depth threshold
            self.components['coverage']['viz'].set_depth_threshold(depth_threshold)

            viz_component = self.components['coverage']['viz']
            all_figures = viz_component.create_all_visualizations(sample_ids)

            if all_figures:
                for title, fig in all_figures.items():
                    if fig and fig.data:
                        figures['figures'].append({
                            'title': title,
                            'description': f'Coverage analysis: {title.lower()} (min depth: {depth_threshold}x)',
                            'figure': fig,
                            'type': 'plotly'
                        })

        except Exception as e:
            logger.error("Error generating coverage visualizations: %s", e)

        return figures

    def get_depth_profiles(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get depth profile visualizations for individual samples.

        Args:
            sample_ids: Optional list of sample IDs to profile

        Returns:
            Dictionary containing depth profile figures
        """
        profiles = {
            'component_type': 'coverage',
            'title': 'Sample Depth Profiles',
            'profiles': []
        }

        if 'coverage' not in self.components:
            return profiles

        try:
            viz_component = self.components['coverage']['viz']

            # Get available samples if none specified
            if not sample_ids:
                sample_ids = self.get_available_samples()[:5]  # Limit to first 5 for performance

            for sample_id in sample_ids:
                try:
                    profile_fig = viz_component.create_depth_profile(sample_id)
                    if profile_fig and profile_fig.data:
                        profiles['profiles'].append({
                            'sample_id': sample_id,
                            'title': f'Depth Profile: {sample_id}',
                            'figure': profile_fig,
                            'type': 'depth_profile'
                        })
                except Exception as e:
                    logger.warning("Error creating depth profile for %s: %s", sample_id, e)

        except Exception as e:
            logger.error("Error generating depth profiles: %s", e)

        return profiles

    def get_raw_data(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get raw data tables for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs

        Returns:
            Dictionary containing raw data tables
        """
        data = {
            'component_type': 'coverage',
            'title': 'Coverage Analysis Raw Data',
            'tables': []
        }

        if 'coverage' not in self.components:
            return data

        try:
            component = self.components['coverage']['stats']
            if hasattr(component, 'data') and component.data:
                for data_name, df in component.data.items():
                    if not df.empty:
                        # Filter by sample IDs if provided
                        filtered_df = df
                        if sample_ids and 'sample' in df.columns:
                            filtered_df = df[df['sample'].isin(sample_ids)]

                        if not filtered_df.empty:
                            data['tables'].append({
                                'title': f'Coverage: {data_name.replace("_", " ").title()}',
                                'data': filtered_df,
                                'type': 'dataframe'
                            })
        except Exception as e:
            logger.error("Error getting coverage raw data: %s", e)

        return data


def get_tab_info() -> Dict[str, Any]:
    """
    Get information about this tab component.

    Returns:
        Dictionary with tab metadata
    """
    return {
        'name': 'coverage',
        'title': 'Coverage Analysis',
        'icon': '📊',
        'description': 'Analysis of coverage depth, genome recovery, and depth profiles',
        'order': 30,
        'requires_data': True,
        'data_subdirs': ['coverage', 'depth']
    }


def create_tab(data_path: Path) -> CoverageTab:
    """
    Factory function to create a coverage tab.

    Args:
        data_path: Path to data directory

    Returns:
        CoverageTab instance
    """
    return CoverageTab(data_path)
