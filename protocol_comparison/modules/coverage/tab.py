"""
Coverage analysis tab component.

This module collects and organizes coverage analysis components for display.
Contains no Streamlit code - returns pure data and visualizations.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
import pandas as pd

from .data import CoverageDataManager
from .summary_stats import CoverageSummaryStats
from .visualizations import CoverageVisualizations
from ...sample_selection import label_for_sample

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
        self.current_depth_threshold = 10  # Default depth threshold
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all coverage analysis components."""
        try:
            # Create a single data manager instance to share across components
            data_manager = CoverageDataManager(self.data_path)

            self.components['coverage'] = {
                'stats': CoverageSummaryStats(self.data_path, data_manager=data_manager),
                'viz': CoverageVisualizations(self.data_path, data_manager=data_manager),
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

    def render_controls(self) -> int:
        """
        Render UI controls for the coverage tab and return selected depth threshold.
        This method contains Streamlit-specific code for the controls.

        Returns:
            Selected depth threshold value
        """
        try:
            import streamlit as st

            # Create a container for controls
            with st.container():
                st.markdown("### ⚙️ Analysis Controls")

                # Create columns for slider and number input
                col1, col2 = st.columns([2, 1])

                with col1:
                    # Depth threshold slider
                    slider_value = st.slider(
                        "Minimum Depth Threshold",
                        min_value=1,
                        max_value=500,
                        value=self.current_depth_threshold,
                        step=10,
                        help="Minimum depth required to consider a genomic position as 'recovered'. "
                             "Higher values are more stringent and may result in lower recovery percentages."
                    )

                with col2:
                    # Number input box
                    number_value = st.number_input(
                        "Exact Value",
                        min_value=1,
                        max_value=500,
                        value=self.current_depth_threshold,
                        step=1,
                        help="Type exact depth threshold value"
                    )

                # Use the more recently changed value
                # Check if either control changed from current threshold
                if slider_value != self.current_depth_threshold:
                    depth_threshold = slider_value
                elif number_value != self.current_depth_threshold:
                    depth_threshold = number_value
                else:
                    depth_threshold = self.current_depth_threshold

                # Update current threshold if changed
                if depth_threshold != self.current_depth_threshold:
                    self.current_depth_threshold = depth_threshold

                # Display current threshold info
                st.info(f"🎯 Current threshold: **{depth_threshold}x** coverage")

                return depth_threshold

        except ImportError:
            # Streamlit not available, return default
            logger.warning("Streamlit not available for UI controls, using default depth threshold")
            return self.current_depth_threshold
        except Exception as e:
            logger.warning("Error rendering controls: %s", e)
            return self.current_depth_threshold

    def _get_current_depth_threshold(self) -> int:
        """
        Get the current depth threshold from sidebar controls or fallback to instance default.

        Returns:
            Current depth threshold value
        """
        try:
            import streamlit as st
            # Get from sidebar controls if available
            if 'coverage_depth_threshold' in st.session_state:
                return st.session_state['coverage_depth_threshold']
        except ImportError:
            pass

        # Fallback to instance default
        return self.current_depth_threshold

    def get_summary_stats(self, sample_ids: Optional[List[str]] = None, depth_threshold: Optional[int] = None) -> Dict[str, Any]:
        """
        Get summary statistics for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze
            depth_threshold: Minimum depth threshold for calculations (uses sidebar/current if None)

        Returns:
            Dictionary containing summary statistics
        """
        # Use current threshold from sidebar if none provided
        if depth_threshold is None:
            depth_threshold = self._get_current_depth_threshold()
        summary = {
            'component_type': 'coverage_stats',
            'title': 'Coverage Statistics Analysis',
            'description': 'Analysis of coverage statistics and genome recovery',
            'sections': []
        }

        # Coverage recovery summary
        if 'coverage' in self.components:
            try:
                # Set depth threshold
                self.components['coverage']['stats'].set_depth_threshold(depth_threshold)

                # Get recovery statistics
                recovery_stats = self.components['coverage']['stats'].calculate_recovery_stats(sample_ids)
                if recovery_stats:
                    for species, segments in recovery_stats.items():
                        summary['sections'].append({
                            'title': f'Recovery Statistics - {species}',
                            'type': 'species_breakdown',
                            'data': segments
                        })

            except Exception as e:
                logger.error("Error generating coverage stats: %s", e)

        return summary

    def get_visualizations(self, sample_ids: Optional[List[str]] = None, depth_threshold: Optional[int] = None) -> Dict[str, Any]:
        """
        Get visualizations for coverage analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize
            depth_threshold: Minimum depth threshold for visualization (uses sidebar/current if None)

        Returns:
            Dictionary containing plotly figures
        """
        # Use current threshold from sidebar if none provided
        if depth_threshold is None:
            depth_threshold = self._get_current_depth_threshold()
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
                    if fig and (fig.data or fig.layout.annotations):
                        figures['figures'].append({
                            'title': title,
                            'description': f'Coverage analysis: {title.lower()}',
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
                    if profile_fig and (profile_fig.data or profile_fig.layout.annotations):
                        profiles['profiles'].append({
                            'sample_id': sample_id,
                            'title': f'Depth Profile: {label_for_sample(str(sample_id))}',
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
            # Ensure we use the same threshold as the rest of the tab
            depth_threshold = self._get_current_depth_threshold()

            # Get recovery statistics as DataFrame
            recovery_stats = self.components['coverage']['stats'].calculate_recovery_stats(sample_ids)
            if recovery_stats:
                # Flatten nested dictionary to DataFrame
                flat_recovery_stats = []
                for species, segments in recovery_stats.items():
                    for segment, stats in segments.items():
                        row = {'species': species, 'segment': segment}
                        row.update(stats)
                        flat_recovery_stats.append(row)

                if flat_recovery_stats:
                    recovery_df = pd.DataFrame(flat_recovery_stats)
                    data['tables'].append({
                        'title': 'Coverage Recovery Statistics',
                        'data': recovery_df,
                        'type': 'dataframe'
                    })

            # Add depth statistics (flattened)
            depth_stats = self.components['coverage']['stats'].calculate_depth_stats(sample_ids)
            if depth_stats:
                flat_depth_stats = []
                for species, segments in depth_stats.items():
                    for segment, stats in segments.items():
                        row = {'species': species, 'segment': segment}
                        row.update(stats)
                        flat_depth_stats.append(row)

                if flat_depth_stats:
                    depth_df = pd.DataFrame(flat_depth_stats)
                    data['tables'].append({
                        'title': 'Coverage Depth Statistics',
                        'data': depth_df,
                        'type': 'dataframe'
                    })

            # Add per-sample, per-reference recovery table for export
            try:
                dm: CoverageDataManager = self.components['coverage']['data_manager']
                per_sample_recovery = dm.get_recovery_data(sample_ids, depth_threshold)
                records: List[Dict[str, Any]] = []
                for sample_id, ref_map in per_sample_recovery.items():
                    for reference, value in ref_map.items():
                        species = None
                        segment = None
                        try:
                            meta = dm.get_species_segment_for_reference(reference)
                            if meta:
                                species = meta.get('species')
                                segment = meta.get('segment')
                        except Exception:
                            pass
                        records.append({
                            'sample_id': sample_id,
                            'reference': reference,
                            'species': species or reference,
                            'segment': segment or 'unknown',
                            'recovery_fraction': float(value),
                            'recovery_percentage': float(value) * 100.0,
                            'depth_threshold': depth_threshold,
                        })

                if records:
                    per_sample_df = pd.DataFrame(records)
                    # Order columns for readability
                    cols = ['sample_id', 'species', 'segment', 'reference', 'recovery_fraction', 'recovery_percentage', 'depth_threshold']
                    per_sample_df = per_sample_df[cols]
                    data['tables'].append({
                        'title': 'Per-sample Recovery by Reference',
                        'data': per_sample_df,
                        'type': 'dataframe'
                    })
            except Exception as ex:
                logger.warning("Failed to build per-sample recovery table: %s", ex)

            # Optionally include the raw coverage depth dataframes for selected samples only
            # To avoid massive payloads, we only include these when specific samples are provided.
            try:
                if sample_ids:
                    dm = self.components['coverage']['data_manager']
                    for sample_id in sample_ids:
                        try:
                            sample_map = dm.get_sample_data(sample_id)
                        except Exception:
                            continue
                        for reference, df in sample_map.items():
                            if df is None or getattr(df, 'empty', True):
                                continue
                            # Title encodes sample and reference for clarity
                            data['tables'].append({
                                'title': f"Coverage Depth: {label_for_sample(str(sample_id))} - {reference}",
                                'data': df,
                                'type': 'dataframe'
                            })
            except Exception as ex:
                logger.warning("Failed adding raw coverage dataframes: %s", ex)


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
