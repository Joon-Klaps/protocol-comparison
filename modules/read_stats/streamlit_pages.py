"""
Streamlit page components for read statistics analysis.

This module provides page components that can be collected and integrated
into larger Streamlit applications.
"""

from typing import Dict, Any, List

try:
    import streamlit as st
    import plotly.graph_objects as go
    import pandas as pd
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from ..streamlit_base import StreamlitPageComponent, PageConfig
from .reads.summary_stats import ReadProcessingDataManager, ReadProcessingSummaryStats
from .reads.visualizations import ReadProcessingVisualizations
from .mapping.summary_stats import MappingDataManager, MappingSummaryStats
from .mapping.visualizations import MappingVisualizations


class ReadProcessingPage(StreamlitPageComponent):
    """Page component for read processing statistics."""

    def __init__(self, data_manager=None):
        config = PageConfig(
            title="Read Processing Statistics",
            icon="📖",
            sidebar_title="Processing",
            description="Analysis of read processing pipeline including trimming and host removal statistics.",
            requires_data=True,
            order=10
        )
        super().__init__(config, data_manager)

    def create_analyzer(self):
        """Create read processing analyzer."""
        return ReadProcessingSummaryStats(self.data_manager.data_path)

    def render_sidebar(self, **kwargs) -> Dict[str, Any]:
        """Render sidebar controls for read processing page."""
        if not STREAMLIT_AVAILABLE:
            return {}

        st.sidebar.subheader("📖 Read Processing Controls")

        # Get available samples
        available_samples = []
        if self.analyzer:
            try:
                available_samples = self.analyzer.data_manager.get_available_samples()
            except Exception:
                pass

        # Sample selection
        sample_selection = st.sidebar.selectbox(
            "Sample Selection",
            options=["All Samples", "Custom Selection"],
            help="Choose whether to analyze all samples or select specific ones"
        )

        selected_samples = None
        if sample_selection == "Custom Selection" and available_samples:
            selected_samples = st.sidebar.multiselect(
                "Select Samples",
                options=available_samples,
                default=available_samples[:10] if len(available_samples) > 10 else available_samples,
                help="Select specific samples to analyze"
            )

        # Analysis options
        show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
        show_details = st.sidebar.checkbox("Show Detailed Statistics", value=False)
        show_visualization = st.sidebar.checkbox("Show Visualization", value=True)

        return {
            'selected_samples': selected_samples,
            'show_summary': show_summary,
            'show_details': show_details,
            'show_visualization': show_visualization
        }

    def render_content(self, **kwargs) -> None:
        """Render read processing analysis content."""
        if not STREAMLIT_AVAILABLE:
            return

        selected_samples = kwargs.get('selected_samples')
        show_summary = kwargs.get('show_summary', True)
        show_details = kwargs.get('show_details', False)
        show_visualization = kwargs.get('show_visualization', True)

        if not self.analyzer:
            st.error("❌ Unable to initialize read processing analyzer")
            return

        # Summary statistics
        if show_summary:
            st.subheader("📊 Summary Statistics")

            try:
                stats = self.analyzer.calculate_efficiency_stats(selected_samples)

                if stats:
                    # Create metrics columns
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Total Samples",
                            stats.get('sample_count', 0)
                        )

                    with col2:
                        retention = stats.get('overall_efficiency', {}).get('mean_retention_pct', 0)
                        st.metric(
                            "Avg Retention %",
                            f"{retention:.1f}%"
                        )

                    with col3:
                        total_raw = stats.get('total_raw_reads', 0)
                        st.metric(
                            "Total Raw Reads",
                            f"{total_raw:,}"
                        )

                    with col4:
                        total_final = stats.get('total_final_reads', 0)
                        st.metric(
                            "Total Final Reads",
                            f"{total_final:,}"
                        )

                else:
                    st.warning("⚠️ No summary statistics available")

            except Exception as e:
                st.error(f"❌ Error generating summary statistics: {str(e)}")

        # Detailed statistics
        if show_details:
            st.subheader("📋 Detailed Statistics")

            try:
                # Get detailed data
                if hasattr(self.analyzer, 'data') and 'multiqc_stats' in self.analyzer.data:
                    df = self.analyzer.data['multiqc_stats']

                    if selected_samples:
                        df = df[df.index.isin(selected_samples)]

                    if not df.empty:
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("⚠️ No detailed data available for selected samples")
                else:
                    st.warning("⚠️ No detailed statistics data available")

            except Exception as e:
                st.error(f"❌ Error displaying detailed statistics: {str(e)}")

        # Visualization
        if show_visualization:
            st.subheader("📈 Visualization")

            try:
                viz = ReadProcessingVisualizations(self.data_manager.data_path)
                fig = viz.create_processing_timeline(selected_samples)

                if fig and fig.data:
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No visualization data available")

            except Exception as e:
                st.error(f"❌ Error generating visualization: {str(e)}")


class MappingStatisticsPage(StreamlitPageComponent):
    """Page component for mapping statistics."""

    def __init__(self, data_manager=None):
        config = PageConfig(
            title="Mapping Statistics",
            icon="🗺️",
            sidebar_title="Mapping",
            description="Analysis of read mapping statistics including species and segment coverage.",
            requires_data=True,
            order=20
        )
        super().__init__(config, data_manager)

    def create_analyzer(self):
        """Create mapping analyzer."""
        return MappingSummaryStats(self.data_manager.data_path)

    def render_sidebar(self, **kwargs) -> Dict[str, Any]:
        """Render sidebar controls for mapping page."""
        if not STREAMLIT_AVAILABLE:
            return {}

        st.sidebar.subheader("🗺️ Mapping Controls")

        # Get available samples
        available_samples = []
        if self.analyzer:
            try:
                available_samples = self.analyzer.data_manager.get_available_samples()
            except Exception:
                pass

        # Sample selection
        sample_selection = st.sidebar.selectbox(
            "Sample Selection",
            options=["All Samples", "Custom Selection"],
            help="Choose whether to analyze all samples or select specific ones"
        )

        selected_samples = None
        if sample_selection == "Custom Selection" and available_samples:
            selected_samples = st.sidebar.multiselect(
                "Select Samples",
                options=available_samples,
                default=available_samples[:10] if len(available_samples) > 10 else available_samples,
                help="Select specific samples to analyze"
            )

        # Analysis options
        show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
        show_species_breakdown = st.sidebar.checkbox("Show Species Breakdown", value=True)
        show_visualization = st.sidebar.checkbox("Show Visualization", value=True)

        return {
            'selected_samples': selected_samples,
            'show_summary': show_summary,
            'show_species_breakdown': show_species_breakdown,
            'show_visualization': show_visualization
        }

    def render_content(self, **kwargs) -> None:
        """Render mapping analysis content."""
        if not STREAMLIT_AVAILABLE:
            return

        selected_samples = kwargs.get('selected_samples')
        show_summary = kwargs.get('show_summary', True)
        show_species_breakdown = kwargs.get('show_species_breakdown', True)
        show_visualization = kwargs.get('show_visualization', True)

        if not self.analyzer:
            st.error("❌ Unable to initialize mapping analyzer")
            return

        # Summary statistics
        if show_summary:
            st.subheader("📊 Mapping Summary")

            try:
                stats = self.analyzer.calculate_species_segment_stats(selected_samples)

                if stats:
                    # Calculate overall metrics
                    total_species = len(stats)
                    total_combinations = sum(len(segments) for segments in stats.values() if segments)

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric("Species Analyzed", total_species)

                    with col2:
                        st.metric("Species-Segment Combinations", total_combinations)

                    with col3:
                        sample_count = len(selected_samples) if selected_samples else len(self.analyzer.data_manager.get_available_samples())
                        st.metric("Samples", sample_count)

                else:
                    st.warning("⚠️ No mapping statistics available")

            except Exception as e:
                st.error(f"❌ Error generating mapping summary: {str(e)}")

        # Species breakdown
        if show_species_breakdown:
            st.subheader("🧬 Species Breakdown")

            try:
                stats = self.analyzer.calculate_species_segment_stats(selected_samples)

                if stats:
                    for species, segments in stats.items():
                        if segments:  # Check if segments exist
                            with st.expander(f"**{species}** ({len(segments)} segments)"):
                                for segment, segment_stats in segments.items():
                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.write(f"**{segment}**")

                                    with col2:
                                        if 'sample_count' in segment_stats:
                                            st.write(f"Samples: {segment_stats['sample_count']}")

                                    with col3:
                                        if 'avg_coverage' in segment_stats:
                                            st.write(f"Avg Coverage: {segment_stats['avg_coverage']:.1f}x")

                else:
                    st.warning("⚠️ No species breakdown available")

            except Exception as e:
                st.error(f"❌ Error generating species breakdown: {str(e)}")

        # Visualization
        if show_visualization:
            st.subheader("📈 Mapping Visualizations")

            try:
                viz = MappingVisualizations(self.data_manager.data_path)
                figures = viz.create_all_visualizations(selected_samples)

                if figures:
                    for title, fig in figures.items():
                        if fig and fig.data:
                            st.subheader(title)
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("⚠️ No visualization data available")

            except Exception as e:
                st.error(f"❌ Error generating visualizations: {str(e)}")


def get_read_stats_pages(data_manager=None) -> List[StreamlitPageComponent]:
    """
    Get all read statistics page components.

    Args:
        data_manager: Optional data manager for the pages

    Returns:
        List of read statistics page components
    """
    pages = []

    # Use a basic data manager if none provided
    if data_manager is None:
        try:
            data_manager = ReadProcessingDataManager(None)  # Will need data path later
        except Exception:
            pass  # Will handle in page validation

    # Create page components
    pages.append(ReadProcessingPage(data_manager))

    # Try to add mapping page if mapping data manager is available
    try:
        mapping_data_manager = MappingDataManager(data_manager.data_path if data_manager else None)
        pages.append(MappingStatisticsPage(mapping_data_manager))
    except Exception:
        # Add with same data manager and let page handle it
        pages.append(MappingStatisticsPage(data_manager))

    return pages
