"""
Streamlit application for viral genomics protocol comparison analysis.

This application provides an interactive dashboard for comparing different
sequencing protocols and analyzing viral genomics data.
"""

import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
from streamlit_option_menu import option_menu

# Import our custom modules
from modules import ConsensusAnalyzer, CoverageAnalyzer, ReadStatsAnalyzer

# Configure page
st.set_page_config(
    page_title="Viral Genomics Protocol Comparison",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@st.cache_data
def load_analyzers(data_path: str) -> Dict[str, Any]:
    """
    Load and cache analyzer instances.

    Args:
        data_path: Path to the data directory

    Returns:
        Dictionary of analyzer instances
    """
    try:
        path = Path(data_path)
        if not path.exists():
            st.error(f"Data path does not exist: {data_path}")
            return {}

        analyzers = {
            'consensus': ConsensusAnalyzer(path),
            'coverage': CoverageAnalyzer(path),
            'read_stats': ReadStatsAnalyzer(path)
        }

        logger.info("Analyzers loaded successfully from %s", data_path)
        return analyzers

    except Exception as e:
        st.error(f"Failed to load analyzers: {str(e)}")
        logger.error("Failed to load analyzers: %s", str(e))
        return {}


def get_available_samples(analyzers: Dict[str, Any]) -> List[str]:
    """Get all available samples from loaded analyzers."""
    all_samples = set()
    for analyzer in analyzers.values():
        if analyzer:
            all_samples.update(analyzer.get_samples())
    return sorted(list(all_samples))


def display_stats_cards(stats: Dict[str, Any], title: str) -> None:
    """Display summary statistics as cards."""
    if not stats:
        st.info("No statistics available")
        return

    st.subheader(f"{title} Summary")

    # Create columns for stats cards
    num_categories = len([k for k, v in stats.items() if isinstance(v, dict)])
    if num_categories > 0:
        cols = st.columns(min(num_categories, 3))

        col_idx = 0
        for category, values in stats.items():
            if isinstance(values, dict):
                with cols[col_idx % 3]:
                    with st.container():
                        st.markdown(f"**{category.replace('_', ' ').title()}**")
                        for key, value in values.items():
                            if isinstance(value, float):
                                st.metric(
                                    label=key.replace('_', ' ').title(),
                                    value=f"{value:.2f}"
                                )
                            else:
                                st.metric(
                                    label=key.replace('_', ' ').title(),
                                    value=str(value)
                                )
                col_idx += 1


def consensus_analysis_page(analyzers: Dict[str, Any]) -> None:
    """Display consensus analysis page."""
    st.header("🧬 Consensus Analysis")
    st.markdown("Analyze genome recovery statistics and nucleotide identity comparisons.")

    if 'consensus' not in analyzers or not analyzers['consensus']:
        st.error("Consensus analyzer not available. Please check your data path.")
        return

    analyzer = analyzers['consensus']
    available_samples = analyzer.get_samples()

    if not available_samples:
        st.warning("No samples found in the data.")
        return

    # Sample selection
    with st.expander("Analysis Settings", expanded=True):
        selected_samples = st.multiselect(
            "Select samples for analysis:",
            options=available_samples,
            default=available_samples[:5] if len(available_samples) > 5 else available_samples,
            help="Choose which samples to include in the analysis"
        )

        analyze_btn = st.button("Generate Consensus Analysis", type="primary")

    if analyze_btn and selected_samples:
        with st.spinner("Generating consensus analysis..."):
            try:
                # Generate statistics
                stats = analyzer.generate_summary_stats(selected_samples)

                # Display stats
                if stats:
                    display_stats_cards(stats, "Consensus")

                # Generate and display visualizations
                figures = analyzer.create_visualizations(selected_samples)

                if figures:
                    st.subheader("Visualizations")

                    # Display each figure
                    for fig_name, fig in figures.items():
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No visualizations generated. Check your data format.")

            except Exception as e:
                st.error(f"Error in consensus analysis: {str(e)}")
                logger.error("Error in consensus analysis: %s", str(e))


def coverage_analysis_page(analyzers: Dict[str, Any]) -> None:
    """Display coverage analysis page."""
    st.header("📊 Coverage Analysis")
    st.markdown("Analyze coverage depth, genome recovery, and overlay plots.")

    if 'coverage' not in analyzers or not analyzers['coverage']:
        st.error("Coverage analyzer not available. Please check your data path.")
        return

    analyzer = analyzers['coverage']
    available_samples = analyzer.get_samples()

    if not available_samples:
        st.warning("No samples found in the data.")
        return

    # Analysis settings
    with st.expander("Analysis Settings", expanded=True):
        col1, col2 = st.columns(2)

        with col1:
            selected_samples = st.multiselect(
                "Select samples for analysis:",
                options=available_samples,
                default=available_samples[:5] if len(available_samples) > 5 else available_samples,
                help="Choose which samples to include in the analysis"
            )

        with col2:
            depth_threshold = st.number_input(
                "Minimum depth threshold:",
                min_value=1,
                max_value=1000,
                value=10,
                help="Minimum read depth for genome recovery calculations"
            )

        analyze_btn = st.button("Generate Coverage Analysis", type="primary")

    if analyze_btn and selected_samples:
        with st.spinner("Generating coverage analysis..."):
            try:
                # Set depth threshold
                analyzer.set_depth_threshold(depth_threshold)

                # Generate statistics
                stats = analyzer.generate_summary_stats(selected_samples)

                # Display stats
                if stats:
                    display_stats_cards(stats, "Coverage")

                # Generate and display visualizations
                figures = analyzer.create_visualizations(selected_samples)

                if figures:
                    st.subheader("Visualizations")

                    # Display each figure
                    for fig_name, fig in figures.items():
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No visualizations generated. Check your data format.")

            except Exception as e:
                st.error(f"Error in coverage analysis: {str(e)}")
                logger.error("Error in coverage analysis: %s", str(e))


def read_stats_analysis_page(analyzers: Dict[str, Any]) -> None:
    """Display read statistics analysis page."""
    st.header("📈 Read Statistics Analysis")
    st.markdown("Analyze mapping efficiency, UMI statistics, and contamination levels.")

    if 'read_stats' not in analyzers or not analyzers['read_stats']:
        st.error("Read statistics analyzer not available. Please check your data path.")
        return

    analyzer = analyzers['read_stats']
    available_samples = analyzer.get_samples()

    if not available_samples:
        st.warning("No samples found in the data.")
        return

    # Sample selection
    with st.expander("Analysis Settings", expanded=True):
        selected_samples = st.multiselect(
            "Select samples for analysis:",
            options=available_samples,
            default=available_samples[:5] if len(available_samples) > 5 else available_samples,
            help="Choose which samples to include in the analysis"
        )

        analyze_btn = st.button("Generate Read Statistics Analysis", type="primary")

    if analyze_btn and selected_samples:
        with st.spinner("Generating read statistics analysis..."):
            try:
                # Generate statistics
                stats = analyzer.generate_summary_stats(selected_samples)

                # Display stats
                if stats:
                    display_stats_cards(stats, "Read Statistics")

                # Generate and display visualizations
                figures = analyzer.create_visualizations(selected_samples)

                if figures:
                    st.subheader("Visualizations")

                    # Display each figure
                    for fig_name, fig in figures.items():
                        st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No visualizations generated. Check your data format.")

            except Exception as e:
                st.error(f"Error in read statistics analysis: {str(e)}")
                logger.error("Error in read statistics analysis: %s", str(e))


def data_overview_page(analyzers: Dict[str, Any]) -> None:
    """Display data overview page."""
    st.header("📋 Data Overview")
    st.markdown("Overview of loaded data and available samples.")

    if not analyzers:
        st.warning("No analyzers loaded. Please configure your data path in the sidebar.")
        return

    # Show analyzer status
    st.subheader("Analyzer Status")
    status_data = []

    for name, analyzer in analyzers.items():
        if analyzer:
            samples = analyzer.get_samples()
            status_data.append({
                "Analyzer": name.replace('_', ' ').title(),
                "Status": "✅ Loaded",
                "Samples": len(samples)
            })
        else:
            status_data.append({
                "Analyzer": name.replace('_', ' ').title(),
                "Status": "❌ Failed",
                "Samples": 0
            })

    if status_data:
        status_df = pd.DataFrame(status_data)
        st.dataframe(status_df, use_container_width=True)

    # Show sample overview
    all_samples = get_available_samples(analyzers)
    if all_samples:
        st.subheader(f"Available Samples ({len(all_samples)})")

        # Create columns for sample display
        cols = st.columns(4)
        for i, sample in enumerate(all_samples):
            with cols[i % 4]:
                st.markdown(f"• {sample}")
    else:
        st.warning("No samples found in the loaded data.")


def main():
    """Main Streamlit application."""

    # Sidebar configuration
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/virus.png", width=80)
        st.title("🧬 Viral Genomics")
        st.markdown("**Protocol Comparison Dashboard**")

        st.markdown("---")

        # Data path configuration
        st.subheader("📁 Data Configuration")

        # Check for default data path from environment or command line
        import os
        default_path = os.environ.get('DEFAULT_DATA_PATH', '')
        if not default_path and Path("sample_data").exists():
            default_path = "sample_data"

        data_path = st.text_input(
            "Data Directory Path:",
            value=default_path,
            help="Path to your analysis data directory",
            placeholder="/path/to/your/data"
        )

        # Load data button
        analyzers = {}
        if data_path and st.button("🔄 Load Data", type="primary"):
            with st.spinner("Loading data..."):
                analyzers = load_analyzers(data_path)
                if analyzers:
                    st.success("✅ Data loaded successfully!")
                    # Store in session state
                    st.session_state.analyzers = analyzers
                    st.session_state.data_path = data_path

        # Use cached analyzers if available
        if 'analyzers' in st.session_state:
            analyzers = st.session_state.analyzers
            if 'data_path' in st.session_state:
                st.info(f"📂 Current data: {st.session_state.data_path}")

        st.markdown("---")

        # Navigation menu
        if analyzers:
            available_samples = get_available_samples(analyzers)
            st.subheader("📊 Quick Stats")
            st.metric("Total Samples", len(available_samples))

            working_analyzers = sum(1 for a in analyzers.values() if a)
            st.metric("Working Analyzers", f"{working_analyzers}/{len(analyzers)}")

    # Main content area
    st.title("🧬 Viral Genomics Protocol Comparison Dashboard")
    st.markdown("Compare different sequencing protocols and analyze viral genomics data.")

    # Navigation menu
    if analyzers:
        selected_page = option_menu(
            menu_title=None,
            options=["Data Overview", "Consensus Analysis", "Coverage Analysis", "Read Statistics"],
            icons=["table", "dna", "bar-chart", "graph-up"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "18px"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )

        # Route to appropriate page
        if selected_page == "Data Overview":
            data_overview_page(analyzers)
        elif selected_page == "Consensus Analysis":
            consensus_analysis_page(analyzers)
        elif selected_page == "Coverage Analysis":
            coverage_analysis_page(analyzers)
        elif selected_page == "Read Statistics":
            read_stats_analysis_page(analyzers)

    else:
        # Welcome page when no data is loaded
        st.markdown("## 👋 Welcome!")
        st.markdown("""
        This dashboard helps you compare different viral genomics sequencing protocols and analyze your data.

        ### 🚀 Getting Started:

        1. **Configure your data path** in the sidebar
        2. **Load your data** using the "Load Data" button
        3. **Navigate** through different analysis tabs
        4. **Generate visualizations** and compare results

        ### 📊 Analysis Features:

        - **Consensus Analysis**: Genome recovery and ANI comparisons
        - **Coverage Analysis**: Depth analysis and overlay plots
        - **Read Statistics**: Mapping efficiency and contamination checks

        ### 💡 Need test data?

        Run the following command to generate sample data:
        ```bash
        python generate_sample_data.py --num-samples 10
        ```
        """)

        # Show data structure requirements
        with st.expander("📁 Expected Data Structure", expanded=False):
            st.code("""
data/
├── consensus/
│   ├── ani_comparison.tsv
│   ├── genome_recovery.tsv
├── coverage/
│   ├── coverage_summary.tsv
├── depth/
│   ├── sample1.depth
│   ├── sample2.depth
├── read_stats/
│   ├── read_counts.tsv
│   ├── umi_stats.tsv
├── mapping/
│   ├── mapping_stats.tsv
├── contamination/
│   ├── lasv_contamination.tsv
│   ├── hazv_contamination.tsv
└── references/
    ├── reference_mapping.tsv
            """)


if __name__ == "__main__":
    main()
