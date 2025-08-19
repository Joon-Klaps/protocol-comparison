#!/usr/bin/env python3
"""
Clean modular Streamlit application for viral genomics protocol comparison.

This application automatically discovers modules and creates tabs dynamically.
All Streamlit UI code is contained here - modules return pure data/visualizations.
"""

import streamlit as st
import sys
import os
from pathlib import Path
import importlib
import logging
from typing import Dict, List, Any, Optional
import time

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import our custom modules
from sample_selection import SampleSelectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Viral Genomics Analysis - Modular Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


class ModuleDiscovery:
    """Discovers and loads available analysis modules."""

    def __init__(self, modules_path: Path):
        """
        Initialize module discovery.

        Args:
            modules_path: Path to modules directory
        """
        self.modules_path = modules_path
        self.available_modules = {}

    def discover_modules(self) -> Dict[str, Any]:
        """
        Discover all available modules with tab.py files.

        Returns:
            Dictionary of discovered modules with their metadata
        """
        self.available_modules = {}

        if not self.modules_path.exists():
            logger.warning("Modules path does not exist: %s", self.modules_path)
            return self.available_modules

        # Look for module directories
        for module_dir in self.modules_path.iterdir():
            if module_dir.is_dir() and not module_dir.name.startswith('_'):
                tab_file = module_dir / 'tab.py'

                if tab_file.exists():
                    try:
                        # Import the module's tab component
                        module_name = f"modules.{module_dir.name}.tab"
                        tab_module = importlib.import_module(module_name)

                        # Get tab info
                        if hasattr(tab_module, 'get_tab_info'):
                            tab_info = tab_module.get_tab_info()
                            tab_info['module'] = tab_module
                            tab_info['module_path'] = module_dir

                            self.available_modules[module_dir.name] = tab_info
                            logger.info("Discovered module: %s", module_dir.name)

                    except Exception as e:
                        logger.warning("Failed to load module %s: %s", module_dir.name, e)

        return self.available_modules

    def get_ordered_modules(self) -> List[tuple]:
        """
        Get modules ordered by their specified order.

        Returns:
            List of (module_name, module_info) tuples in display order
        """
        items = list(self.available_modules.items())
        items.sort(key=lambda x: x[1].get('order', 999))
        return items


def load_css() -> str:
    """
    Load CSS from the styles.css file.

    Returns:
        CSS content as string
    """
    css_file = Path(__file__).parent / "styles.css"

    if css_file.exists():
        content = css_file.read_text(encoding='utf-8')
        logger.info("CSS file loaded successfully from: %s (%d characters)", css_file, len(content))
        return content
    else:
        logger.warning("CSS file not found: %s", css_file)
        return ""


class DataPathManager:
    """Manages data path configuration and validation."""

    @staticmethod
    def get_default_data_path() -> Optional[str]:
        """Get default data path from various sources."""
        # Check environment variable
        env_path = os.environ.get('DEFAULT_DATA_PATH', '')
        if env_path and Path(env_path).exists():
            return env_path

        # Check common locations
        possible_paths = [
            "../../data/app",
            "sample_data",
            "data",
            str(Path.cwd() / "sample_data"),
            str(Path.cwd() / "data")
        ]

        for path in possible_paths:
            if Path(path).exists():
                return path

        return None

    @staticmethod
    def validate_data_path(data_path: str) -> tuple[bool, str]:
        """
        Validate data path and return status.

        Args:
            data_path: Path to validate

        Returns:
            Tuple of (is_valid, status_message)
        """
        if not data_path:
            return False, "No data path provided"

        path = Path(data_path)
        if not path.exists():
            return False, f"Path does not exist: {data_path}"

        if not path.is_dir():
            return False, f"Path is not a directory: {data_path}"

        # Check for expected subdirectories
        expected_subdirs = ['consensus', 'custom_vcfs', 'read_stats', 'mapping']
        expected_files = ['mapping.parquet', 'reads.parquet']
        found_data = [p.name for p in path.glob("*") if p.name in expected_files or p.name in expected_subdirs]

        if not found_data:
            logger.warning("No expected data subdirectories found in %s", [p.name for p in path.glob("*")])
            return False, f"No expected data subdirectories found in {data_path}"

        return True, f"Valid data path: {data_path}"


def sync_coverage_threshold_from_slider():
    """Callback to sync threshold when slider changes."""
    if 'coverage_depth_slider' in st.session_state:
        st.session_state.coverage_depth_threshold = st.session_state.coverage_depth_slider

def sync_coverage_threshold_from_number():
    """Callback to sync threshold when number input changes."""
    if 'coverage_depth_number' in st.session_state:
        st.session_state.coverage_depth_threshold = st.session_state.coverage_depth_number

def render_sidebar() -> tuple[str, bool, bool, Optional[SampleSelectionManager]]:
    """
    Render sidebar configuration.

    Returns:
        Tuple of (data_path, data_path_valid, reload_requested, sample_selection_manager)
    """
    with st.sidebar:
        st.title("🔧 Configuration")

        # Data path configuration
        st.subheader("📁 Data Configuration")

        default_path = DataPathManager.get_default_data_path()

        data_path = st.text_input(
            "Data Directory Path:",
            value=default_path or "",
            help="Path to your analysis data directory",
            placeholder="/path/to/your/data"
        )

        # Validate data path
        data_path_valid = False
        sample_selection_manager = None

        if data_path:
            is_valid, status_msg = DataPathManager.validate_data_path(data_path)
            if is_valid:
                # st.success(status_msg)
                data_path_valid = True

                # Initialize sample selection manager
                try:
                    sample_selection_manager = SampleSelectionManager(data_path)
                    sample_selection_manager.load_preconfigured_selections()

                    # Show preconfigured selections info
                    total_selections, num_datasets = sample_selection_manager.get_selection_info_for_sidebar()

                    if total_selections == 0:
                        st.warning("⚠️ No preconfigured selections found in comparison_excels directory")

                except Exception as e:
                    st.warning(f"Error loading preconfigured selections: {str(e)}")
            else:
                st.error(status_msg)
        else:
            st.info("👆 Please enter a data directory path to get started")

        reload_requested = False
        if data_path_valid:
            if st.button("🔄 Reload Data", type="primary", use_container_width=True):
                reload_requested = True
                # Show loading animation
                with st.spinner("Loading data..."):
                    time.sleep(1)  # Brief delay for visual feedback
                    # Clear any cached data
                    if 'cached_samples' in st.session_state:
                        del st.session_state['cached_samples']
                    if 'cached_modules' in st.session_state:
                        del st.session_state['cached_modules']
                    if 'selected_preconfigured' in st.session_state:
                        del st.session_state['selected_preconfigured']

                    # Clear coverage data cache
                    try:
                        from modules.coverage.data import clear_coverage_cache
                        clear_coverage_cache()
                    except ImportError:
                        pass  # Module might not be available

                st.success("Data reloaded successfully!")
                st.rerun()
        else:
            st.button("🔄 Reload Data", disabled=True, use_container_width=True)
            st.caption("Configure valid data path first")

        # Analysis Controls Section
        if data_path_valid:
            st.markdown("---")
            st.subheader("⚙️ Analysis Controls")

            # Show available modules as checkboxes for controls
            available_module_names = []
            if 'available_modules' in st.session_state:
                available_module_names = [
                    (name, info.get('title', name)) for name, info in st.session_state['available_modules']
                ]

            if available_module_names:
                # Coverage controls toggle
                coverage_enabled = st.checkbox(
                    "📊 Coverage Analysis Controls",
                    value=st.session_state.get('show_coverage_controls', False),
                    help="Show/hide coverage analysis parameter controls",
                    key="show_coverage_controls"
                )

                if coverage_enabled:
                    # Initialize the threshold in session state if not present
                    if 'coverage_depth_threshold' not in st.session_state:
                        st.session_state.coverage_depth_threshold = 10

                    # Coverage depth threshold controls
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        # Slider control with callback
                        st.slider(
                            "Min Depth Threshold",
                            min_value=1,
                            max_value=500,
                            value=st.session_state.coverage_depth_threshold,
                            step=1,
                            help="Minimum depth required to consider a genomic position as 'recovered'",
                            key="coverage_depth_slider",
                            on_change=sync_coverage_threshold_from_slider
                        )

                    with col2:
                        # Number input control with callback
                        st.number_input(
                            "Exact Value",
                            min_value=1,
                            max_value=500,
                            value=st.session_state.coverage_depth_threshold,
                            step=1,
                            help="Type exact depth threshold",
                            key="coverage_depth_number",
                            on_change=sync_coverage_threshold_from_number
                        )

                    # Display current setting
                    current_threshold = st.session_state.coverage_depth_threshold
                    st.caption(f"🎯 Current depth threshold: **{current_threshold}x**")

                # Add checkboxes for other modules as needed
                # read_stats_enabled = st.checkbox(
                #     "📈 Read Statistics Controls",
                #     value=False,
                #     help="Show/hide read statistics parameter controls"
                # )
                #
                # consensus_enabled = st.checkbox(
                #     "🧬 Consensus Analysis Controls",
                #     value=False,
                #     help="Show/hide consensus analysis parameter controls"
                # )

            else:
                st.info("No analysis modules available")

        return data_path, data_path_valid, reload_requested, sample_selection_manager


def render_metrics_section(section_data: Dict[str, Any]):
    """Render a metrics section."""
    data = section_data.get('data', {})
    if not data:
        return

    # Create columns for metrics
    cols = st.columns(len(data))

    for i, (label, value) in enumerate(data.items()):
        with cols[i]:
            st.metric(label, value)


def render_table_section(section_data: Dict[str, Any]):
    """Render a table section."""
    data = section_data.get('data', {})
    if not data:
        return

    # Display as expandable table
    with st.expander("📋 View Details", expanded=False):
        st.json(data)


def render_species_breakdown(section_data: Dict[str, Any]):
    """Render species breakdown section."""
    data = section_data.get('data', {})
    if not data:
        return

    for species, stats in data.items():
        with st.expander(f"🦠 **{species}**"):
            if isinstance(stats, dict):
                cols = st.columns(len(stats))
                for i, (key, value) in enumerate(stats.items()):
                    with cols[i]:
                        st.metric(key.replace('_', ' ').title(), f"{value}")


def render_summary_stats(summary_data: Dict[str, Any]):
    """
    Render summary statistics section.

    Args:
        summary_data: Summary statistics data from tab component
    """
    if not summary_data or not summary_data.get('sections'):
        st.info("No summary statistics available")
        return

    for section in summary_data['sections']:
        section_type = section.get('type', 'unknown')
        title = section.get('title', 'Unknown Section')

        st.subheader(title)

        if section_type == 'metrics':
            render_metrics_section(section)
        elif section_type == 'table':
            render_table_section(section)
        elif section_type in ['species_breakdown', 'species_recovery', 'species_coverage']:
            render_species_breakdown(section)
        else:
            # Generic rendering
            data = section.get('data', {})
            if data:
                st.json(data)


def render_visualizations(viz_data: Dict[str, Any]):
    """
    Render visualizations section.

    Args:
        viz_data: Visualization data from tab component
    """
    figures = viz_data.get('figures', [])

    if not figures:
        st.info("No visualizations available")
        return

    for fig_data in figures:
        title = fig_data.get('title', 'Visualization')
        description = fig_data.get('description', '')
        figure = fig_data.get('figure')

        st.subheader(title)
        if description:
            st.caption(description)

        if figure:
            try:
                st.plotly_chart(figure, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying visualization: {str(e)}")


def render_custom_html(html_data: Dict[str, Any]):
    """
    Render custom HTML components.

    Args:
        html_data: Custom HTML data from tab component
    """
    components = html_data.get('components', [])

    if not components:
        st.info("No custom components available")
        return

    for component in components:
        title = component.get('title', 'Custom Component')
        description = component.get('description', '')
        html_content = component.get('html', '')

        st.subheader(title)
        if description:
            st.caption(description)

        if html_content:
            try:
                st.components.v1.html(html_content, height=400, scrolling=True)
            except Exception as e:
                st.error(f"Error displaying custom HTML: {str(e)}")


def render_raw_data(data_dict: Dict[str, Any]):
    """
    Render raw data tables.

    Args:
        data_dict: Raw data from tab component
    """
    tables = data_dict.get('tables', [])

    if not tables:
        st.info("No raw data available")
        return

    for table_data in tables:
        title = table_data.get('title', 'Data Table')
        df = table_data.get('data')

        with st.expander(f"📊 {title}"):
            if df is not None and not df.empty:
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No data available")


def render_module_tab(module_name: str, module_info: Dict[str, Any], data_path: str, selected_samples: Optional[List[str]]):
    """
    Render a module as a single page with sub-tabs for different content types.

    Args:
        module_name: Name of the module
        module_info: Module information dictionary
        data_path: Path to data directory
        selected_samples: Selected sample IDs
    """
    try:
        # Create tab instance
        tab_module = module_info['module']
        if not hasattr(tab_module, 'create_tab'):
            st.error(f"Module {module_name} does not have create_tab function")
            return

        tab_instance = tab_module.create_tab(Path(data_path))

        # Get available samples
        available_samples = tab_instance.get_available_samples()

        if not available_samples:
            st.warning(f"No samples found for {module_info['title']} analysis")
            return

        # Add some spacing and styling for sub-tabs
        st.markdown("<br>", unsafe_allow_html=True)

        # Create sub-tabs for this module with enhanced styling
        sub_tab1, sub_tab2 = st.tabs([
            "📊 **Summary & Plots**",
            "📋 **Raw Data**"
        ])

        # Summary & Plots Tab
        with sub_tab1:
            st.markdown('<div class="sub-tab-content">', unsafe_allow_html=True)
            # Summary Statistics Section
            st.markdown("## 📊 Summary Statistics")
            try:
                if hasattr(tab_instance, 'get_summary_stats'):
                    summary_data = tab_instance.get_summary_stats(selected_samples)
                    render_summary_stats(summary_data)
                else:
                    st.info("Summary statistics not available for this module")
            except Exception as e:
                st.error(f"Error loading summary statistics: {str(e)}")

            st.markdown("---")

            # Visualizations Section
            st.markdown("## 📈 Visualizations")
            try:
                if hasattr(tab_instance, 'get_custom_html'):
                    html_data = tab_instance.get_custom_html(selected_samples)
                    render_custom_html(html_data)
                if hasattr(tab_instance, 'get_visualizations'):
                    viz_data = tab_instance.get_visualizations(selected_samples)
                    render_visualizations(viz_data)
                else:
                    st.info("Visualizations not available for this module")
            except Exception as e:
                st.error(f"Error loading visualizations: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Raw Data Tab
        with sub_tab2:
            st.markdown('<div class="sub-tab-content">', unsafe_allow_html=True)
            st.markdown("## 📋 Raw Data Tables")
            try:
                if hasattr(tab_instance, 'get_raw_data'):
                    raw_data = tab_instance.get_raw_data(selected_samples)
                    render_raw_data(raw_data)
                else:
                    st.info("Raw data view not available for this module")
            except Exception as e:
                st.error(f"Error loading raw data: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error initializing {module_info['title']} module: {str(e)}")
        logger.error("Error in module %s: %s", module_name, e)


def main():
    """Main application entry point."""

    # Load and apply custom CSS early
    css_content = load_css()
    if css_content:
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    # Application header
    st.title("🧬 Viral Genomics Protocol Comparison")
    st.markdown("**Analysis Dashboard**")

    # Sidebar configuration
    data_path, data_path_valid, reload_requested, sample_selection_manager = render_sidebar()

    if not data_path_valid:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Analysis Dashboard! 🎉

        Each page provides:

        - **📊 Summary Statistics** - Key metrics and analysis results
        - **📈 Visualizations** - Interactive plots and charts
        - **📋 Raw Data** - Access to underlying data tables

        ### Getting Started:

        1. **Configure Data Path** 👈 Enter your data directory path in the sidebar
        2. **Select Samples** - Choose which samples to analyze
        3. **Explore Page** - Navigate through the generated analysis tabs
        """)
        return

    # Discover modules
    modules_path = Path(__file__).parent / "modules"
    discovery = ModuleDiscovery(modules_path)
    available_modules = discovery.discover_modules()

    if not available_modules:
        st.error("No analysis modules found. Please check your modules code directory.")
        return

    # Get ordered modules and store in session state for sidebar access
    ordered_modules = discovery.get_ordered_modules()
    st.session_state['available_modules'] = ordered_modules

    # Get sample information from first available module
    all_samples = []

    try:
        # Try to get samples from the first working module
        for module_name, module_info in ordered_modules:
            try:
                tab_module = module_info['module']
                temp_tab = tab_module.create_tab(Path(data_path))
                all_samples = temp_tab.get_available_samples()
                if all_samples:
                    break
            except Exception:
                continue
    except Exception as e:
        logger.warning("Error getting sample information: %s", e)

    # Sample selection
    selected_samples = None
    selected_preconfigured_info = None

    if all_samples and sample_selection_manager:
        selected_samples, selected_preconfigured_info = sample_selection_manager.render_sample_selection(all_samples)

        # Display selected preconfigured info in sidebar if available
        sample_selection_manager.render_sidebar_info(selected_preconfigured_info)

    st.markdown("---")

    # Create main tabs for modules
    if ordered_modules:
        tab_names = [f"{info.get('icon', '📊')} {info.get('title', name)}"
                    for name, info in ordered_modules]

        main_tabs = st.tabs(tab_names)

        for i, (module_name, module_info) in enumerate(ordered_modules):
            with main_tabs[i]:
                render_module_tab(module_name, module_info, data_path, selected_samples)

    # Footer
    st.markdown("---")


if __name__ == "__main__":
    main()
