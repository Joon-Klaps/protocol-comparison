#!/usr/bin/env python3
"""
Modular Streamlit application for viral genomics protocol comparison.

This application demonstrates the new modular page collection system that
automatically discovers and organizes analysis modules into a unified dashboard.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import logging

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('modular_dashboard.log')
    ]
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Viral Genomics Analysis - Modular Dashboard",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)


def main():
    """Main application entry point."""

    # Check if modules are available
    try:
        from modules import get_global_page_manager
        modules_available = True
    except ImportError as e:
        st.error(f"❌ Analysis modules not available: {str(e)}")
        st.info("Please ensure all required dependencies are installed.")
        modules_available = False
        return

    # Application header
    st.title("🧬 Viral Genomics Protocol Comparison")
    st.markdown("**Modular Analysis Dashboard**")
    st.markdown("---")

    # Sidebar configuration
    with st.sidebar:
        st.title("🔧 Configuration")

        # Data path configuration
        st.subheader("📁 Data Configuration")

        # Check for default data path from environment
        default_path = os.environ.get('DEFAULT_DATA_PATH', '')

        # Look for common data directories
        possible_paths = [
            default_path,
            "data",
            "sample_data",
            "../data",
            str(Path.cwd() / "data")
        ]

        valid_default = None
        for path in possible_paths:
            if path and Path(path).exists():
                valid_default = path
                break

        data_path = st.text_input(
            "Data Directory Path:",
            value=valid_default or "",
            help="Path to your analysis data directory",
            placeholder="/path/to/your/data"
        )

        # Initialize page manager
        page_manager = None
        if data_path and Path(data_path).exists():
            try:
                with st.spinner("Initializing analysis modules..."):
                    page_manager = get_global_page_manager(data_path)
                    page_manager.load_all_modules()

                st.success("✅ Modules initialized successfully!")

                # Show module status
                available_pages = page_manager.get_available_pages()
                total_pages = sum(len(pages) for pages in available_pages.values())

                if total_pages > 0:
                    st.metric("Available Analysis Pages", total_pages)
                    st.metric("Analysis Categories", len(available_pages))
                else:
                    st.warning("⚠️ No analysis pages found")

            except Exception as e:
                st.error(f"❌ Error initializing modules: {str(e)}")
                logger.error("Error initializing page manager: %s", str(e))

        elif data_path:
            st.error(f"❌ Data path does not exist: {data_path}")
        else:
            st.info("👆 Please enter a data directory path to get started")

        st.markdown("---")

        # Module information
        st.subheader("📊 Module Information")

        if page_manager:
            available_pages = page_manager.get_available_pages()

            if available_pages:
                for category, pages in available_pages.items():
                    with st.expander(f"{category} ({len(pages)} pages)"):
                        for page in pages:
                            st.write(f"• {page.config.title}")
            else:
                st.info("No analysis modules loaded")
        else:
            st.info("Initialize modules to see available analyses")

    # Main content area
    if not page_manager:
        # Welcome screen
        st.markdown("""
        ## Welcome to the Modular Viral Genomics Dashboard! 🎉

        This dashboard uses a **modular page collection system** that automatically discovers and organizes
        analysis components from different modules into a unified interface.

        ### Features:

        - **🧩 Modular Architecture**: Each analysis type (read stats, consensus, coverage) is a separate module
        - **📄 Page Components**: Each module contributes page components that can be combined
        - **🔄 Automatic Discovery**: The system automatically finds and registers available analysis pages
        - **🎛️ Unified Interface**: All modules are accessible through a single, consistent interface
        - **⚙️ Configurable**: Easy to add new modules and analysis types

        ### Getting Started:

        1. **Configure Data Path** 👈 Enter your data directory path in the sidebar
        2. **Initialize Modules** - The system will automatically discover available analyses
        3. **Select Analysis** - Choose from the available analysis pages
        4. **Explore Results** - Interact with the analysis tools and visualizations

        ### Module Structure:

        ```
        modules/
        ├── read_stats/           # Read processing & mapping statistics
        │   ├── streamlit_pages.py    # Page components for this module
        │   ├── reads/               # Read processing analysis
        │   └── mapping/             # Mapping analysis
        ├── consensus/            # Consensus sequence analysis (future)
        ├── coverage/             # Coverage depth analysis (future)
        ├── streamlit_base.py     # Base classes for page components
        └── main.py              # Module page manager and registry
        ```

        This modular approach makes it easy to:
        - Add new analysis types by creating new modules
        - Develop and test individual components independently
        - Maintain a clean, organized codebase
        - Scale the application as new requirements emerge
        """)

        # Show example code
        with st.expander("💻 Example: Adding a New Module"):
            st.code("""
# 1. Create a new module directory: modules/my_analysis/

# 2. Create streamlit_pages.py in your module:
from ..streamlit_base import StreamlitPageComponent, PageConfig

class MyAnalysisPage(StreamlitPageComponent):
    def __init__(self, data_manager=None):
        config = PageConfig(
            title="My Analysis",
            icon="📈",
            description="Custom analysis component"
        )
        super().__init__(config, data_manager)

    def render_content(self, **kwargs):
        st.write("My custom analysis content!")

def get_my_analysis_pages(data_manager=None):
    return [MyAnalysisPage(data_manager)]

# 3. The system automatically discovers and includes your module!
            """, language="python")

        return

    # Get navigation structure
    nav_items = page_manager.get_navigation_menu_items()

    if not nav_items:
        st.warning("⚠️ No analysis pages available. Please check your data path and module configuration.")
        return

    # Create main navigation
    st.subheader("🎯 Select Analysis")

    # Category selection
    categories = list(nav_items.keys())

    # Use columns for category selection
    col1, col2 = st.columns([1, 2])

    with col1:
        selected_category = st.selectbox(
            "Analysis Category:",
            categories,
            help="Choose the type of analysis to perform"
        )

    with col2:
        if selected_category and selected_category in nav_items:
            category_pages = nav_items[selected_category]

            if category_pages:
                page_options = [f"{p['icon']} {p['title']}" for p in category_pages]
                page_ids = [p['id'] for p in category_pages]

                selected_idx = st.selectbox(
                    "Analysis Page:",
                    range(len(page_options)),
                    format_func=lambda x: page_options[x],
                    help=f"Choose specific {selected_category.lower()} analysis"
                )

                if selected_idx is not None:
                    selected_page_id = page_ids[selected_idx]

                    st.markdown("---")

                    # Render the selected page
                    try:
                        page_manager.render_page(selected_page_id, data_path=data_path)
                    except Exception as e:
                        st.error(f"❌ Error rendering page: {str(e)}")
                        logger.error("Error rendering page %s: %s", selected_page_id, str(e))

                        # Show troubleshooting info
                        with st.expander("🔧 Troubleshooting"):
                            st.write("**Possible solutions:**")
                            st.write("- Check that your data directory contains the required files")
                            st.write("- Verify that file formats match the expected structure")
                            st.write("- Check the application logs for detailed error information")
                            st.write("- Try reloading the data or restarting the application")
            else:
                st.warning(f"⚠️ No pages available in {selected_category}")
        else:
            st.info("👆 Select a category to see available analysis pages")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Viral Genomics Protocol Comparison Dashboard - Modular Architecture Demo<br>
        Built with Streamlit and a custom modular page collection system
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
