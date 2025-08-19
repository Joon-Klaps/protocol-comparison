"""
Main module page manager for viral genomics protocol comparison.

This module provides functionality to collect and organize all Streamlit page
components from different analysis modules into a unified dashboard structure.
"""
import streamlit as st
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .streamlit_base import PageRegistry, ModulePageCollector, StreamlitPageComponent, PageConfig

logger = logging.getLogger(__name__)


class ModulePageManager:
    """
    Manager for collecting and organizing page components from all analysis modules.

    This class serves as the central registry for all Streamlit page components
    across different analysis modules (read_stats, consensus, coverage, etc.).
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize the module page manager.

        Args:
            data_path: Optional path to data directory for analysis modules
        """
        self.data_path = Path(data_path) if data_path else None
        self.registry = PageRegistry()
        self._modules_loaded = False

    def load_all_modules(self) -> None:
        """Load page components from all available analysis modules."""
        if self._modules_loaded:
            return

        # Define available modules and their categories
        module_configs = {
            'read_stats': {
                'category': 'Read Analysis',
                'order': 10
            },
            'consensus': {
                'category': 'Consensus Analysis',
                'order': 20
            },
            'coverage': {
                'category': 'Coverage Analysis',
                'order': 30
            }
        }

        for module_name, config in module_configs.items():
            try:
                self._load_module_pages(module_name, config)
            except Exception as e:
                logger.warning(f"Failed to load pages from module {module_name}: {e}")

        self._modules_loaded = True

    def _load_module_pages(self, module_name: str, config: Dict[str, Any]) -> None:
        """Load pages from a specific module."""
        try:
            # Get pages from the module
            pages = self._get_module_pages(module_name)

            # Register each page
            for i, page in enumerate(pages):
                page_id = f"{module_name}_{i}"

                # Update page order based on module order
                page.config.order = config['order'] + i

                self.registry.register_page(
                    page_id=page_id,
                    page=page,
                    category=config['category']
                )

                logger.info(f"Registered page '{page.config.title}' from module '{module_name}'")

        except ImportError:
            logger.info(f"Module '{module_name}' does not have Streamlit pages or is not available")
        except Exception as e:
            logger.error(f"Error loading pages from module '{module_name}': {e}")

    def _get_module_pages(self, module_name: str) -> List[StreamlitPageComponent]:
        """Get page components from a specific module."""
        pages = []

        if module_name == 'read_stats':
            from .read_stats.streamlit_pages import get_read_stats_pages
            from .read_stats.reads.summary_stats import ReadProcessingDataManager

            data_manager = None
            if self.data_path:
                try:
                    data_manager = ReadProcessingDataManager(self.data_path)
                except Exception as e:
                    logger.warning(f"Failed to create data manager for read_stats: {e}")

            pages = get_read_stats_pages(data_manager)

        elif module_name == 'consensus':
            # Future implementation for consensus module
            # from .consensus.streamlit_pages import get_consensus_pages
            # pages = get_consensus_pages(data_manager)
            pass

        elif module_name == 'coverage':
            # Future implementation for coverage module
            # from .coverage.streamlit_pages import get_coverage_pages
            # pages = get_coverage_pages(data_manager)
            pass

        return pages

    def get_registry(self) -> PageRegistry:
        """Get the page registry with all loaded pages."""
        if not self._modules_loaded:
            self.load_all_modules()
        return self.registry

    def get_available_pages(self) -> Dict[str, List[StreamlitPageComponent]]:
        """Get all available pages organized by category."""
        registry = self.get_registry()
        result = {}

        for category in registry.get_all_categories():
            result[category] = registry.get_pages_by_category(category)

        return result

    def get_navigation_menu_items(self) -> Dict[str, List[Dict[str, str]]]:
        """
        Get navigation menu items suitable for streamlit navigation.

        Returns:
            Dictionary with categories as keys and list of menu items as values
        """
        registry = self.get_registry()
        menu_items = {}

        for category in registry.get_all_categories():
            menu_items[category] = []
            pages = registry.get_pages_by_category(category)

            for page in pages:
                page_id = next(pid for pid, p in registry.pages.items() if p == page)
                menu_items[category].append({
                    'id': page_id,
                    'title': page.config.sidebar_title or page.config.title,
                    'icon': page.config.icon
                })

        return menu_items

    def create_page_by_id(self, page_id: str) -> Optional[StreamlitPageComponent]:
        """Get a specific page component by its ID."""
        registry = self.get_registry()
        return registry.get_page(page_id)

    def render_page(self, page_id: str, **kwargs) -> None:
        """
        Render a specific page by ID.

        Args:
            page_id: ID of the page to render
            **kwargs: Additional arguments to pass to the page
        """
        page = self.create_page_by_id(page_id)
        if page:
            try:
                page.render(**kwargs)
            except Exception as e:
                logger.error(f"Error rendering page {page_id}: {e}")
                # Import streamlit conditionally
                try:
                    import streamlit as st
                    st.error(f"Error rendering page: {str(e)}")
                except ImportError:
                    pass
        else:
            logger.error(f"Page with ID '{page_id}' not found")
            try:
                import streamlit as st
                st.error(f"Page '{page_id}' not found")
            except ImportError:
                pass

    def update_data_path(self, new_data_path: str) -> None:
        """
        Update the data path and reload modules.

        Args:
            new_data_path: New path to data directory
        """
        self.data_path = Path(new_data_path)
        self._modules_loaded = False
        self.registry = PageRegistry()  # Reset registry
        logger.info(f"Updated data path to: {new_data_path}")


# Global page manager instance
_global_page_manager = None

def get_global_page_manager(data_path: Optional[str] = None) -> ModulePageManager:
    """
    Get the global page manager instance.

    Args:
        data_path: Optional data path to initialize with

    Returns:
        Global ModulePageManager instance
    """
    global _global_page_manager

    if _global_page_manager is None:
        _global_page_manager = ModulePageManager(data_path)
    elif data_path and _global_page_manager.data_path != Path(data_path):
        _global_page_manager.update_data_path(data_path)

    return _global_page_manager

def create_main_streamlit_app(data_path: Optional[str] = None):
    """
    Create a main Streamlit application with all module pages.

    This function can be used as the entry point for a complete Streamlit dashboard
    that includes all analysis modules.

    Args:
        data_path: Optional path to data directory
    """
    # Initialize page manager
    page_manager = get_global_page_manager(data_path)

    # Get navigation items
    nav_items = page_manager.get_navigation_menu_items()

    if not nav_items:
        st.error("No pages available. Please check module installation and data path.")
        return

    # Create main navigation
    st.title("🧬 Viral Genomics Protocol Comparison Dashboard")

    # Sidebar navigation
    with st.sidebar:
        st.title("📊 Analysis Modules")

        # Category selection
        categories = list(nav_items.keys())
        if categories:
            selected_category = st.selectbox(
                "Select Analysis Category",
                categories,
                help="Choose the type of analysis to perform"
            )

            # Page selection within category
            if selected_category and selected_category in nav_items:
                category_pages = nav_items[selected_category]

                if category_pages:
                    page_options = [f"{p['icon']} {p['title']}" for p in category_pages]
                    page_ids = [p['id'] for p in category_pages]

                    selected_idx = st.selectbox(
                        f"Select {selected_category} Page",
                        range(len(page_options)),
                        format_func=lambda x: page_options[x],
                        help=f"Choose specific {selected_category.lower()} analysis"
                    )

                    if selected_idx is not None:
                        selected_page_id = page_ids[selected_idx]

                        # Render the selected page
                        page_manager.render_page(selected_page_id, data_path=data_path)
                else:
                    st.warning(f"No pages available in {selected_category}")
            else:
                st.warning("No categories available")
        else:
            st.error("No analysis modules available")


if __name__ == "__main__":
    # Example usage
    import os

    # Get data path from environment or use default
    default_data_path = os.environ.get('DEFAULT_DATA_PATH')
    create_main_streamlit_app(default_data_path)
