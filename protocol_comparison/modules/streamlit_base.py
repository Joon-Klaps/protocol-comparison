"""
Base Streamlit page component system for modular dashboard construction.

This module provides base classes and interfaces for creating modular
Streamlit pages that can be collected and combined into larger applications.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
import streamlit as st
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PageConfig:
    """Configuration for a Streamlit page component."""
    title: str
    icon: str = "📊"
    sidebar_title: Optional[str] = None
    description: Optional[str] = None
    requires_data: bool = True
    order: int = 100  # Lower numbers appear first in navigation


@dataclass
class PageRegistry:
    """Registry for collecting and organizing Streamlit page components."""
    pages: Dict[str, 'StreamlitPageComponent'] = field(default_factory=dict)
    categories: Dict[str, List[str]] = field(default_factory=dict)

    def register_page(self, page_id: str, page: 'StreamlitPageComponent', category: str = "General"):
        """Register a page component with the registry."""
        self.pages[page_id] = page

        if category not in self.categories:
            self.categories[category] = []

        if page_id not in self.categories[category]:
            self.categories[category].append(page_id)

    def get_page(self, page_id: str) -> Optional['StreamlitPageComponent']:
        """Get a page component by ID."""
        return self.pages.get(page_id)

    def get_pages_by_category(self, category: str) -> List['StreamlitPageComponent']:
        """Get all pages in a category, sorted by order."""
        page_ids = self.categories.get(category, [])
        pages = [self.pages[pid] for pid in page_ids if pid in self.pages]
        return sorted(pages, key=lambda p: p.config.order)

    def get_all_categories(self) -> List[str]:
        """Get all available categories."""
        return sorted(self.categories.keys())

    def get_navigation_structure(self) -> Dict[str, Dict[str, str]]:
        """Get navigation structure for menu creation."""
        structure = {}

        for category in self.get_all_categories():
            structure[category] = {}
            pages = self.get_pages_by_category(category)

            for page in pages:
                page_id = next(pid for pid, p in self.pages.items() if p == page)
                structure[category][page_id] = {
                    'title': page.config.sidebar_title or page.config.title,
                    'icon': page.config.icon
                }

        return structure


class StreamlitPageComponent(ABC):
    """
    Abstract base class for Streamlit page components.

    Each page component represents a self-contained piece of functionality
    that can be rendered in a Streamlit application.
    """

    def __init__(self, config: PageConfig, data_manager=None):
        """
        Initialize the page component.

        Args:
            config: Page configuration
            data_manager: Optional data manager for the page
        """
        self.config = config
        self.data_manager = data_manager
        self._analyzer = None

    @property
    def analyzer(self):
        """Get or create analyzer instance."""
        if self._analyzer is None and self.data_manager is not None:
            self._analyzer = self.create_analyzer()
        return self._analyzer

    @abstractmethod
    def create_analyzer(self):
        """Create the analyzer instance for this page. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def render_content(self, **kwargs) -> None:
        """
        Render the main content of the page.
        Must be implemented by subclasses.

        Args:
            **kwargs: Additional arguments passed from the main app
        """
        pass

    def render_sidebar(self, **kwargs) -> Dict[str, Any]:
        """
        Render sidebar controls for the page.
        Can be overridden by subclasses.

        Args:
            **kwargs: Additional arguments passed from the main app

        Returns:
            Dictionary of sidebar values/settings
        """
        return {}

    def validate_requirements(self, **kwargs) -> bool:
        """
        Validate that all requirements for this page are met.
        Can be overridden by subclasses.

        Args:
            **kwargs: Additional arguments passed from the main app

        Returns:
            True if requirements are met, False otherwise
        """
        if self.config.requires_data and self.data_manager is None:
            return False
        return True

    def render_error_state(self, error_message: str) -> None:
        """Render an error state for the page."""
        st.error(f"❌ **{self.config.title} Error**")
        st.error(error_message)

        with st.expander("📋 Troubleshooting"):
            if self.config.requires_data:
                st.write("- Ensure data path is correctly configured")
                st.write("- Check that required data files exist")
            st.write("- Verify module dependencies are installed")
            st.write("- Check application logs for detailed error information")

    def render_loading_state(self) -> None:
        """Render a loading state for the page."""
        with st.spinner(f"Loading {self.config.title}..."):
            st.info(f"🔄 Initializing {self.config.title}")

    def render(self, **kwargs) -> None:
        """
        Main render method that handles the complete page rendering workflow.

        Args:
            **kwargs: Additional arguments passed from the main app
        """
        # Page header
        st.title(f"{self.config.icon} {self.config.title}")

        if self.config.description:
            st.markdown(self.config.description)
            st.divider()

        # Validate requirements
        if not self.validate_requirements(**kwargs):
            self.render_error_state("Page requirements not met")
            return

        try:
            # Render sidebar controls
            sidebar_values = self.render_sidebar(**kwargs)

            # Add sidebar values to kwargs for content rendering
            render_kwargs = {**kwargs, **sidebar_values}

            # Render main content
            self.render_content(**render_kwargs)

        except Exception as e:
            self.render_error_state(f"Error rendering page: {str(e)}")


class ModulePageCollector:
    """
    Helper class for collecting pages from analysis modules.
    """

    @staticmethod
    def collect_pages_from_module(module_name: str, data_manager=None) -> List[StreamlitPageComponent]:
        """
        Collect all page components from a module.

        Args:
            module_name: Name of the module to collect pages from
            data_manager: Optional data manager to pass to pages

        Returns:
            List of page components from the module
        """
        pages = []

        try:
            # Import the module
            if module_name == 'read_stats':
                from .read_stats.streamlit_pages import get_read_stats_pages
                pages.extend(get_read_stats_pages(data_manager))

            elif module_name == 'consensus':
                from .consensus.streamlit_pages import get_consensus_pages
                pages.extend(get_consensus_pages(data_manager))

            elif module_name == 'coverage':
                from .coverage.streamlit_pages import get_coverage_pages
                pages.extend(get_coverage_pages(data_manager))

        except ImportError as e:
            # Module doesn't have streamlit pages or isn't available
            pass
        except Exception as e:
            # Log error but don't fail
            import logging
            logging.getLogger(__name__).warning(f"Error collecting pages from {module_name}: {e}")

        return pages

    @staticmethod
    def create_default_registry(data_path: Optional[str] = None) -> PageRegistry:
        """
        Create a default page registry with all available modules.

        Args:
            data_path: Optional data path for data managers

        Returns:
            Populated page registry
        """
        registry = PageRegistry()

        # List of known modules
        modules = ['read_stats', 'consensus', 'coverage']

        for module_name in modules:
            try:
                # Create data manager for this module if path provided
                data_manager = None
                if data_path:
                    if module_name == 'read_stats':
                        from .read_stats.reads import ReadProcessingDataManager
                        data_manager = ReadProcessingDataManager(data_path)
                    # Add other data managers as needed

                # Collect pages from module
                pages = ModulePageCollector.collect_pages_from_module(module_name, data_manager)

                # Register pages
                for i, page in enumerate(pages):
                    page_id = f"{module_name}_{i}"
                    registry.register_page(page_id, page, category=module_name.replace('_', ' ').title())

            except Exception as e:
                # Log error but continue with other modules
                import logging
                logging.getLogger(__name__).warning(f"Error setting up {module_name} pages: {e}")

        return registry


# Global registry instance
_global_registry = PageRegistry()

def get_global_registry() -> PageRegistry:
    """Get the global page registry."""
    return _global_registry

def register_page(page_id: str, page: StreamlitPageComponent, category: str = "General"):
    """Register a page with the global registry."""
    _global_registry.register_page(page_id, page, category)
