"""
Base Dash component system for modular dashboard construction.

This module provides base classes and interfaces for creating modular
Dash components that can be collected and combined into larger applications.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path

import dash
from dash import html, dcc
import dash_bootstrap_components as dbc


@dataclass
class ComponentConfig:
    """Configuration for a Dash component."""
    title: str
    icon: str = "📊"
    description: Optional[str] = None
    requires_data: bool = True
    order: int = 100  # Lower numbers appear first in navigation


@dataclass
class ComponentRegistry:
    """Registry for collecting and organizing Dash components."""
    components: Dict[str, 'DashComponent'] = field(default_factory=dict)
    categories: Dict[str, List[str]] = field(default_factory=dict)

    def register_component(self, component_id: str, component: 'DashComponent', category: str = "General"):
        """Register a component with the registry."""
        self.components[component_id] = component

        if category not in self.categories:
            self.categories[category] = []

        if component_id not in self.categories[category]:
            self.categories[category].append(component_id)

    def get_component(self, component_id: str) -> Optional['DashComponent']:
        """Get a component by ID."""
        return self.components.get(component_id)

    def get_components_by_category(self, category: str) -> List['DashComponent']:
        """Get all components in a category, sorted by order."""
        component_ids = self.categories.get(category, [])
        components = [self.components[cid] for cid in component_ids if cid in self.components]
        return sorted(components, key=lambda c: c.config.order)

    def get_all_categories(self) -> List[str]:
        """Get all available categories."""
        return sorted(self.categories.keys())


class DashComponent(ABC):
    """
    Abstract base class for Dash components.

    Each component represents a self-contained piece of functionality
    that can be rendered in a Dash application.
    """

    def __init__(self, config: ComponentConfig, data_manager=None):
        """
        Initialize the component.

        Args:
            config: Component configuration
            data_manager: Optional data manager for the component
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
        """Create the analyzer instance for this component. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def render_content(self, **kwargs) -> html.Div:
        """
        Render the main content of the component.
        Must be implemented by subclasses.

        Args:
            **kwargs: Additional arguments

        Returns:
            Dash HTML component
        """
        pass

    def render_controls(self, **kwargs) -> html.Div:
        """
        Render control elements for the component.
        Can be overridden by subclasses.

        Args:
            **kwargs: Additional arguments

        Returns:
            Dash HTML component with controls
        """
        return html.Div()

    def validate_requirements(self, **kwargs) -> bool:
        """
        Validate that all requirements for this component are met.
        Can be overridden by subclasses.

        Args:
            **kwargs: Additional arguments

        Returns:
            True if requirements are met, False otherwise
        """
        if self.config.requires_data and self.data_manager is None:
            return False
        return True

    def render_error_state(self, error_message: str) -> html.Div:
        """Render an error state for the component."""
        return dbc.Alert([
            html.H4(f"❌ {self.config.title} Error", className="alert-heading"),
            html.P(error_message),
            html.Hr(),
            html.P([
                "Troubleshooting steps:",
                html.Ul([
                    html.Li("Ensure data path is correctly configured"),
                    html.Li("Check that required data files exist"),
                    html.Li("Verify module dependencies are installed"),
                    html.Li("Check application logs for detailed error information")
                ])
            ], className="mb-0")
        ], color="danger")

    def render_loading_state(self) -> html.Div:
        """Render a loading state for the component."""
        return dbc.Spinner(
            html.Div([
                html.H5(f"Loading {self.config.title}..."),
                html.P(f"🔄 Initializing {self.config.title}")
            ]),
            color="primary"
        )

    def render(self, **kwargs) -> html.Div:
        """
        Main render method that handles the complete component rendering workflow.

        Args:
            **kwargs: Additional arguments

        Returns:
            Dash HTML component
        """
        # Validate requirements
        if not self.validate_requirements(**kwargs):
            return self.render_error_state("Component requirements not met")

        try:
            # Render controls
            controls = self.render_controls(**kwargs)
            
            # Render main content
            content = self.render_content(**kwargs)

            return html.Div([
                html.H2(f"{self.config.icon} {self.config.title}"),
                html.P(self.config.description) if self.config.description else None,
                html.Hr(),
                controls,
                content
            ])

        except Exception as e:
            return self.render_error_state(f"Error rendering component: {str(e)}")


class ModuleComponentCollector:
    """
    Helper class for collecting components from analysis modules.
    """

    @staticmethod
    def collect_components_from_module(module_name: str, data_manager=None) -> List[DashComponent]:
        """
        Collect all components from a module.

        Args:
            module_name: Name of the module to collect components from
            data_manager: Optional data manager to pass to components

        Returns:
            List of components from the module
        """
        components = []

        try:
            # Import module-specific components
            if module_name == 'read_stats':
                # Future implementation for read_stats dash components
                pass
            elif module_name == 'consensus':
                # Future implementation for consensus dash components
                pass
            elif module_name == 'coverage':
                # Future implementation for coverage dash components
                pass

        except ImportError:
            # Module doesn't have dash components or isn't available
            pass
        except Exception as e:
            # Log error but don't fail
            import logging
            logging.getLogger(__name__).warning(f"Error collecting components from {module_name}: {e}")

        return components

    @staticmethod
    def create_default_registry(data_path: Optional[str] = None) -> ComponentRegistry:
        """
        Create a default component registry with all available modules.

        Args:
            data_path: Optional data path for data managers

        Returns:
            Populated component registry
        """
        registry = ComponentRegistry()

        # List of known modules
        modules = ['read_stats', 'consensus', 'coverage']

        for module_name in modules:
            try:
                # Create data manager for module if data path provided
                data_manager = None
                if data_path:
                    # Module-specific data manager creation would go here
                    pass

                # Get components from module
                components = ModuleComponentCollector.collect_components_from_module(
                    module_name, data_manager
                )

                # Register components
                for i, component in enumerate(components):
                    component_id = f"{module_name}_{i}"
                    registry.register_component(component_id, component, module_name.title())

            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Error loading components from {module_name}: {e}")

        return registry


# Global registry instance
_global_registry = ComponentRegistry()

def get_global_registry() -> ComponentRegistry:
    """Get the global component registry."""
    return _global_registry

def register_component(component_id: str, component: DashComponent, category: str = "General"):
    """Register a component with the global registry."""
    _global_registry.register_component(component_id, component, category)


def create_dash_tab_layout(tab_content: Dict[str, Any]) -> html.Div:
    """
    Create a standardized tab layout for module content.

    Args:
        tab_content: Dictionary containing tab content data

    Returns:
        Dash HTML component with tab layout
    """
    # Summary statistics section
    summary_sections = []
    if 'summary_stats' in tab_content:
        stats_data = tab_content['summary_stats']
        for section in stats_data.get('sections', []):
            summary_sections.append(create_summary_section(section))

    # Visualizations section
    viz_sections = []
    if 'visualizations' in tab_content:
        viz_data = tab_content['visualizations']
        for figure in viz_data.get('figures', []):
            viz_sections.append(create_visualization_section(figure))

    # Raw data section
    raw_data_sections = []
    if 'raw_data' in tab_content:
        raw_data = tab_content['raw_data']
        for table in raw_data.get('tables', []):
            raw_data_sections.append(create_raw_data_section(table))

    return html.Div([
        dbc.Tabs([
            dbc.Tab(
                label="📊 Summary & Visualizations",
                tab_id="summary-viz",
                children=html.Div(summary_sections + viz_sections, style={'padding': '20px'})
            ),
            dbc.Tab(
                label="📋 Raw Data",
                tab_id="raw-data",
                children=html.Div(raw_data_sections, style={'padding': '20px'})
            )
        ], active_tab="summary-viz")
    ])


def create_summary_section(section_data: Dict[str, Any]) -> html.Div:
    """Create a summary statistics section."""
    title = section_data.get('title', 'Summary')
    section_type = section_data.get('type', 'unknown')
    data = section_data.get('data', {})

    if section_type == 'metrics':
        # Create metric cards
        metric_cards = []
        for label, value in data.items():
            metric_cards.append(
                dbc.Card([
                    dbc.CardBody([
                        html.H4(str(value), className="card-title"),
                        html.P(label, className="card-text")
                    ])
                ], className="text-center")
            )
        
        return html.Div([
            html.H4(title),
            dbc.Row([
                dbc.Col(card, width=12//len(metric_cards) if metric_cards else 12)
                for card in metric_cards
            ], className="mb-4")
        ])

    elif section_type == 'table':
        # Create table from pandas DataFrame
        if hasattr(data, 'to_dict'):
            table_data = data.to_dict('records')
            columns = [{'name': col, 'id': col} for col in data.columns]
            
            return html.Div([
                html.H4(title),
                # Would use dash_table.DataTable here
                html.Pre(str(data), style={'whiteSpace': 'pre-wrap'})
            ], className="mb-4")

    # Default rendering
    return html.Div([
        html.H4(title),
        html.Pre(str(data), style={'whiteSpace': 'pre-wrap'})
    ], className="mb-4")


def create_visualization_section(figure_data: Dict[str, Any]) -> html.Div:
    """Create a visualization section."""
    title = figure_data.get('title', 'Visualization')
    description = figure_data.get('description', '')
    figure = figure_data.get('figure')

    return html.Div([
        html.H4(title),
        html.P(description) if description else None,
        dcc.Graph(figure=figure) if figure else html.P("No visualization data available")
    ], className="mb-4")


def create_raw_data_section(table_data: Dict[str, Any]) -> html.Div:
    """Create a raw data section."""
    title = table_data.get('title', 'Raw Data')
    description = table_data.get('description', '')
    data = table_data.get('data')

    return html.Div([
        html.H4(title),
        html.P(description) if description else None,
        html.Pre(str(data), style={'whiteSpace': 'pre-wrap'}) if data is not None else html.P("No data available")
    ], className="mb-4")
