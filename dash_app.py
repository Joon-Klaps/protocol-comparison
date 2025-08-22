#!/usr/bin/env python3
"""
Dash application for viral genomics protocol comparison.

This application replaces the Streamlit version to enable proper dash_bio component integration
for sequence alignments and bioinformatics visualizations.
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import Dict, List, Any, Optional

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import pandas as pd

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

# NOTE: Removed streamlit-dependent sample_selection import
# Will implement Dash-native sample selection in the future
# from sample_selection import SampleSelectionManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
    title="Viral Genomics Analysis - Modular Dashboard"
)
app.server.secret_key = "viral-genomics-dashboard-key"  # For session management

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
            logger.warning(f"Modules path does not exist: {self.modules_path}")
            return self.available_modules

        # Look for module directories
        for module_dir in self.modules_path.iterdir():
            if not module_dir.is_dir() or module_dir.name.startswith('.'):
                continue

            # First check if there's a Dash adapter
            dash_adapter_file = module_dir / "dash_adapter.py"
            tab_file = module_dir / "tab.py"
            
            if dash_adapter_file.exists():
                try:
                    # Import the Dash adapter
                    module_name = module_dir.name
                    spec = importlib.util.spec_from_file_location(f"modules.{module_name}.dash_adapter", dash_adapter_file)
                    dash_adapter_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(dash_adapter_module)
                    
                    # Get module info from the original tab.py for metadata
                    if tab_file.exists():
                        tab_spec = importlib.util.spec_from_file_location(f"modules.{module_name}.tab", tab_file)
                        tab_module = importlib.util.module_from_spec(tab_spec)
                        tab_spec.loader.exec_module(tab_module)
                        
                        if hasattr(tab_module, 'get_tab_info'):
                            module_info = tab_module.get_tab_info()
                            module_info['module_path'] = str(module_dir)  # Convert PosixPath to string
                            # Note: Store modules themselves for runtime use, but not in JSON-serializable data
                            self.available_modules[module_name] = module_info
                            # Store runtime objects separately
                            self.available_modules[module_name]['_dash_adapter_module'] = dash_adapter_module
                            self.available_modules[module_name]['_tab_module'] = tab_module
                            logger.info(f"Discovered Dash module: {module_name}")
                        else:
                            logger.warning(f"Module {module_name} missing get_tab_info function")
                    else:
                        logger.warning(f"Module {module_name} has dash_adapter but no tab.py for metadata")

                except Exception as e:
                    logger.error(f"Error loading Dash adapter for {module_name}: {e}")
                    
            elif tab_file.exists():
                # Fallback to original tab-only modules (legacy support)
                try:
                    module_name = module_dir.name
                    spec = importlib.util.spec_from_file_location(f"modules.{module_name}.tab", tab_file)
                    tab_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(tab_module)
                    
                    if hasattr(tab_module, 'get_tab_info'):
                        module_info = tab_module.get_tab_info()
                        module_info['module_path'] = str(module_dir)  # Convert PosixPath to string
                        # Store runtime objects separately (not JSON-serializable)
                        self.available_modules[module_name] = module_info
                        self.available_modules[module_name]['_tab_module'] = tab_module
                        self.available_modules[module_name]['_dash_adapter_module'] = None  # No Dash adapter
                        logger.warning(f"Discovered legacy module without Dash adapter: {module_name}")
                    else:
                        logger.warning(f"Module {module_name} missing get_tab_info function")

                except Exception as e:
                    logger.error(f"Error loading module {module_name}: {e}")

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


class DataPathManager:
    """Manages data path configuration and validation."""

    @staticmethod
    def get_default_data_path() -> Optional[str]:
        """Get default data path from various sources."""
        # Try environment variable first
        env_path = os.environ.get('DEFAULT_DATA_PATH')
        if env_path and Path(env_path).exists():
            return env_path
        
        # Try relative path
        relative_path = Path(__file__).parent / "data"
        if relative_path.exists():
            return str(relative_path.absolute())
            
        # Try the current working directory structure
        app_data_path = Path("../../data/app")
        if app_data_path.exists():
            return str(app_data_path.absolute())
            
        return None

    @staticmethod
    def validate_data_path(data_path: str) -> tuple[bool, str]:
        """
        Validate data path.

        Args:
            data_path: Path to validate

        Returns:
            Tuple of (is_valid, message)
        """
        if not data_path:
            return False, "Please provide a data path"
            
        path = Path(data_path)
        if not path.exists():
            return False, f"Path does not exist: {data_path}"
            
        if not path.is_dir():
            return False, f"Path is not a directory: {data_path}"
            
        return True, f"Valid data path: {data_path}"


def create_sidebar_layout() -> dbc.Col:
    """Create the sidebar layout."""
    return dbc.Col([
        dbc.Card([
            dbc.CardHeader([
                html.H4("🔧 Configuration", className="mb-0")
            ]),
            dbc.CardBody([
                # Data path configuration
                html.H5("📁 Data Configuration", className="mb-3"),
                dbc.InputGroup([
                    dbc.Input(
                        id="data-path-input",
                        placeholder="/path/to/your/data",
                        value=DataPathManager.get_default_data_path() or "",
                        type="text"
                    ),
                    dbc.Button("Validate", id="validate-path-btn", color="primary", outline=True)
                ], className="mb-3"),
                
                html.Div(id="data-path-status", className="mb-3"),
                
                # Sample selection will be added here dynamically
                html.Div(id="sample-selection-container"),
                
                # Module controls will be added here dynamically
                html.Div(id="module-controls-container")
            ])
        ])
    ], width=3, className="pe-3")


def create_main_content_layout() -> dbc.Col:
    """Create the main content layout."""
    return dbc.Col([
        # Main content area
        dcc.Loading(
            id="main-content-loading",
            children=[html.Div(id="main-content-area")],
            type="cube"
        )
    ], width=9)


def create_app_layout() -> html.Div:
    """Create the main application layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H1("🧬 Viral Genomics Protocol Comparison", className="mb-2"),
                html.P("Analysis Dashboard", className="text-muted mb-4"),
                html.Hr()
            ])
        ]),
        
        # Main layout
        dbc.Row([
            create_sidebar_layout(),
            create_main_content_layout()
        ]),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P("Viral Genomics Analysis Dashboard", 
                      className="text-center text-muted small")
            ])
        ]),
        
        # Store components for data sharing between callbacks
        dcc.Store(id="modules-store"),
        dcc.Store(id="samples-store"),
        dcc.Store(id="current-module-store"),
        dcc.Store(id="selected-samples-store")
    ], fluid=True)


# Set the app layout
app.layout = create_app_layout()


@callback(
    [Output("data-path-status", "children"),
     Output("modules-store", "data"),
     Output("samples-store", "data")],
    [Input("validate-path-btn", "n_clicks"),
     Input("data-path-input", "value")]
)
def validate_data_path_and_load_modules(n_clicks, data_path):
    """Validate data path and load modules when path is valid."""
    if not data_path:
        return (
            dbc.Alert("Please provide a data path", color="warning"),
            {},
            []
        )
    
    is_valid, message = DataPathManager.validate_data_path(data_path)
    
    if not is_valid:
        return (
            dbc.Alert(message, color="danger"),
            {},
            []
        )
    
    # Load modules
    modules_path = Path(__file__).parent / "modules"
    discovery = ModuleDiscovery(modules_path)
    available_modules = discovery.discover_modules()
    
    if not available_modules:
        return (
            dbc.Alert("No analysis modules found", color="warning"),
            {},
            []
        )

    # Create JSON-serializable modules data (exclude runtime objects)
    serializable_modules = {}
    for module_name, module_info in available_modules.items():
        # Copy the info but exclude non-serializable keys
        serializable_info = {k: v for k, v in module_info.items() 
                           if not k.startswith('_') and not callable(v)}
        serializable_modules[module_name] = serializable_info
    
    # Get all available samples from modules
    all_samples = []
    for module_name, module_info in available_modules.items():
        try:
            if '_tab_module' in module_info and hasattr(module_info['_tab_module'], 'create_tab'):
                tab_instance = module_info['_tab_module'].create_tab(Path(data_path))
                if hasattr(tab_instance, 'get_available_samples'):
                    # Handle different method signatures for different modules
                    import inspect
                    sig = inspect.signature(tab_instance.get_available_samples)
                    
                    if len(sig.parameters) == 0:
                        # Method takes no parameters (like coverage, read_stats)
                        samples = tab_instance.get_available_samples()
                    else:
                        # Method requires parameters (like consensus)
                        if module_name == 'consensus':
                            # For consensus, we need to get available keys first
                            if hasattr(tab_instance, 'get_available_keys'):
                                keys = tab_instance.get_available_keys()
                                if keys:
                                    # Use the first available key to get samples
                                    first_key = keys[0]
                                    samples = tab_instance.get_available_samples(first_key)
                                else:
                                    samples = []
                            else:
                                samples = []
                        else:
                            # Skip modules with unknown parameter requirements
                            logger.warning(f"Module {module_name} has unsupported get_available_samples signature")
                            samples = []
                    
                    all_samples.extend(samples)
        except Exception as e:
            logger.warning(f"Error getting samples from {module_name}: {e}")
    
    # Remove duplicates and sort
    all_samples = sorted(list(set(all_samples)))
    
    status_alert = dbc.Alert([
        html.I(className="fas fa-check-circle me-2"),
        f"✅ {message}",
        html.Br(),
        f"📊 Found {len(serializable_modules)} modules, {len(all_samples)} samples"
    ], color="success")
    
    return status_alert, serializable_modules, all_samples


@callback(
    Output("main-content-area", "children"),
    [Input("modules-store", "data"),
     Input("samples-store", "data")],
    [State("data-path-input", "value")]
)
def render_main_content(modules_data, samples_data, data_path):
    """Render the main content area with module tabs."""
    if not modules_data or not data_path:
        return dbc.Alert("Please configure a valid data path to begin analysis", color="info")
    
    # Create module tabs
    ordered_modules = sorted(modules_data.items(), key=lambda x: x[1].get('order', 999))
    
    tab_items = []
    tab_contents = []
    
    for module_name, module_info in ordered_modules:
        # Create tab
        tab_items.append(
            dbc.Tab(
                label=f"{module_info.get('icon', '📊')} {module_info.get('title', module_name)}",
                tab_id=module_name,
                active_tab_style={"fontWeight": "bold"}
            )
        )
        
        # Create tab content
        tab_contents.append(
            html.Div(
                id=f"{module_name}-content",
                children=[html.Div(f"Loading {module_info.get('title', module_name)}...")],
                style={"padding": "20px"}
            )
        )
    
    if not tab_items:
        return dbc.Alert("No modules available", color="warning")
    
    return html.Div([
        dbc.Tabs(
            tab_items,
            id="main-tabs",
            active_tab=ordered_modules[0][0] if ordered_modules else None,
            className="mb-4"
        ),
        html.Div(id="active-tab-content")
    ])


@callback(
    Output("active-tab-content", "children"),
    [Input("main-tabs", "active_tab")],
    [State("modules-store", "data"),
     State("samples-store", "data"),
     State("data-path-input", "value")]
)
def render_active_tab_content(active_tab, modules_data, samples_data, data_path):
    """Render content for the active tab using Dash adapters."""
    if not active_tab or not modules_data or not data_path:
        return html.Div("No content available")
    
    module_info = modules_data.get(active_tab, {})
    if not module_info:
        return dbc.Alert(f"Module {active_tab} not found", color="danger")
    
    try:
        # Re-discover modules to get runtime objects
        modules_path = Path(__file__).parent / "modules"
        discovery = ModuleDiscovery(modules_path)
        runtime_modules = discovery.discover_modules()
        
        runtime_module_info = runtime_modules.get(active_tab, {})
        if not runtime_module_info:
            return dbc.Alert(f"Runtime module {active_tab} not found", color="danger")
        
        # Check if we have a Dash adapter
        dash_adapter_module = runtime_module_info.get('_dash_adapter_module')
        
        if dash_adapter_module:
            # Use the Dash adapter
            if hasattr(dash_adapter_module, f'create_{active_tab}_dash_component'):
                # Use the factory function
                factory_func = getattr(dash_adapter_module, f'create_{active_tab}_dash_component')
                dash_component = factory_func(Path(data_path))
                
                # Create the layout and register callbacks
                layout = dash_component.create_layout()
                
                # Register callbacks for this component
                try:
                    dash_component.register_callbacks(app)
                    logger.info(f"Registered callbacks for {active_tab}")
                except Exception as callback_error:
                    logger.warning(f"Could not register callbacks for {active_tab}: {callback_error}")
                
                return layout
                
            elif hasattr(dash_adapter_module, f'Dash{active_tab.title()}Tab'):
                # Use the class directly
                dash_class = getattr(dash_adapter_module, f'Dash{active_tab.title()}Tab')
                dash_component = dash_class(Path(data_path))
                
                # Create the layout and register callbacks
                layout = dash_component.create_layout()
                
                # Register callbacks for this component
                try:
                    dash_component.register_callbacks(app)
                    logger.info(f"Registered callbacks for {active_tab}")
                except Exception as callback_error:
                    logger.warning(f"Could not register callbacks for {active_tab}: {callback_error}")
                
                return layout
            else:
                logger.warning(f"Dash adapter for {active_tab} found but no create function or class")
                return dbc.Alert(f"Dash adapter for {active_tab} is not properly configured", color="warning")
        
        else:
            # Fallback for legacy modules without Dash adapters
            tab_module = runtime_module_info.get('_tab_module')
            if tab_module and hasattr(tab_module, 'create_tab'):
                # Try to create a basic view using the original tab module
                try:
                    tab_instance = tab_module.create_tab(Path(data_path))
                    if hasattr(tab_instance, 'get_available_samples'):
                        samples = tab_instance.get_available_samples()
                        return dbc.Alert([
                            html.H4(f"📊 {module_info.get('title', active_tab)}"),
                            html.P(f"Module: {active_tab}"),
                            html.P(f"Description: {module_info.get('description', 'No description available')}"),
                            html.Hr(),
                            html.P(f"Found {len(samples)} samples", className="text-info"),
                            html.P("This module doesn't have a Dash adapter yet. Please create one to enable full functionality.", 
                                   className="text-muted"),
                        ], color="info")
                except Exception as e:
                    logger.error(f"Error creating legacy tab content for {active_tab}: {e}")
            
            return dbc.Alert([
                html.H4(f"📊 {module_info.get('title', active_tab)}"),
                html.P(f"Module: {active_tab}"),
                html.P(f"Description: {module_info.get('description', 'No description available')}"),
                html.Hr(),
                html.P("This module doesn't have a Dash adapter yet. Please create one to enable full functionality.", 
                       className="text-muted"),
                dbc.Button("Create Dash Adapter", color="primary", outline=True, disabled=True)
            ], color="info")
        
    except Exception as e:
        logger.error(f"Error rendering {active_tab}: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert(f"Error loading {active_tab}: {str(e)}", color="danger")


# Dynamic callback registration for module sub-tabs
def register_module_callbacks():
    """Register callbacks for each module's sub-tabs."""
    modules_path = Path(__file__).parent / "modules"
    discovery = ModuleDiscovery(modules_path)
    
    # This will be called after modules are discovered
    # For now, we'll create a generic callback pattern
    pass


if __name__ == "__main__":
    # Check for debug flag
    debug_mode = len(sys.argv) > 1 and sys.argv[1] == "--debug"
    
    # Run the app (updated for newer Dash versions)
    if debug_mode:
        print("🛠️  Running in debug mode with auto-reload")
        app.run(debug=True, host="0.0.0.0", port=8050)
    else:
        print("🚀 Starting Dash application")
        print("🌐 Access at: http://localhost:8050")
        app.run(debug=False, host="0.0.0.0", port=8050)
