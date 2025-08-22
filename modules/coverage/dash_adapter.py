"""
Dash adapter for coverage analysis module.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from .tab import CoverageTab as OriginalCoverageTab

logger = logging.getLogger(__name__)


class DashCoverageTab:
    """Dash adapter for coverage analysis."""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.original_tab = OriginalCoverageTab(data_path)
        self.app_id = "coverage"
    
    def get_available_samples(self) -> List[str]:
        """Get available samples."""
        return self.original_tab.get_available_samples()
    
    def create_layout(self) -> html.Div:
        """Create layout for coverage tab."""
        return html.Div([
            html.H3("📊 Coverage Analysis"),
            html.P("Coverage analysis and visualizations"),
            
            # Sample selection
            dbc.Card([
                dbc.CardBody([
                    html.H5("Sample Selection"),
                    dcc.Dropdown(
                        id=f"{self.app_id}-sample-dropdown",
                        options=[{"label": s, "value": s} for s in self.get_available_samples()],
                        multi=True,
                        placeholder="Select samples"
                    ),
                    dbc.Button(
                        "🔄 Update Analysis",
                        id=f"{self.app_id}-update-btn",
                        color="primary",
                        className="mt-3"
                    )
                ])
            ], className="mb-4"),
            
            # Results area
            html.Div(id=f"{self.app_id}-results")
        ])
    
    def register_callbacks(self, app: dash.Dash):
        """Register callbacks for coverage tab."""
        
        @app.callback(
            Output(f"{self.app_id}-results", "children"),
            [Input(f"{self.app_id}-update-btn", "n_clicks")],
            [State(f"{self.app_id}-sample-dropdown", "value")]
        )
        def update_coverage_results(n_clicks, selected_samples):
            if not selected_samples:
                return dbc.Alert("Please select samples to analyze", color="info")
            
            try:
                # Get summary stats
                summary_stats = self.original_tab.get_summary_stats(selected_samples)
                
                # Create results display
                content = []
                
                # Add overview
                if 'sections' in summary_stats:
                    for section in summary_stats['sections']:
                        content.append(self._create_section_display(section))
                else:
                    content.append(
                        dbc.Alert(f"Coverage analysis completed for {len(selected_samples)} samples", color="success")
                    )
                
                return html.Div(content)
                
            except Exception as e:
                logger.error(f"Error updating coverage results: {e}")
                return dbc.Alert(f"Error: {str(e)}", color="danger")
    
    def _create_section_display(self, section: Dict[str, Any]) -> html.Div:
        """Create display for a section."""
        title = section.get('title', 'Section')
        data = section.get('data', {})
        
        return dbc.Card([
            dbc.CardHeader(html.H5(title)),
            dbc.CardBody([
                html.Pre(str(data), style={'whiteSpace': 'pre-wrap'})
            ])
        ], className="mb-3")


def create_coverage_dash_component(data_path: Path) -> DashCoverageTab:
    """Factory function for coverage Dash component."""
    return DashCoverageTab(data_path)
