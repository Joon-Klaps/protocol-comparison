"""
Dash-specific adapter for consensus analysis module.

This adapter allows the consensus module to work with Dash while using dash_bio components
for interactive sequence alignments and clustergrams.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

# Temporarily disable dash_bio due to numpy.compat issue
# import dash_bio
try:
    import dash_bio
    DASH_BIO_AVAILABLE = True
except ImportError as e:
    DASH_BIO_AVAILABLE = False
    print(f"Warning: dash_bio not available: {e}")

# Import the original consensus module components
from .tab import ConsensusTab as OriginalConsensusTab
from .data import ConsensusDataManager
from .visualizations import ConsensusVisualizations

logger = logging.getLogger(__name__)


class DashConsensusTab:
    """
    Dash adapter for the consensus analysis tab.
    
    This class wraps the original ConsensusTab to provide Dash-specific functionality
    while enabling dash_bio component integration.
    """

    def __init__(self, data_path: Path):
        """
        Initialize Dash consensus tab.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.original_tab = OriginalConsensusTab(data_path)
        self.app_id = "consensus"

    def get_available_alignments(self) -> List[Tuple[str, str, str]]:
        """Get list of available (method, species, segment) combinations."""
        return self.original_tab.get_available_alignments()

    def get_available_samples(self, selected_key: Tuple[str, str, str]) -> List[str]:
        """Get list of available samples for a specific alignment."""
        return self.original_tab.get_available_samples(selected_key)

    def create_alignment_selector(self) -> html.Div:
        """Create alignment selection controls."""
        available_alignments = self.get_available_alignments()
        
        if not available_alignments:
            return dbc.Alert("No alignments available", color="warning")

        # Create options for dropdown
        alignment_options = []
        for method, species, segment in available_alignments:
            label = f"{method} - {species} {segment}"
            value = f"{method}|{species}|{segment}"
            alignment_options.append({"label": label, "value": value})

        return html.Div([
            html.H5("🔧 Alignment Selection"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Alignment:"),
                    dcc.Dropdown(
                        id=f"{self.app_id}-alignment-dropdown",
                        options=alignment_options,
                        value=alignment_options[0]["value"] if alignment_options else None,
                        clearable=False
                    )
                ], width=6),
                dbc.Col([
                    dbc.Label("Sample Selection:"),
                    dcc.Dropdown(
                        id=f"{self.app_id}-sample-dropdown",
                        options=[],
                        value=[],
                        multi=True,
                        placeholder="Select samples (leave empty for all)"
                    )
                ], width=6)
            ], className="mb-3"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Button(
                        "🔄 Update Analysis",
                        id=f"{self.app_id}-update-btn",
                        color="primary",
                        className="me-2"
                    ),
                    dbc.Button(
                        "📊 View Statistics",
                        id=f"{self.app_id}-stats-btn",
                        color="secondary",
                        outline=True
                    )
                ])
            ])
        ])

    def create_layout(self) -> html.Div:
        """Create the main layout for the consensus tab."""
        return html.Div([
            # Controls section
            dbc.Card([
                dbc.CardBody([
                    self.create_alignment_selector()
                ])
            ], className="mb-4"),
            
            # Main content tabs
            dbc.Tabs([
                dbc.Tab(
                    label="🧬 Interactive Alignment",
                    tab_id="alignment-tab",
                    children=html.Div(id=f"{self.app_id}-alignment-content")
                ),
                dbc.Tab(
                    label="🔗 Identity Matrix",
                    tab_id="identity-tab", 
                    children=html.Div(id=f"{self.app_id}-identity-content")
                ),
                dbc.Tab(
                    label="📊 Statistics",
                    tab_id="stats-tab",
                    children=html.Div(id=f"{self.app_id}-stats-content")
                ),
                dbc.Tab(
                    label="📋 Raw Data",
                    tab_id="raw-data-tab",
                    children=html.Div(id=f"{self.app_id}-raw-data-content")
                )
            ], 
            id=f"{self.app_id}-main-tabs",
            active_tab="alignment-tab"
            ),
            
            # Loading component
            dcc.Loading(
                id=f"{self.app_id}-loading",
                type="circle",
                children=html.Div(id=f"{self.app_id}-loading-output")
            )
        ])

    def register_callbacks(self, app: dash.Dash):
        """Register all callbacks for the consensus tab."""
        
        # Callback to update sample dropdown based on selected alignment
        @app.callback(
            Output(f"{self.app_id}-sample-dropdown", "options"),
            Input(f"{self.app_id}-alignment-dropdown", "value")
        )
        def update_sample_options(selected_alignment):
            if not selected_alignment:
                return []
                
            try:
                method, species, segment = selected_alignment.split("|")
                selected_key = (method, species, segment)
                
                available_samples = self.original_tab.get_available_samples(selected_key)
                
                return [{"label": sample, "value": sample} for sample in available_samples]
            except Exception as e:
                logger.error(f"Error updating sample options: {e}")
                return []

        # Callback to update tab content based on selection
        @app.callback(
            [Output(f"{self.app_id}-alignment-content", "children"),
             Output(f"{self.app_id}-identity-content", "children"),
             Output(f"{self.app_id}-stats-content", "children"),
             Output(f"{self.app_id}-raw-data-content", "children")],
            [Input(f"{self.app_id}-update-btn", "n_clicks"),
             Input(f"{self.app_id}-main-tabs", "active_tab")],
            [State(f"{self.app_id}-alignment-dropdown", "value"),
             State(f"{self.app_id}-sample-dropdown", "value")]
        )
        def update_tab_content(n_clicks, active_tab, selected_alignment, selected_samples):
            if not selected_alignment:
                empty_content = dbc.Alert("Please select an alignment", color="info")
                return empty_content, empty_content, empty_content, empty_content
                
            try:
                method, species, segment = selected_alignment.split("|")
                selected_key = (method, species, segment)
                
                # Get alignment content
                alignment_content = self.create_alignment_viewer(selected_key, selected_samples)
                
                # Get identity matrix content
                identity_content = self.create_identity_matrix(selected_key, selected_samples)
                
                # Get statistics content
                stats_content = self.create_statistics_content(selected_key, selected_samples)
                
                # Get raw data content
                raw_data_content = self.create_raw_data_content(selected_key, selected_samples)
                
                return alignment_content, identity_content, stats_content, raw_data_content
                
            except Exception as e:
                error_content = dbc.Alert(f"Error updating content: {str(e)}", color="danger")
                return error_content, error_content, error_content, error_content

    def create_alignment_viewer(self, selected_key: Tuple[str, str, str], sample_ids: Optional[List[str]]) -> html.Div:
        """Create the dash_bio alignment viewer."""
        try:
            # Get alignment data from the original tab's visualization component
            viz_component = self.original_tab.components.get('consensus', {}).get('viz')
            if not viz_component:
                return dbc.Alert("Visualization component not available", color="warning")
            
            # Get alignment data formatted for dash_bio
            alignment_data = viz_component.create_alignment_visualization(
                selected_key, sample_ids, max_sequences=50
            )
            
            if not alignment_data or 'data' not in alignment_data:
                return dbc.Alert("No alignment data available for selected parameters", color="info")
            
            # Parse FASTA data for dash_bio
            fasta_data = alignment_data['data']
            sequences = self._parse_fasta_for_dash_bio(fasta_data)
            
            if not sequences:
                return html.Div("No valid sequences found in alignment")
            
            if DASH_BIO_AVAILABLE:
                # Create dash_bio AlignmentChart
                alignment_chart = dash_bio.AlignmentChart(
                    id=f"{self.app_id}-alignment-chart",
                    data=sequences,
                    colorscale='clustal2',
                    showconsensus=True,
                    showgap=True,
                    showconservation=True,
                    height=600,
                    tilewidth=30
                )
                
                return html.Div([
                    html.H4(f"Multiple Sequence Alignment - {selected_key[0]} {selected_key[1]} {selected_key[2]}"),
                    html.P(f"Showing {len(sequences)} sequences"),
                    alignment_chart
                ])
            else:
                # Fallback when dash_bio is not available
                return html.Div([
                    html.H4(f"Multiple Sequence Alignment - {selected_key[0]} {selected_key[1]} {selected_key[2]}"),
                    html.P(f"Found {len(sequences)} sequences"),
                    html.P("dash_bio is not available. Please install it for interactive alignment visualization."),
                    html.Pre("\n".join([f"{seq.get('name', 'Unknown')}: {len(seq.get('sequence', ''))}" for seq in sequences[:10]]))
                ])
            
        except Exception as e:
            logger.error(f"Error creating alignment viewer: {e}")
            return html.Div(f"Error creating alignment viewer: {str(e)}")

    def create_identity_matrix(self, selected_key: Tuple[str, str, str], sample_ids: Optional[List[str]]) -> html.Div:
        """Create the identity matrix clustergram using dash_bio."""
        try:
            viz_component = self.original_tab.components.get('consensus', {}).get('viz')
            if not viz_component:
                return html.Div("Visualization component not available")
            
            # Get clustergram data
            clustergram_data = viz_component.create_identity_clustergram(selected_key, sample_ids)
            
            if not clustergram_data:
                return html.Div("No identity matrix data available")
            
            if DASH_BIO_AVAILABLE:
                # Create dash_bio Clustergram
                clustergram = dash_bio.Clustergram(
                    data=clustergram_data.get('data', []),
                    row_labels=clustergram_data.get('row_labels', []),
                    column_labels=clustergram_data.get('column_labels', []),
                    color_threshold={'row': 0.5, 'col': 0.5},
                    height=600,
                    width=800
                )
                
                return html.Div([
                    html.H4(f"Pairwise Identity Matrix - {selected_key[0]} {selected_key[1]} {selected_key[2]}"),
                    html.P("Hierarchical clustering of sequence identities"),
                    clustergram
                ])
            else:
                # Fallback when dash_bio is not available
                return html.Div([
                    html.H4(f"Pairwise Identity Matrix - {selected_key[0]} {selected_key[1]} {selected_key[2]}"),
                    html.P("dash_bio is not available. Please install it for interactive clustergram visualization."),
                    html.P(f"Matrix size: {len(clustergram_data.get('row_labels', []))} sequences")
                ])
            
        except Exception as e:
            logger.error(f"Error creating identity matrix: {e}")
            return html.Div(f"Error creating identity matrix: {str(e)}")

    def create_statistics_content(self, selected_key: Tuple[str, str, str], sample_ids: Optional[List[str]]) -> html.Div:
        """Create statistics content using Plotly figures."""
        try:
            # Get summary statistics
            summary_stats = self.original_tab.get_summary_stats([selected_key], sample_ids)
            
            # Get visualizations
            viz_data = self.original_tab.get_visualizations(selected_key, sample_ids)
            
            content = []
            
            # Add summary statistics
            for section in summary_stats.get('sections', []):
                content.append(self._create_stats_section(section))
            
            # Add visualizations
            for figure in viz_data.get('figures', []):
                if 'figure' in figure:
                    content.append(html.Div([
                        html.H5(figure.get('title', 'Visualization')),
                        html.P(figure.get('description', '')),
                        dcc.Graph(figure=figure['figure'])
                    ], className="mb-4"))
            
            return html.Div(content) if content else html.Div("No statistics available")
            
        except Exception as e:
            logger.error(f"Error creating statistics content: {e}")
            return html.Div(f"Error creating statistics: {str(e)}")

    def create_raw_data_content(self, selected_key: Tuple[str, str, str], sample_ids: Optional[List[str]]) -> html.Div:
        """Create raw data content."""
        try:
            raw_data = self.original_tab.get_raw_data(sample_ids)
            
            content = []
            for table_info in raw_data.get('tables', []):
                title = table_info.get('title', 'Raw Data')
                description = table_info.get('description', '')
                data = table_info.get('data')
                
                if data is not None:
                    content.append(html.Div([
                        html.H5(title),
                        html.P(description) if description else None,
                        html.Pre(str(data), style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
                    ], className="mb-4"))
            
            return html.Div(content) if content else html.Div("No raw data available")
            
        except Exception as e:
            logger.error(f"Error creating raw data content: {e}")
            return html.Div(f"Error creating raw data: {str(e)}")

    def _parse_fasta_for_dash_bio(self, fasta_data: str) -> List[Dict[str, str]]:
        """Parse FASTA data for dash_bio AlignmentChart."""
        sequences = []
        lines = fasta_data.split('\n')
        
        current_id = None
        current_seq = ""
        
        for line in lines:
            line = line.strip()
            if line.startswith('>'):
                if current_id:
                    sequences.append({
                        'name': current_id,
                        'sequence': current_seq
                    })
                current_id = line[1:]
                current_seq = ""
            else:
                current_seq += line
        
        if current_id:
            sequences.append({
                'name': current_id,
                'sequence': current_seq
            })
        
        return sequences

    def _create_stats_section(self, section_data: Dict[str, Any]) -> html.Div:
        """Create a statistics section."""
        title = section_data.get('title', 'Statistics')
        section_type = section_data.get('type', 'unknown')
        data = section_data.get('data', {})
        
        if section_type == 'overview':
            # Create overview metrics
            metric_cards = []
            for key, value in data.items():
                if isinstance(value, dict):
                    continue
                metric_cards.append(
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H4(str(value), className="text-primary"),
                                html.P(key.replace('_', ' ').title())
                            ])
                        ], className="text-center")
                    ], width=3)
                )
            
            return html.Div([
                html.H5(title),
                dbc.Row(metric_cards, className="mb-3")
            ])
        
        elif section_type == 'table':
            # Create table display
            return html.Div([
                html.H5(title),
                html.Pre(str(data), style={'whiteSpace': 'pre-wrap', 'fontSize': '12px'})
            ], className="mb-4")
        
        # Default rendering
        return html.Div([
            html.H5(title),
            html.Pre(str(data), style={'whiteSpace': 'pre-wrap'})
        ], className="mb-4")


def create_consensus_dash_component(data_path: Path) -> DashConsensusTab:
    """
    Factory function to create a Dash consensus component.

    Args:
        data_path: Path to data directory

    Returns:
        DashConsensusTab instance
    """
    return DashConsensusTab(data_path)
