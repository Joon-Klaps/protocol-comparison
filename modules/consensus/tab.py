"""
Consensus analysis tab component.

This module collects and organizes consensus analysis components for display.
Contains no Streamlit code - returns pure data, visualizations, and custom HTML.
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

from .data import ConsensusDataManager
from .visualizations import ConsensusVisualizations

logger = logging.getLogger(__name__)


class ConsensusTab:
    """
    Consensus analysis tab component that collects all consensus-related analysis.
    Framework-agnostic - returns data structures, figures, and custom HTML.
    """

    def __init__(self, data_path: Path):
        """
        Initialize consensus tab.

        Args:
            data_path: Path to data directory
        """
        self.data_path = Path(data_path)
        self.components = {}
        self._initialize_components()

    def _initialize_components(self):
        """Initialize all consensus analysis components."""
        try:
            # Consensus components
            data_manager = ConsensusDataManager(self.data_path)
            self.components['consensus'] = {
                'data_manager': data_manager,
                'viz': ConsensusVisualizations(self.data_path)
            }

        except Exception as e:
            logger.error("Error initializing consensus components: %s", e)
            self.components = {}

    def get_available_alignments(self) -> List[Tuple[str, str, str]]:
        """Get list of available (method, species, segment) combinations."""
        if 'consensus' in self.components:
            try:
                data_manager = self.components['consensus']['data_manager']
                return list(data_manager.alignment_data.keys())
            except Exception as e:
                logger.warning("Error getting alignments: %s", e)
        return []

    def get_available_samples(self, selected_key: Tuple[str, str, str]) -> List[str]:
        """Get list of available samples for a specific alignment."""
        if 'consensus' in self.components:
            try:
                data_manager = self.components['consensus']['data_manager']
                alignment_data = data_manager.alignment_data.get(selected_key, {})
                return list(alignment_data.keys())
            except Exception as e:
                logger.warning("Error getting samples: %s", e)
        return []

    def get_summary_stats(
        self,
        selected_keys: List[Tuple[str, str, str]],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for consensus analysis.

        Args:
            selected_keys: List of (method, species, segment) tuples
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis',
            'description': 'Analysis of alignment statistics and method comparisons',
            'sections': []
        }

        if 'consensus' not in self.components or not selected_keys:
            return summary

        try:
            data_manager = self.components['consensus']['data_manager']

            # Calculate alignment summaries using data manager methods
            alignment_summaries = {}

            for selected_key in selected_keys:
                alignment_data = data_manager.alignment_data.get(selected_key, {})

                if not alignment_data:
                    continue

                # Filter samples if specified
                if sample_ids:
                    alignment_data = {
                        sample_id: seq_record
                        for sample_id, seq_record in alignment_data.items()
                        if sample_id in sample_ids
                    }

                if not alignment_data:
                    continue

                # Get basic stats using data manager helper
                stats = data_manager.get_alignment_summary_stats(selected_key, list(alignment_data.keys()))

                # Add key information
                method, species, segment = selected_key
                stats['method'] = method
                stats['species'] = species
                stats['segment'] = segment

                alignment_summaries[f"{method}_{species}_{segment}"] = stats

            if alignment_summaries:
                # Calculate overall summary
                methods = {}
                for key, stats in alignment_summaries.items():
                    method = stats.get('method', 'unknown')
                    if method not in methods:
                        methods[method] = []
                    methods[method].append(stats)

                overall = {
                    'total_alignments': len(alignment_summaries),
                    'methods_compared': list(methods.keys()),
                    'method_summary': {}
                }

                # Calculate method-level summaries
                for method, method_stats in methods.items():
                    total_samples = sum(s.get('total_samples', 0) for s in method_stats)
                    avg_length = sum(s.get('alignment_length', 0) for s in method_stats) / len(method_stats) if method_stats else 0

                    overall['method_summary'][method] = {
                        'alignments': len(method_stats),
                        'total_samples': total_samples,
                        'avg_alignment_length': round(avg_length, 1)
                    }

                # Add overall summary section
                summary['sections'].append({
                    'title': 'Method Comparison Overview',
                    'type': 'overview',
                    'data': {
                        'Total Alignments': overall.get('total_alignments', 0),
                        'Methods Compared': ', '.join(overall.get('methods_compared', [])),
                        'Method Details': overall.get('method_summary', {})
                    }
                })

                # Individual alignment details
                for key, stats in alignment_summaries.items():
                    summary['sections'].append({
                        'title': f"{stats.get('method', 'Unknown')} - {stats.get('species', 'Unknown')} {stats.get('segment', 'Unknown')}",
                        'type': 'alignment_stats',
                        'data': {
                            'Sample Count': stats.get('total_samples', 0),
                            'Alignment Length': stats.get('alignment_length', 0),
                            'Most Divergent Sample': stats.get('most_divergent_sample', 'N/A')
                        }
                    })

        except Exception as e:
            logger.error("Error generating consensus stats: %s", e)

        return summary

    def get_visualizations(
        self,
        selected_key: Tuple[str, str, str],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get visualizations for consensus analysis.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary containing plotly figures and dash_bio components
        """
        figures = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis Visualizations',
            'figures': [],
            'dash_bio_components': []
        }

        if 'consensus' not in self.components:
            return figures

        try:
            viz_component = self.components['consensus']['viz']

            # Check for missing samples warning
            if sample_ids:
                warning = viz_component.get_missing_samples_warning(selected_key, sample_ids)
                if warning:
                    figures['missing_samples_warning'] = warning

            # Get all visualizations for this specific alignment
            all_visualizations = viz_component.create_all_visualizations(selected_key, sample_ids)

            if all_visualizations:
                # Handle dash_bio alignment viewer
                if 'alignment_viewer' in all_visualizations:
                    figures['dash_bio_components'].append({
                        'title': 'Multiple Sequence Alignment',
                        'description': 'Interactive alignment visualization using dash_bio AlignmentChart',
                        'component': all_visualizations['alignment_viewer'],
                        'type': 'alignment_chart'
                    })

                # Handle dash_bio clustergram
                if 'identity_clustergram' in all_visualizations:
                    figures['dash_bio_components'].append({
                        'title': 'Pairwise Identity Clustergram',
                        'description': 'Identity matrix clustergram with dendrogram using dash_bio',
                        'component': all_visualizations['identity_clustergram'],
                        'type': 'clustergram'
                    })

                # Handle regular plotly figures
                if 'alignment_statistics' in all_visualizations:
                    fig = all_visualizations['alignment_statistics']
                    if hasattr(fig, 'data') and fig.data:
                        figures['figures'].append({
                            'title': 'Alignment Statistics',
                            'description': f'Statistical analysis for {selected_key[0]} {selected_key[1]} {selected_key[2]}',
                            'figure': fig,
                            'type': 'plotly'
                        })

        except Exception as e:
            logger.error("Error generating consensus visualizations: %s", e)

        return figures

    def get_custom_html(
        self,
        selected_key: Tuple[str, str, str],
        sample_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get custom HTML components for consensus analysis.
        This includes the dash_bio alignment viewer and clustergram instructions.

        Args:
            selected_key: Tuple of (method, species, segment)
            sample_ids: Optional list of sample IDs

        Returns:
            Dictionary containing custom HTML components
        """
        html_components = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis - Interactive Views',
            'components': []
        }

        if 'consensus' not in self.components:
            return html_components

        try:
            viz_component = self.components['consensus']['viz']

            # Get alignment visualization data
            alignment_data = viz_component.create_alignment_visualization(selected_key, sample_ids, max_sequences=50)
            if alignment_data:
                # Create HTML for dash_bio alignment viewer instructions
                alignment_html = self._create_alignment_viewer_html(alignment_data)
                if alignment_html:
                    html_components['components'].append({
                        'title': 'Multiple Sequence Alignment Viewer',
                        'description': 'Interactive alignment visualization with gap-only columns removed',
                        'html': alignment_html,
                        'type': 'alignment_viewer'
                    })

            # Get clustergram data
            clustergram_data = viz_component.create_identity_clustergram(selected_key, sample_ids)
            if clustergram_data:
                # Create HTML for dash_bio clustergram instructions
                clustergram_html = self._create_clustergram_html(clustergram_data)
                if clustergram_html:
                    html_components['components'].append({
                        'title': 'Pairwise Identity Clustergram',
                        'description': 'Identity matrix clustergram with hierarchical clustering',
                        'html': clustergram_html,
                        'type': 'clustergram'
                    })

        except Exception as e:
            logger.error("Error generating custom HTML: %s", e)

        return html_components

    def _create_alignment_viewer_html(self, alignment_data: Dict[str, Any]) -> str:
        """
        Create HTML for dash_bio alignment viewer.

        Args:
            alignment_data: Alignment data dictionary from visualizations

        Returns:
            HTML string containing alignment viewer
        """
        if not alignment_data or 'data' not in alignment_data:
            return ""

        # Create a simple alignment viewer using HTML/CSS/JS
        # This is a fallback since dash_bio requires dash framework
        # For a full implementation, you'd need to integrate dash_bio properly

        fasta_data = alignment_data['data']
        lines = fasta_data.split('\n')

        # Parse FASTA data
        sequences = []
        current_id = None
        current_seq = ""

        for line in lines:
            if line.startswith('>'):
                if current_id:
                    sequences.append({'id': current_id, 'seq': current_seq})
                current_id = line[1:].strip()
                current_seq = ""
            else:
                current_seq += line.strip()

        if current_id:
            sequences.append({'id': current_id, 'seq': current_seq})

        if not sequences:
            return ""

        # Create HTML table for alignment
        seq_length = len(sequences[0]['seq']) if sequences else 0

        # Limit display for performance
        display_width = min(seq_length, 500)  # Show first 500 positions
        display_sequences = sequences[:20]  # Show first 20 sequences

        table_rows = ""
        for seq in display_sequences:
            sequence_segment = seq['seq'][:display_width]
            colored_sequence = ""

            # Simple coloring based on nucleotide
            for nucleotide in sequence_segment:
                color = {
                    'A': '#FF6B6B', 'a': '#FF6B6B',  # Red
                    'T': '#4ECDC4', 't': '#4ECDC4',  # Teal
                    'G': '#45B7D1', 'g': '#45B7D1',  # Blue
                    'C': '#96CEB4', 'c': '#96CEB4',  # Green
                    'N': '#FFA07A', 'n': '#FFA07A',  # Light salmon
                    '-': '#E0E0E0'                   # Light gray
                }.get(nucleotide, '#E0E0E0')

                colored_sequence += f'<span style="background-color: {color}; padding: 1px; margin: 0; font-family: monospace; font-size: 10px;">{nucleotide}</span>'

            table_rows += f"""
                <tr>
                    <td class="seq-id">{seq['id'][:20]}...</td>
                    <td class="seq-data">{colored_sequence}</td>
                </tr>
            """

        html = f"""
        <div class="alignment-viewer">
            <div class="alignment-header">
                <h3>🧬 Multiple Sequence Alignment</h3>
                <p>Showing {len(display_sequences)} sequences, first {display_width} positions (gap-only columns removed)</p>
                <div class="legend">
                    <span class="legend-item"><span style="background: #FF6B6B;">A</span> Adenine</span>
                    <span class="legend-item"><span style="background: #4ECDC4;">T</span> Thymine</span>
                    <span class="legend-item"><span style="background: #45B7D1;">G</span> Guanine</span>
                    <span class="legend-item"><span style="background: #96CEB4;">C</span> Cytosine</span>
                    <span class="legend-item"><span style="background: #FFA07A;">N</span> Ambiguous</span>
                    <span class="legend-item"><span style="background: #E0E0E0;">-</span> Gap</span>
                </div>
            </div>

            <div class="alignment-container">
                <table class="alignment-table">
                    <thead>
                        <tr>
                            <th>Sequence ID</th>
                            <th>Alignment</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>

            <style>
                .alignment-viewer {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #dee2e6;
                    margin: 10px 0;
                    max-width: 100%;
                }}
                .alignment-header h3 {{
                    margin: 0 0 10px 0;
                    color: #495057;
                }}
                .legend {{
                    display: flex;
                    gap: 15px;
                    margin: 10px 0;
                    flex-wrap: wrap;
                }}
                .legend-item {{
                    display: flex;
                    align-items: center;
                    gap: 5px;
                    font-size: 0.9em;
                }}
                .legend-item span:first-child {{
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: monospace;
                    font-weight: bold;
                }}
                .alignment-container {{
                    overflow-x: auto;
                    max-height: 600px;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                }}
                .alignment-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-family: monospace;
                }}
                .alignment-table th {{
                    background: #e9ecef;
                    padding: 8px;
                    text-align: left;
                    border-bottom: 2px solid #dee2e6;
                    position: sticky;
                    top: 0;
                }}
                .alignment-table td {{
                    padding: 4px 8px;
                    border-bottom: 1px solid #dee2e6;
                    vertical-align: top;
                }}
                .seq-id {{
                    background: #f8f9fa;
                    font-weight: bold;
                    min-width: 150px;
                    max-width: 150px;
                    position: sticky;
                    left: 0;
                    border-right: 2px solid #dee2e6;
                }}
                .seq-data {{
                    font-family: monospace;
                    font-size: 10px;
                    line-height: 1.2;
                    word-break: break-all;
                }}
            </style>
        </div>
        """
        return html

    def _create_clustergram_html(self, clustergram_data: Dict[str, Any]) -> str:
        """
        Create HTML for dash_bio clustergram instructions.

        Args:
            clustergram_data: Clustergram data dictionary from visualizations

        Returns:
            HTML string containing clustergram information
        """
        if not clustergram_data:
            return ""

        # Extract information from clustergram data
        data_matrix = clustergram_data.get('data', [])
        row_labels = clustergram_data.get('row_labels', [])
        col_labels = clustergram_data.get('column_labels', [])

        if not data_matrix or not row_labels:
            return ""

        # Create a simple HTML representation of the identity matrix
        matrix_size = len(row_labels)

        # Create header row
        header_html = "<tr><th>Sample</th>"
        for label in col_labels[:10]:  # Limit display for performance
            header_html += f"<th>{label[:8]}...</th>"
        header_html += "</tr>"

        # Create data rows
        rows_html = ""
        for i, row_label in enumerate(row_labels[:10]):  # Limit display
            row_html = f"<tr><td class='matrix-label'>{row_label[:8]}...</td>"
            for j, value in enumerate(data_matrix[i][:10]):  # Limit columns
                # Color based on identity value (0-1 scale)
                if isinstance(value, (int, float)):
                    intensity = max(0, min(1, value))  # Clamp to 0-1
                    color_intensity = int(255 * (1 - intensity))  # Invert for better visualization
                    bg_color = f"rgb({color_intensity}, {255 - color_intensity//2}, {255 - color_intensity//4})"
                    row_html += f"<td style='background-color: {bg_color}; text-align: center;'>{value:.3f}</td>"
                else:
                    row_html += f"<td style='text-align: center;'>{value}</td>"
            row_html += "</tr>"
            rows_html += row_html

        html = f"""
        <div class="clustergram-viewer">
            <div class="clustergram-header">
                <h3>🔍 Pairwise Identity Matrix</h3>
                <p>Identity matrix for {matrix_size} samples (showing first 10x10 for display)</p>
                <p><strong>Note:</strong> This is a simplified view. For full interactive clustergram with dendrogram, integrate dash_bio.Clustergram component.</p>
            </div>

            <div class="matrix-container">
                <table class="identity-matrix">
                    {header_html}
                    {rows_html}
                </table>
            </div>

            <div class="matrix-info">
                <h4>Matrix Information:</h4>
                <ul>
                    <li>Matrix size: {matrix_size} x {matrix_size}</li>
                    <li>Values represent pairwise sequence identity (0.0 = 0%, 1.0 = 100%)</li>
                    <li>Higher values (lighter colors) indicate more similar sequences</li>
                    <li>Diagonal values should be 1.0 (100% identity with self)</li>
                </ul>
            </div>

            <style>
                .clustergram-viewer {{
                    background: white;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 20px;
                    margin: 10px 0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                }}
                .clustergram-header {{
                    margin-bottom: 20px;
                    padding-bottom: 10px;
                    border-bottom: 2px solid #e9ecef;
                }}
                .clustergram-header h3 {{
                    color: #2c3e50;
                    margin: 0 0 10px 0;
                }}
                .matrix-container {{
                    overflow-x: auto;
                    max-height: 400px;
                    overflow-y: auto;
                    border: 1px solid #dee2e6;
                    border-radius: 5px;
                    margin-bottom: 15px;
                }}
                .identity-matrix {{
                    width: 100%;
                    border-collapse: collapse;
                    font-family: monospace;
                    font-size: 12px;
                }}
                .identity-matrix th {{
                    background: #343a40;
                    color: white;
                    padding: 8px;
                    text-align: center;
                    border: 1px solid #dee2e6;
                    position: sticky;
                    top: 0;
                    z-index: 10;
                }}
                .identity-matrix td {{
                    padding: 6px 8px;
                    border: 1px solid #dee2e6;
                    font-size: 10px;
                }}
                .matrix-label {{
                    background: #f8f9fa !important;
                    font-weight: bold;
                    position: sticky;
                    left: 0;
                    border-right: 2px solid #343a40 !important;
                    z-index: 5;
                }}
                .matrix-info {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                }}
                .matrix-info h4 {{
                    margin: 0 0 10px 0;
                    color: #495057;
                }}
                .matrix-info ul {{
                    margin: 0;
                    padding-left: 20px;
                }}
                .matrix-info li {{
                    margin-bottom: 5px;
                }}
            </style>
        </div>
        """
        return html

    def _create_genome_recovery_html(self, recovery_data: Dict[str, Any]) -> str:
        """
        Create custom HTML for genome recovery visualization.
        Replace this with calls to your mini HTML package.

        Args:
            recovery_data: Genome recovery statistics

        Returns:
            HTML string
        """
        if not recovery_data:
            return ""

        # Example custom HTML - replace with your mini package
        html = f"""
        <div class="genome-recovery-dashboard">
            <div class="recovery-header">
                <h3>🧬 Genome Recovery Analysis</h3>
                <p>Comprehensive analysis of genome reconstruction quality</p>
            </div>

            <div class="recovery-metrics">
                <div class="metric-card">
                    <h4>Total Samples</h4>
                    <span class="metric-value">{recovery_data.get('total_samples', 0)}</span>
                </div>
                <div class="metric-card">
                    <h4>Average Recovery</h4>
                    <span class="metric-value">{recovery_data.get('mean_recovery_pct', 0):.1f}%</span>
                </div>
                <div class="metric-card">
                    <h4>High Quality</h4>
                    <span class="metric-value">{recovery_data.get('high_quality_count', 0)}</span>
                </div>
            </div>

            <style>
                .genome-recovery-dashboard {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 20px;
                    border-radius: 10px;
                    color: white;
                    margin: 10px 0;
                }}
                .recovery-header h3 {{
                    margin: 0 0 5px 0;
                    font-size: 1.5em;
                }}
                .recovery-metrics {{
                    display: flex;
                    gap: 15px;
                    margin-top: 15px;
                }}
                .metric-card {{
                    background: rgba(255,255,255,0.1);
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    flex: 1;
                }}
                .metric-card h4 {{
                    margin: 0 0 5px 0;
                    font-size: 0.9em;
                    opacity: 0.8;
                }}
                .metric-value {{
                    font-size: 1.8em;
                    font-weight: bold;
                }}
            </style>
        </div>
        """
        return html

    def _create_species_comparison_html(self, species_data: Dict[str, Any]) -> str:
        """
        Create custom HTML for species comparison.
        Replace this with calls to your mini HTML package.

        Args:
            species_data: Species breakdown data

        Returns:
            HTML string
        """
        if not species_data:
            return ""

        # Example custom HTML - replace with your mini package
        species_items = ""
        for species, data in species_data.items():
            recovery = data.get('mean_recovery', 0)
            color = "green" if recovery > 80 else "orange" if recovery > 50 else "red"
            species_items += f"""
                <div class="species-item">
                    <span class="species-name">{species}</span>
                    <div class="recovery-bar">
                        <div class="recovery-fill" style="width: {recovery}%; background-color: {color};"></div>
                        <span class="recovery-text">{recovery:.1f}%</span>
                    </div>
                </div>
            """

        html = f"""
        <div class="species-comparison">
            <div class="comparison-header">
                <h3>🦠 Species Recovery Comparison</h3>
                <p>Recovery rates across different viral species</p>
            </div>

            <div class="species-list">
                {species_items}
            </div>

            <style>
                .species-comparison {{
                    background: #f8f9fa;
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #dee2e6;
                    margin: 10px 0;
                }}
                .comparison-header h3 {{
                    margin: 0 0 5px 0;
                    color: #495057;
                }}
                .species-list {{
                    margin-top: 15px;
                }}
                .species-item {{
                    display: flex;
                    align-items: center;
                    margin: 10px 0;
                    gap: 15px;
                }}
                .species-name {{
                    font-weight: bold;
                    min-width: 150px;
                    color: #495057;
                }}
                .recovery-bar {{
                    flex: 1;
                    height: 25px;
                    background: #e9ecef;
                    border-radius: 12px;
                    position: relative;
                    overflow: hidden;
                }}
                .recovery-fill {{
                    height: 100%;
                    border-radius: 12px;
                    transition: width 0.3s ease;
                }}
                .recovery-text {{
                    position: absolute;
                    right: 10px;
                    top: 50%;
                    transform: translateY(-50%);
                    font-weight: bold;
                    color: #495057;
                    font-size: 0.9em;
                }}
            </style>
        </div>
        """
        return html

    def get_raw_data(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get raw data tables for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs

        Returns:
            Dictionary containing raw data tables
        """
        data = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis Raw Data',
            'tables': []
        }

        if 'consensus' not in self.components:
            return data

        try:
            component = self.components['consensus']['stats']
            if hasattr(component, 'data') and component.data:
                for data_name, df in component.data.items():
                    if not df.empty:
                        # Filter by sample IDs if provided
                        filtered_df = df
                        if sample_ids and 'sample' in df.columns:
                            filtered_df = df[df['sample'].isin(sample_ids)]

                        if not filtered_df.empty:
                            data['tables'].append({
                                'title': f'Consensus: {data_name.replace("_", " ").title()}',
                                'data': filtered_df,
                                'type': 'dataframe'
                            })
        except Exception as e:
            logger.error("Error getting consensus raw data: %s", e)

        return data


def get_tab_info() -> Dict[str, Any]:
    """
    Get information about this tab component.

    Returns:
        Dictionary with tab metadata
    """
    return {
        'name': 'consensus',
        'title': 'Consensus Analysis',
        'icon': '🧬',
        'description': 'Analysis of genome recovery statistics and nucleotide identity comparisons',
        'order': 20,
        'requires_data': True,
        'data_subdirs': ['consensus']
    }


def create_tab(data_path: Path) -> ConsensusTab:
    """
    Factory function to create a consensus tab.

    Args:
        data_path: Path to data directory

    Returns:
        ConsensusTab instance
    """
    return ConsensusTab(data_path)
