"""
Consensus analysis tab component.

This module collects and organizes consensus analysis components for display.
Contains no Streamlit code - returns pure data, visualizations, and custom HTML.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

from .summary_stats import ConsensusDataManager, ConsensusSummaryStats
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
                'stats': ConsensusSummaryStats(self.data_path),
                'viz': ConsensusVisualizations(self.data_path),
                'data_manager': data_manager
            }

        except Exception as e:
            logger.error("Error initializing consensus components: %s", e)
            self.components = {}

    def get_available_samples(self) -> List[str]:
        """Get list of available samples."""
        if 'consensus' in self.components:
            try:
                return self.components['consensus']['data_manager'].get_available_samples()
            except Exception as e:
                logger.warning("Error getting samples: %s", e)
        return []

    def get_summary_stats(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get summary statistics for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to analyze

        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis',
            'description': 'Analysis of genome recovery statistics and nucleotide identity comparisons',
            'sections': []
        }

        if 'consensus' not in self.components:
            return summary

        try:
            stats = self.components['consensus']['stats'].calculate_genome_recovery_stats(sample_ids)

            if stats:
                # Genome recovery metrics
                recovery_data = stats.get('genome_recovery', {})
                if recovery_data:
                    summary['sections'].append({
                        'title': 'Genome Recovery Statistics',
                        'type': 'metrics',
                        'data': {
                            'Total Samples': recovery_data.get('total_samples', 0),
                            'Average Recovery': f"{recovery_data.get('mean_recovery_pct', 0):.1f}%",
                            'Median Recovery': f"{recovery_data.get('median_recovery_pct', 0):.1f}%",
                            'High Quality Genomes (>80%)': recovery_data.get('high_quality_count', 0)
                        }
                    })

                # Species breakdown
                species_data = stats.get('species_breakdown', {})
                if species_data:
                    summary['sections'].append({
                        'title': 'Species Recovery Breakdown',
                        'type': 'species_recovery',
                        'data': species_data
                    })

                # ANI comparison if available
                ani_data = stats.get('ani_comparison', {})
                if ani_data:
                    summary['sections'].append({
                        'title': 'Average Nucleotide Identity (ANI)',
                        'type': 'ani_metrics',
                        'data': {
                            'Average ANI': f"{ani_data.get('mean_ani', 0):.2f}%",
                            'Median ANI': f"{ani_data.get('median_ani', 0):.2f}%",
                            'High Identity (>95%)': ani_data.get('high_identity_count', 0)
                        }
                    })

        except Exception as e:
            logger.error("Error generating consensus stats: %s", e)

        return summary

    def get_visualizations(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get visualizations for consensus analysis.

        Args:
            sample_ids: Optional list of sample IDs to visualize

        Returns:
            Dictionary containing plotly figures
        """
        figures = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis Visualizations',
            'figures': []
        }

        if 'consensus' not in self.components:
            return figures

        try:
            viz_component = self.components['consensus']['viz']
            all_figures = viz_component.create_all_visualizations(sample_ids)

            if all_figures:
                for title, fig in all_figures.items():
                    if fig and fig.data:
                        figures['figures'].append({
                            'title': title,
                            'description': f'Consensus analysis: {title.lower()}',
                            'figure': fig,
                            'type': 'plotly'
                        })

        except Exception as e:
            logger.error("Error generating consensus visualizations: %s", e)

        return figures

    def get_custom_html(self, sample_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get custom HTML components for consensus analysis.
        This is where you can integrate your mini HTML package.

        Args:
            sample_ids: Optional list of sample IDs

        Returns:
            Dictionary containing custom HTML components
        """
        html_components = {
            'component_type': 'consensus',
            'title': 'Consensus Analysis - Custom Views',
            'components': []
        }

        if 'consensus' not in self.components:
            return html_components

        try:
            # Example custom HTML component
            # Replace this with your mini HTML package calls
            stats = self.components['consensus']['stats'].calculate_genome_recovery_stats(sample_ids)

            if stats:
                # Custom genome recovery HTML view
                recovery_html = self._create_genome_recovery_html(stats.get('genome_recovery', {}))
                if recovery_html:
                    html_components['components'].append({
                        'title': 'Genome Recovery Dashboard',
                        'description': 'Custom HTML view of genome recovery statistics',
                        'html': recovery_html,
                        'type': 'custom_html'
                    })

                # Custom species comparison HTML
                species_html = self._create_species_comparison_html(stats.get('species_breakdown', {}))
                if species_html:
                    html_components['components'].append({
                        'title': 'Species Comparison Matrix',
                        'description': 'Interactive species comparison view',
                        'html': species_html,
                        'type': 'custom_html'
                    })

        except Exception as e:
            logger.error("Error generating custom HTML: %s", e)

        return html_components

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
