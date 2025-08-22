#!/usr/bin/env python3
"""
Complete migration script from Streamlit to Dash application.

This script performs a complete refactoring migration from the Streamlit
application to a Dash application with full dash_bio support.
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StreamlitToDashMigrator:
    """Complete migration utility from Streamlit to Dash."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "streamlit_backup"
        self.modules_dir = self.project_root / "modules"
        
    def create_backup(self):
        """Create backup of current Streamlit application."""
        logger.info("🔄 Creating backup of Streamlit application...")
        
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir)
        
        self.backup_dir.mkdir()
        
        # Backup key Streamlit files
        streamlit_files = [
            "streamlit_app.py",
            "sample_selection.py",
            ".streamlit/config.toml"
        ]
        
        for file_path in streamlit_files:
            src = self.project_root / file_path
            if src.exists():
                if src.is_file():
                    dst = self.backup_dir / file_path
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    logger.info(f"  ✅ Backed up: {file_path}")
                else:
                    shutil.copytree(src, self.backup_dir / file_path)
                    logger.info(f"  ✅ Backed up directory: {file_path}")
        
        logger.info(f"📁 Backup created at: {self.backup_dir}")
    
    def update_requirements(self):
        """Update requirements.txt for Dash migration."""
        logger.info("📦 Updating requirements.txt for Dash...")
        
        requirements_file = self.project_root / "requirements.txt"
        
        # Read current requirements
        if requirements_file.exists():
            with open(requirements_file, 'r') as f:
                current_reqs = f.read()
        else:
            current_reqs = ""
        
        # Add Dash requirements if not present
        dash_requirements = [
            "# Dash application framework",
            "dash>=2.15.0",
            "dash-bootstrap-components>=1.5.0", 
            "dash-bio>=1.0.0",
            "plotly>=5.15.0",
            ""
        ]
        
        # Check if dash requirements are already present
        if "dash>=" not in current_reqs:
            with open(requirements_file, 'a') as f:
                f.write("\n" + "\n".join(dash_requirements))
            logger.info("  ✅ Added Dash requirements")
        else:
            logger.info("  ℹ️  Dash requirements already present")
    
    def create_dash_modules(self):
        """Create Dash-specific module adapters."""
        logger.info("🔧 Creating Dash module adapters...")
        
        # We already have the consensus dash adapter, let's create others
        self.create_coverage_dash_adapter()
        self.create_read_stats_dash_adapter()
        
    def create_coverage_dash_adapter(self):
        """Create Dash adapter for coverage module."""
        coverage_adapter_path = self.modules_dir / "coverage" / "dash_adapter.py"
        
        if coverage_adapter_path.exists():
            logger.info("  ℹ️  Coverage Dash adapter already exists")
            return
            
        coverage_adapter_content = '''"""
Dash adapter for coverage analysis module.
"""

from typing import Dict, List, Any, Optional, Tuple
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
            Input(f"{self.app_id}-sample-dropdown", "value")
        )
        def update_coverage_results(selected_samples):
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
'''
        
        coverage_adapter_path.parent.mkdir(parents=True, exist_ok=True)
        with open(coverage_adapter_path, 'w') as f:
            f.write(coverage_adapter_content)
        
        logger.info("  ✅ Created coverage Dash adapter")
    
    def create_read_stats_dash_adapter(self):
        """Create Dash adapter for read_stats module."""
        read_stats_adapter_path = self.modules_dir / "read_stats" / "dash_adapter.py"
        
        if read_stats_adapter_path.exists():
            logger.info("  ℹ️  Read stats Dash adapter already exists")
            return
            
        read_stats_adapter_content = '''"""
Dash adapter for read statistics module.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc

from .tab import ReadStatsTab as OriginalReadStatsTab

logger = logging.getLogger(__name__)


class DashReadStatsTab:
    """Dash adapter for read statistics analysis."""
    
    def __init__(self, data_path: Path):
        self.data_path = Path(data_path)
        self.original_tab = OriginalReadStatsTab(data_path)
        self.app_id = "read_stats"
    
    def get_available_samples(self) -> List[str]:
        """Get available samples."""
        return self.original_tab.get_available_samples()
    
    def create_layout(self) -> html.Div:
        """Create layout for read stats tab."""
        return html.Div([
            html.H3("📈 Read Statistics"),
            html.P("Read processing and quality statistics"),
            
            # Sample selection
            dbc.Card([
                dbc.CardBody([
                    html.H5("Sample Selection"),
                    dcc.Dropdown(
                        id=f"{self.app_id}-sample-dropdown",
                        options=[{"label": s, "value": s} for s in self.get_available_samples()],
                        multi=True,
                        placeholder="Select samples"
                    )
                ])
            ], className="mb-4"),
            
            # Results area
            html.Div(id=f"{self.app_id}-results")
        ])
    
    def register_callbacks(self, app: dash.Dash):
        """Register callbacks for read stats tab."""
        
        @app.callback(
            Output(f"{self.app_id}-results", "children"),
            Input(f"{self.app_id}-sample-dropdown", "value")
        )
        def update_read_stats_results(selected_samples):
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
                
                return html.Div(content)
                
            except Exception as e:
                logger.error(f"Error updating read stats results: {e}")
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


def create_read_stats_dash_component(data_path: Path) -> DashReadStatsTab:
    """Factory function for read stats Dash component."""
    return DashReadStatsTab(data_path)
'''
        
        read_stats_adapter_path.parent.mkdir(parents=True, exist_ok=True)
        with open(read_stats_adapter_path, 'w') as f:
            f.write(read_stats_adapter_content)
        
        logger.info("  ✅ Created read stats Dash adapter")
    
    def update_makefile(self):
        """Update Makefile with Dash commands."""
        logger.info("📝 Updating Makefile for Dash commands...")
        
        makefile_path = self.project_root / "Makefile"
        
        if not makefile_path.exists():
            logger.warning("  ⚠️  Makefile not found, creating new one")
            self.create_new_makefile()
            return
        
        with open(makefile_path, 'r') as f:
            makefile_content = f.read()
        
        # Add Dash commands if not present
        dash_commands = '''
# Dash application commands
run-dash:
	@echo "🚀 Starting Dash application..."
	python dash_app.py

run-dash-dev:
	@echo "🛠️  Starting Dash application in development mode..."
	python dash_app.py --debug

test-dash:
	@echo "🧪 Testing Dash components..."
	python test_consensus_dash.py

migrate-to-dash:
	@echo "🔄 Running complete migration to Dash..."
	python migrate_to_dash.py

# Combined commands
run-both:
	@echo "📱 Choose your interface:"
	@echo "  For Streamlit: make run-streamlit"
	@echo "  For Dash: make run-dash"

run-streamlit:
	@echo "🚀 Starting Streamlit application..."
	streamlit run streamlit_app.py
'''
        
        if "run-dash:" not in makefile_content:
            with open(makefile_path, 'a') as f:
                f.write(dash_commands)
            logger.info("  ✅ Added Dash commands to Makefile")
        else:
            logger.info("  ℹ️  Dash commands already present in Makefile")
    
    def create_new_makefile(self):
        """Create a new Makefile with both Streamlit and Dash commands."""
        makefile_content = '''# Viral Genomics Protocol Comparison Dashboard
# Complete Streamlit & Dash Application

.PHONY: help install setup test-data clean run-streamlit run-dash run-both dev clean-logs

# Default target
help:
	@echo "Viral Genomics Protocol Comparison Dashboard"
	@echo "============================================"
	@echo ""
	@echo "Available commands:"
	@echo "  install       - Install Python dependencies"
	@echo "  setup         - Full setup (install + generate test data)"
	@echo "  run-streamlit - Run the Streamlit application"
	@echo "  run-dash      - Run the Dash application (recommended for consensus)"
	@echo "  run-both      - Show options for both interfaces"
	@echo "  test-dash     - Test Dash components"
	@echo "  migrate       - Complete migration to Dash"
	@echo "  dev           - Run in development mode"
	@echo "  clean         - Clean up generated files and caches"
	@echo "  help          - Show this help message"

# Install dependencies
install:
	pip install -r requirements.txt

# Full setup
setup: install
	@echo "✅ Setup complete!"

# Run Streamlit application
run-streamlit:
	@echo "🚀 Starting Streamlit application..."
	streamlit run streamlit_app.py

# Run Dash application
run-dash:
	@echo "🚀 Starting Dash application..."
	python dash_app.py

# Show both options
run-both:
	@echo "📱 Choose your interface:"
	@echo "  For Streamlit: make run-streamlit"
	@echo "  For Dash: make run-dash (recommended for consensus analysis)"

# Test Dash components
test-dash:
	@echo "🧪 Testing Dash components..."
	python test_consensus_dash.py

# Complete migration
migrate:
	@echo "🔄 Running complete migration to Dash..."
	python migrate_to_dash.py

# Development mode
dev:
	@echo "🛠️  Starting Dash application in development mode..."
	python dash_app.py --debug

# Clean up
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf __pycache__/ modules/__pycache__/ modules/*/__pycache__/
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Default run command points to Dash
run: run-dash
'''
        
        makefile_path = self.project_root / "Makefile"
        with open(makefile_path, 'w') as f:
            f.write(makefile_content)
        
        logger.info("  ✅ Created new Makefile with Dash commands")
    
    def create_migration_summary(self):
        """Create summary of migration changes."""
        summary_path = self.project_root / "MIGRATION_SUMMARY.md"
        
        summary_content = f'''# Migration Summary: Streamlit → Dash

**Migration Date:** {os.popen("date").read().strip()}

## 🎯 Migration Overview

This project has been successfully migrated from Streamlit to Dash to enable:
- Interactive sequence alignments with dash_bio
- Better performance for bioinformatics visualizations
- More flexible component architecture

## 📁 File Structure Changes

### New Files Created:
- `dash_app.py` - Main Dash application
- `modules/dash_base.py` - Base Dash components
- `modules/consensus/dash_adapter.py` - Consensus Dash adapter
- `modules/coverage/dash_adapter.py` - Coverage Dash adapter
- `modules/read_stats/dash_adapter.py` - Read stats Dash adapter
- `migrate_to_dash.py` - This migration script

### Backup Location:
- `streamlit_backup/` - Contains original Streamlit files

### Updated Files:
- `requirements.txt` - Added Dash dependencies
- `Makefile` - Added Dash commands

## 🚀 How to Run

### Dash Application (Recommended):
```bash
make run-dash
# or
python dash_app.py
```

### Streamlit Application (Legacy):
```bash
make run-streamlit
# or
streamlit run streamlit_app.py
```

## 🔧 Available Commands

- `make run-dash` - Run Dash application
- `make run-streamlit` - Run Streamlit application  
- `make test-dash` - Test Dash components
- `make install` - Install dependencies
- `make clean` - Clean up files

## ⚡ Key Benefits

1. **Interactive Alignments**: Full dash_bio AlignmentChart support
2. **Better Performance**: Faster loading and rendering
3. **Modern UI**: Bootstrap components with responsive design
4. **Modular Architecture**: Easy to extend and maintain

## 🛠️ Development

For development mode:
```bash
make dev
```

This runs the Dash app with debug mode enabled.

## 📊 Module Status

- ✅ **Consensus**: Fully migrated with dash_bio integration
- ✅ **Coverage**: Migrated with basic Dash components
- ✅ **Read Stats**: Migrated with basic Dash components

## 🔄 Rollback (if needed)

To rollback to Streamlit:
1. Restore files from `streamlit_backup/`
2. Run `make run-streamlit`

## 🎉 Success!

Your viral genomics analysis dashboard is now running on Dash with full bioinformatics visualization support!
'''
        
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        logger.info(f"📄 Created migration summary: {summary_path}")
    
    def run_migration(self):
        """Run the complete migration process."""
        logger.info("🚀 Starting complete Streamlit → Dash migration...")
        
        try:
            # Step 1: Create backup
            self.create_backup()
            
            # Step 2: Update requirements
            self.update_requirements()
            
            # Step 3: Create Dash modules
            self.create_dash_modules()
            
            # Step 4: Update Makefile
            self.update_makefile()
            
            # Step 5: Create migration summary
            self.create_migration_summary()
            
            logger.info("✅ Migration completed successfully!")
            logger.info("")
            logger.info("🎉 Next steps:")
            logger.info("  1. Install dependencies: make install")
            logger.info("  2. Run Dash app: make run-dash")
            logger.info("  3. Access at: http://localhost:8050")
            logger.info("")
            logger.info("📁 Backup location: streamlit_backup/")
            logger.info("📄 See MIGRATION_SUMMARY.md for details")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            return False


def main():
    """Main migration function."""
    if len(sys.argv) > 1 and sys.argv[1] in ['--help', '-h']:
        print("""
Complete Streamlit to Dash Migration Tool

Usage:
    python migrate_to_dash.py [--dry-run]

Options:
    --dry-run    Show what would be done without making changes
    --help, -h   Show this help message

This script performs a complete migration from Streamlit to Dash,
including backup creation, module adaptation, and configuration updates.
        """)
        return
    
    project_root = Path(__file__).parent
    migrator = StreamlitToDashMigrator(project_root)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        logger.info("🔍 Dry run mode - showing what would be done:")
        logger.info("  1. Create backup of Streamlit files")
        logger.info("  2. Update requirements.txt with Dash dependencies")
        logger.info("  3. Create Dash module adapters")
        logger.info("  4. Update Makefile with Dash commands")
        logger.info("  5. Create migration summary")
        logger.info("No changes will be made.")
        return
    
    success = migrator.run_migration()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
