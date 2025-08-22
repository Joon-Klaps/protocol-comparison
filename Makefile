# Makefile for Viral Genomics Protocol Comparison Dashboard (Dash Application)

.PHONY: help install setup test-data clean run dev test clean-logs

# Default target
help:
	@echo "🧬 Viral Genomics Protocol Comparison Dashboard"
	@echo "=============================================="
	@echo "Pure Dash Application for Bioinformatics Analysis"
	@echo ""
	@echo "Available commands:"
	@echo "  install       - Install Python dependencies"
	@echo "  setup         - Full setup (install + validate data)"
	@echo "  run           - Run the Dash application (default)"
	@echo "  dev           - Run in development mode with debug"
	@echo "  test          - Test Dash components and modules"
	@echo "  clean         - Clean up generated files and caches"
	@echo "  clean-logs    - Clean up log files only"
	@echo "  help          - Show this help message"
	@echo ""
	@echo "🚀 Quick start: make setup && make run"
	@echo "🌐 Access at: http://localhost:8050"

# Install dependencies
install:
	@echo "📦 Installing Dash application dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed successfully!"

# Full setup
setup: install
	@echo "🔧 Setting up Dash application..."
	@echo "� Validating data paths..."
	python -c "from pathlib import Path; print('✅ Data validation complete!' if Path('../../data/app').exists() else '⚠️  Data path not found - update dash_app.py with correct path')"
	@echo "🎉 Setup complete! Run 'make run' to start the application."

# Run Dash application (default)
run:
	@echo "🚀 Starting Dash application..."
	@echo "🌐 Access the dashboard at: http://localhost:8050"
	@echo "� Press Ctrl+C to stop the server"
	python dash_app.py

# Run in development mode
dev:
	@echo "🛠️  Starting Dash application in development mode..."
	@echo "� Auto-reload enabled"
	@echo "🌐 Access the dashboard at: http://localhost:8050"
	python dash_app.py --debug

# Test Dash components
test:
	@echo "🧪 Testing Dash components and modules..."
	python test_simple.py
	@echo "✅ All tests completed!"

# Clean up generated files and caches
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf __pycache__/ modules/__pycache__/ modules/*/__pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Clean up log files only
clean-logs:
	@echo "🧹 Cleaning up log files..."
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Log cleanup complete!"

# Show system info
info:
	@echo "📊 System Information:"
	@echo "====================="
	@python --version
	@pip show dash 2>/dev/null | grep Version || echo "Dash: Not installed"
	@pip show plotly 2>/dev/null | grep Version || echo "Plotly: Not installed"
	@pip show dash-bio 2>/dev/null | grep Version || echo "Dash-Bio: Not installed"
	@echo ""
	@echo "📁 Project Structure:"
	@echo "===================="
	@echo "Main app:       dash_app.py"
	@echo "Modules:        modules/"
	@echo "Data path:      ../../data/app/"
	@echo "Requirements:   requirements.txt"
