# Makefile for Viral Genomics Protocol Comparison Dashboard

.PHONY: help install setup test-data clean run run-modular run-legacy dev clean-logs

# Default target
help:
	@echo "Viral Genomics Protocol Comparison Dashboard"
	@echo "============================================"
	@echo ""
	@echo "Available commands:"
	@echo "  install       - Install Python dependencies"
	@echo "  setup         - Full setup (install + generate test data)"
	@echo "  test-data     - Generate sample data for testing"
	@echo "  run           - Run the modular Streamlit dashboard (recommended)"
	@echo "  run-modular   - Run the new modular Streamlit dashboard"
	@echo "  run-legacy    - Run the legacy monolithic Streamlit dashboard"
	@echo "  dev           - Run modular dashboard in development mode"
	@echo "  clean         - Clean up generated files and caches"
	@echo "  clean-logs    - Clean up log files only"
	@echo "  help          - Show this help message"

# Install dependencies
install:
	pip install -r requirements.txt

# Full setup
setup:
	python setup.py --full-setup

# Generate test data
test-data:
	python generate_sample_data.py --num-samples 10

# Run modular Streamlit dashboard with sample data (default and recommended)
run: run-modular

# Run new modular Streamlit dashboard with sample data
run-modular:
	@echo "🚀 Starting modular dashboard..."
	streamlit run modular_streamlit_app.py

# Run modular dashboard with specific data path
run-modular-with-data:
	@echo "🚀 Starting modular dashboard with sample data..."
	DEFAULT_DATA_PATH=sample_data streamlit run modular_streamlit_app.py

# Run legacy monolithic Streamlit dashboard (for compatibility)
run-legacy:
	@echo "⚠️  Starting legacy dashboard..."
	python run_streamlit.py --data-path sample_data

# Run modular dashboard in development mode
dev:
	@echo "🛠️  Starting modular dashboard in development mode..."
	DEFAULT_DATA_PATH=sample_data streamlit run modular_streamlit_app.py --server.address 0.0.0.0

# Clean up generated files and caches
clean:
	@echo "🧹 Cleaning up generated files..."
	rm -rf sample_data/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Cleanup complete!"

# Clean up log files only
clean-logs:
	@echo "🧹 Cleaning up log files..."
	find . -name "*.log" -delete 2>/dev/null || true
	@echo "✅ Log cleanup complete!"

# Quick start for new users (setup + run modular)
quickstart: setup run-modular

# Test the modular system
test-modular: test-data run-modular

# Show system info
info:
	@echo "📊 System Information:"
	@echo "====================="
	@python --version
	@pip show streamlit 2>/dev/null | grep Version || echo "Streamlit: Not installed"
	@pip show plotly 2>/dev/null | grep Version || echo "Plotly: Not installed"
	@echo ""
	@echo "📁 Project Structure:"
	@echo "===================="
	@echo "Modular app:     modular_streamlit_app.py"
	@echo "Legacy app:      streamlit_app.py"
	@echo "Launcher:        run_streamlit.py"
	@echo "Modules:         modules/"
	@echo "Sample data:     sample_data/"
