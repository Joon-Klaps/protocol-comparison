# Makefile for Viral Genomics Protocol Comparison Dashboard

.PHONY: help install setup test-data clean run-dash run-streamlit run dev

# Default target
help:
	@echo "Viral Genomics Protocol Comparison Dashboard"
	@echo "============================================"
	@echo ""
	@echo "Available commands:"
	@echo "  install       - Install Python dependencies"
	@echo "  setup         - Full setup (install + generate test data)"
	@echo "  test-data     - Generate sample data for testing"
	@echo "  run           - Run the Streamlit dashboard with sample data"
	@echo "  run-streamlit - Run Streamlit dashboard with sample data"
	@echo "  run-dash      - Run original Dash dashboard with sample data"
	@echo "  dev           - Run Streamlit dashboard in development mode"
	@echo "  clean         - Clean up generated files"
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

# Run Streamlit dashboard with sample data (default)
run: run-streamlit

# Run Streamlit dashboard with sample data
run-streamlit:
	python run_streamlit.py --data-path sample_data

# Run original Dash dashboard with sample data
run-dash:
	python run_dashboard.py --data-path sample_data

# Run Streamlit in development mode
dev:
	python run_streamlit.py --data-path sample_data --host 0.0.0.0

# Clean up generated files
clean:
	rm -rf sample_data/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	find . -name "*.log" -delete

# Quick start for new users
quickstart: setup run
