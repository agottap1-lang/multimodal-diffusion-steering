# Multimodal Diffusion Policy - Quick Commands
# Usage: make <target>

.PHONY: help reorganize eval videos verify debug clean test

# Default checkpoint
CHECKPOINT := runs/diffusion_20260222_195530/ckpt_ep100.pt

help:
	@echo "🚀 Multimodal Diffusion Policy - Available Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make reorganize    - Reorganize project structure"
	@echo "  make install       - Install dependencies"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval          - Run paired evaluation (10 episodes)"
	@echo "  make eval-full     - Run full evaluation (50 episodes)"
	@echo "  make quick         - Quick test (3 episodes)"
	@echo ""
	@echo "Analysis:"
	@echo "  make videos        - Generate arc 15-19 videos"
	@echo "  make verify        - Verify arc diversity"
	@echo "  make debug         - Debug VLM selection"
	@echo ""
	@echo "Testing:"
	@echo "  make test          - Run all tests"
	@echo "  make test-vlm      - Test VLM integration"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean         - Clean outputs and cache"
	@echo ""

reorganize:
	@echo "📦 Reorganizing project structure..."
	@pwsh -ExecutionPolicy Bypass -File reorganize.ps1

install:
	@echo "📥 Installing dependencies..."
	@pip install -r requirements.txt

eval:
	@echo "🔬 Running paired evaluation (10 episodes)..."
	@python cli.py evaluate-paired --episodes 10

eval-full:
	@echo "🔬 Running full evaluation (50 episodes)..."
	@python cli.py evaluate-paired --episodes 50

quick:
	@echo "⚡ Running quick evaluation..."
	@python cli.py quick-eval --episodes 3

videos:
	@echo "🎥 Generating arc 15-19 videos..."
	@python cli.py generate-videos --n-videos 10

verify:
	@echo "🔍 Verifying arc diversity..."
	@python cli.py verify-arc --samples 100

verify-videos:
	@echo "🔍 Verifying arc with videos..."
	@python cli.py verify-arc --with-videos

debug:
	@echo "🐛 Debugging VLM selection..."
	@python cli.py debug-vlm --episode 42

test:
	@echo "🧪 Running tests..."
	@python -m pytest experiments/ -v

test-vlm:
	@echo "🧪 Testing VLM integration..."
	@python experiments/test_vlm_integration.py

clean:
	@echo "🧹 Cleaning up..."
	@rm -rf __pycache__/ */__pycache__/ *.pyc */*.pyc
	@rm -rf .pytest_cache/ */.pytest_cache/
	@echo "✓ Cleanup complete"

lint:
	@echo "✨ Running linter..."
	@ruff check . --fix

format:
	@echo "💅 Formatting code..."
	@black .

# Windows-specific targets
.PHONY: reorganize-win install-win clean-win

reorganize-win:
	@powershell -ExecutionPolicy Bypass -File reorganize.ps1

install-win:
	@python -m pip install -r requirements.txt

clean-win:
	@if exist __pycache__ rmdir /s /q __pycache__
	@if exist .pytest_cache rmdir /s /q .pytest_cache
	@del /s /q *.pyc 2>nul
	@echo Cleanup complete
