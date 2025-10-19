.PHONY: install dev-install clean test lint format check help

# Default target
.DEFAULT_GOAL := help

help: ## Show this help message
	@echo 'Vanish - Audio Noise Removal System'
	@echo ''
	@echo 'Usage:'
	@echo '  make <target>'
	@echo ''
	@echo 'Targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install: ## Install package and dependencies
	pip install -r requirements.txt
	pip install -e .

dev-install: ## Install package with development dependencies
	pip install -r requirements.txt
	pip install -e ".[dev]"

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete

test: ## Run tests with coverage
	pytest tests/ -v --cov=vanish --cov-report=term-missing

test-quick: ## Run tests without coverage
	pytest tests/ -v

lint: ## Run linting checks
	flake8 src/
	mypy src/

format: ## Format code with black
	black src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/

check: format-check lint test ## Run all checks (format, lint, test)

info: ## Show system information
	python -c "import sys; print(f'Python: {sys.version}')"
	python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

example: ## Run basic example
	python examples/basic_usage.py

config: ## Generate default configuration file
	python -c "from vanish.config import PipelineConfig; PipelineConfig().to_yaml('config.yaml')"
	@echo "Configuration file created: config.yaml"

build: ## Build distribution packages
	python -m build

upload-test: build ## Upload to TestPyPI
	python -m twine upload --repository testpypi dist/*

upload: build ## Upload to PyPI
	python -m twine upload dist/*

docs: ## Generate documentation
	@echo "Documentation generation not implemented yet"

.PHONY: install-resemble
install-resemble: ## Install Resemble-Enhance (optional)
	pip install git+https://github.com/resemble-ai/resemble-enhance.git

.PHONY: install-cuda
install-cuda: ## Install PyTorch with CUDA support
	pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
