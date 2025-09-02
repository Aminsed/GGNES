# GGNES Makefile for common development tasks

.PHONY: test coverage lint clean install

# Install dependencies
install:
	pip install -r requirements.txt

# Run tests
test:
	pytest tests/ -v

# Run tests with coverage
coverage:
	pytest tests/ --cov=ggnes --cov-report=term-missing --cov-report=html

# Run linting
lint:
	flake8 ggnes/ tests/
	ruff check ggnes/ tests/

# Clean up
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/

# Run all checks (CI pipeline simulation)
ci: lint test coverage
	@echo "All CI checks passed!"
