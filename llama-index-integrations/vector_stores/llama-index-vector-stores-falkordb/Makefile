.PHONY: all format lint test clean

all: format lint test

format:
    black .
    isort .

lint:
    flake8 .
    mypy .

test:
    pytest tests/

clean:
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    rm -rf .pytest_cache
    rm -rf .mypy_cache
