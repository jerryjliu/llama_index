pyproject_str = """[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"

[tool.codespell]
check-filenames = true
check-hidden = true
# Feel free to un-skip examples, and experimental, you will just need to
# work through many typos (--write-changes and --interactive will help)
skip = "*.csv,*.html,*.json,*.jsonl,*.pdf,*.txt,*.ipynb"

[tool.mypy]
disallow_untyped_defs = true
# Remove venv skip when integrated with pre-commit
exclude = ["_static", "build", "examples", "notebooks", "venv"]
ignore_missing_imports = true
python_version = "3.8"

[tool.poetry]
name = "{PACKAGE_NAME}"
version = "0.1.0"
description = "llama-index {TYPE} {NAME} integration"
authors = ["Your Name <you@example.com>"]
license = "MIT"
readme = "README.md"
packages = [{{include = "llama_index/"}}]

[tool.poetry.dependencies]
python = ">=3.8.1,<3.12"
llama-index-core = "^0.10.0"

[tool.poetry.group.dev.dependencies]
black = {{extras = ["jupyter"], version = "<=23.9.1,>=23.7.0"}}
codespell = {{extras = ["toml"], version = ">=v2.2.6"}}
ipython = "8.10.0"
jupyter = "^1.0.0"
mypy = "0.991"
pre-commit = "3.2.0"
pylint = "2.15.10"
pytest = "7.2.1"
pytest-mock = "3.11.1"
ruff = "0.0.292"
tree-sitter-languages = "^1.8.0"
types-Deprecated = ">=0.1.0"
types-PyYAML = "^6.0.12.12"
types-protobuf = "^4.24.0.4"
types-redis = "4.5.5.0"
types-requests = "2.28.11.8" # TODO: unpin when mypy>0.991
types-setuptools = "67.1.0.0"
"""
