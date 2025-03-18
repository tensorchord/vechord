PY_SOURCE=.

lint:
	@uv run -- ruff check ${PY_SOURCE}

typecheck:
	@uv run -- mypy --non-interactive --install-types ${PY_SOURCE}

format:
	@uv run -- ruff check --fix ${PY_SOURCE}
	@uv run -- ruff format ${PY_SOURCE}

clean:
	@-rm -rf dist build */__pycache__ *.egg-info vechord/__version__.py

build:
	@uv build

publish: build
	@uv publish

test:
	@uv run -- pytest -v tests

sync:
	@uv sync --all-extras --all-groups

doc:
	@cd docs && make html && cd ..
	@uv run -m http.server -d docs/build/html -b 127.0.0.1 8000

.PHONY: lint format test doc
