PYTHON ?= .venv/Scripts/python

.PHONY: install test test-aws lint format manifest

install:
	python -m venv .venv
	$(PYTHON) -m pip install -e ".[jobs,models,dictionary,backend,mcp,dev]"

test:
	$(PYTHON) -m pytest

# Opt-in: needs real AWS credentials and the configured bucket.
test-aws:
	ML_FACTORY_AWS_TESTS=1 $(PYTHON) -m pytest -m aws

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format:
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

# Re-record inherited-file hashes after a reviewed, deliberate change to an inherited file.
manifest:
	$(PYTHON) scripts/inherited_manifest.py --write
