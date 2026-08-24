.DEFAULT_GOAL := help
PYTHON ?= python3
VENV := .venv-dev

.PHONY: help venv test lint check agent-check build up down logs clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create the development virtualenv and install both components
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r client/requirements.txt
	$(VENV)/bin/pip install -r dashboard/backend/requirements.txt
	$(VENV)/bin/pip install -r requirements-dev.txt

test: ## Run every test (client, dashboard, and shared code)
	$(VENV)/bin/pytest

lint: ## Check formatting and import hygiene
	$(VENV)/bin/ruff check .

agent-check: ## Verify the student SP-Agent loads and returns a valid placement
	$(VENV)/bin/python scripts/check_sp_agent.py

check: lint test agent-check ## Everything CI runs

build: ## Build the container images
	docker compose build

up: ## Start client, dashboard backend, and dashboard frontend
	docker compose up -d --build

down: ## Stop everything
	docker compose down

logs: ## Follow the client logs
	docker compose logs -f client

clean: ## Remove caches and build artifacts
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
