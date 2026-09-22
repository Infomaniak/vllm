.PHONY: sync dev help

VENV ?= .venv
PYTHON ?= $(VENV)/bin/python

help:
	@echo "Available commands:"
	@echo "  make sync    Sync fork (origin/main) and local main with upstream/main"
	@echo "  make dev     Setup/sync local dev environment (uv, precompiled vLLM, pre-commit)"

sync:
	@git remote get-url upstream >/dev/null 2>&1 || { echo "Error: 'upstream' remote not found."; exit 1; }
	@git remote get-url origin >/dev/null 2>&1 || { echo "Error: 'origin' remote not found."; exit 1; }
	@echo "==> Fetching from upstream/main..."
	git fetch upstream main --tags
	@echo "==> Pushing to origin/main..."
	git push origin refs/remotes/upstream/main:refs/heads/main
	@CURRENT_BRANCH=$$(git rev-parse --abbrev-ref HEAD); \
	if [ "$$CURRENT_BRANCH" = "main" ]; then \
		echo "==> Fast-forwarding local main..."; \
		git merge --ff-only upstream/main; \
	else \
		echo "==> Fast-forwarding local main reference..."; \
		git fetch . refs/remotes/upstream/main:refs/heads/main 2>/dev/null || true; \
	fi
	@echo "==> Successfully synced with upstream!"

dev:
	@which uv >/dev/null 2>&1 || { echo "Error: 'uv' is not installed. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
	@if [ ! -d "$(VENV)" ]; then \
		echo "==> Creating virtual environment with Python 3.12..."; \
		uv venv $(VENV) --python 3.12; \
	fi
	@echo "==> Installing lint requirements & pre-commit hooks..."
	@VIRTUAL_ENV=$(CURDIR)/$(VENV) uv pip install -r requirements/lint.txt
	@$(VENV)/bin/pre-commit install
	@echo "==> Resolving latest available precompiled wheel commit..."
	@PRECOMPILED_COMMIT=$$(python3 tools/get_precompiled_commit.py 2>/dev/null); \
	if [ -n "$$PRECOMPILED_COMMIT" ]; then \
		echo "==> Using precompiled wheel from commit: $$PRECOMPILED_COMMIT"; \
		VIRTUAL_ENV=$(CURDIR)/$(VENV) VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$$PRECOMPILED_COMMIT uv pip install -e . --torch-backend=auto; \
	else \
		echo "==> No recent built wheel commit found, attempting default precompiled setup..."; \
		VIRTUAL_ENV=$(CURDIR)/$(VENV) VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto; \
	fi
	@echo "==> Dev environment is ready! Activate with: source $(VENV)/bin/activate"
