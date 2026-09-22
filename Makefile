.PHONY: sync help

help:
	@echo "Available commands:"
	@echo "  make sync    Sync fork (origin/main) and local main with upstream/main"

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
