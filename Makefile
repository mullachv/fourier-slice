# Fourier Slice — rapid reproducible setup and run
# Recommended: uv (pip install uv  or  brew install uv). Fallback: make install-pip

.PHONY: venv venv-recreate install sync test test-cov run lock clean help install-pip test-pip run-pip
.PHONY: docker-build docker-run docker-build-multi

# Default: show help
help:
	@echo "fourier-slice — targets:"
	@echo "  make install   — ensure .venv + install deps from uv.lock (no replace prompt)"
	@echo "  make sync      — same as install"
	@echo "  make venv-recreate — remove .venv and re-sync (full consistency)"
	@echo "  make test      — run pytest"
	@echo "  make test-cov  — run pytest with coverage"
	@echo "  make run       — run Streamlit app"
	@echo "  make lock      — refresh uv.lock from pyproject.toml"
	@echo "  make clean     — remove .venv and cache"
	@echo "  --- Docker ---"
	@echo "  make docker-build       — build image (current arch)"
	@echo "  make docker-build-multi — build for amd64 + arm64"
	@echo "  make docker-run         — run app in container (port 8501)"
	@echo "  --- pip fallback (no uv): ---"
	@echo "  make install-pip  — venv + pip install"
	@echo "  make test-pip     — run pytest (after install-pip)"
	@echo "  make run-pip      — streamlit run app.py (after install-pip)"

# uv sync creates .venv if missing; does not replace existing (no prompt)
venv:
	uv venv

install sync:
	uv sync

# Force a fresh env for strict reproducibility (e.g. after lock change)
venv-recreate: clean
	uv sync

# Run tests
test: install
	uv run pytest

test-cov: install
	uv run pytest --cov=. --cov-report=term-missing

# Run the Streamlit app
run: install
	uv run streamlit run app.py

# Regenerate lockfile (e.g. after editing pyproject.toml)
lock:
	uv lock

# Pip fallback (no uv): create venv and install from requirements.txt
install-pip:
	python3 -m venv .venv
	.venv/bin/pip install -q --upgrade pip
	.venv/bin/pip install -q -r requirements.txt pytest pytest-cov

test-pip: install-pip
	.venv/bin/pytest

run-pip: install-pip
	.venv/bin/streamlit run app.py

# Docker: build image (current platform)
DOCKER_IMAGE ?= fourier-slice
docker-build:
	docker build -t $(DOCKER_IMAGE):latest .

# Docker: run Streamlit (port 8501)
docker-run:
	docker run --rm -p 8501:8501 $(DOCKER_IMAGE):latest

# Docker: multi-arch (amd64 + arm64). Set DOCKER_REGISTRY=user to push (e.g. docker.io/myuser)
docker-build-multi:
	@if [ -n "$(DOCKER_REGISTRY)" ]; then \
		docker buildx build --platform linux/amd64,linux/arm64 -t $(DOCKER_REGISTRY)/$(DOCKER_IMAGE):latest . --push; \
	else \
		docker buildx build --platform linux/amd64,linux/arm64 -t $(DOCKER_IMAGE):latest .; \
	fi

# Clean artifacts
clean:
	rm -rf .venv
	rm -rf .pytest_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
