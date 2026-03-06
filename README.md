# Fourier Slice Theorem Visualizer

Interactive Streamlit app for the Central Slice Theorem (Fourier-slice): projections and 2D FFT slices.

## Setup (recommended: uv)

```bash
brew install uv   # or: pip install uv
make install      # creates .venv and installs deps
make test         # run tests
make run          # run Streamlit app
```

Without uv: `make install-pip` then `make test-pip` / `make run-pip`.

**macOS:** If you need to build packages from source (e.g. after a new Python version), install the Xcode Command Line Tools once: `xcode-select --install`. The dev environment also optionally installs `watchdog` on macOS only (for better file watching with Streamlit).

## Docker

```bash
make docker-build        # build image (current arch)
make docker-run          # run app at http://localhost:8501
make docker-build-multi  # build for amd64 + arm64 (set DOCKER_REGISTRY=user to push)
```

When running in Docker, use **Local URL** (`http://localhost:8501`) only. The Network/External URLs Streamlit prints are for container networking and are not reachable from your host.

## Reproducibility

- **uv**: `make install` runs `uv sync` only (no venv replace prompt). Install uses `uv.lock` for exact versions. Use `make venv-recreate` when you want a fresh env.
- **uv.lock** is similar in purpose to **poetry.lock**: pinned, reproducible dependency tree; format differs.
- **pip**: `pip install -r requirements.txt` for loose versions.
