# Fourier Slice Streamlit app — multi-stage, uv, amd64/arm64
FROM python:3.12-slim AS builder

WORKDIR /app

# Install uv (official distroless image)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Dependencies layer (better cache when only app code changes)
COPY pyproject.toml uv.lock* README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# App code and Streamlit config (COPY .streamlit as dir so /app/.streamlit exists)
COPY core.py app.py ./
COPY .streamlit ./.streamlit
RUN uv sync --frozen --no-dev

# Final image: runtime only
FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/core.py /app/app.py ./
COPY --from=builder /app/.streamlit ./

ENV PATH="/app/.venv/bin:$PATH"

# Streamlit in container must bind to 0.0.0.0
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
