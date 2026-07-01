# =============================================================================
# Stage 1: Build (install production dependencies)
# =============================================================================
FROM python:3.13-slim AS builder

# Install UV for fast dependency resolution
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /build

# Copy only dependency metadata for layer caching
COPY pyproject.toml uv.lock* ./

# Install production dependencies into a venv under /build/.venv
RUN uv sync --frozen --no-dev --no-editable

# =============================================================================
# Stage 2: Runtime
# =============================================================================
FROM python:3.13-slim

# Install ca-certificates for HTTPS calls to LLM providers, and curl for
# health-check probing. Remove apt lists to keep the image lean.
RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user (uid/gid 1000) for security.
RUN groupadd -r llmbench && useradd -r -g llmbench -d /app -s /sbin/nologin llmbench

WORKDIR /app

# Copy the pre-built venv from the builder stage
COPY --from=builder --chown=llmbench:llmbench /build/.venv /app/.venv

# Copy application code (owned by non-root user)
COPY --chown=llmbench:llmbench . .

# Create the persistent data directory
RUN mkdir -p /app/data && chown llmbench:llmbench /app/data

# Ensure the venv is used
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Default to production; override via env file or compose.
    ENVIRONMENT=production

# Health check: the app must respond within 5 seconds, with 3 retries,
# checked every 30 seconds after a 15-second startup grace period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -sf http://localhost:8000/dashboard || exit 1

EXPOSE 8000

# Drop privileges before starting the server
USER llmbench

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips=*"]