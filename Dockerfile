# ─────────────────────────────────────────────────────────────────────
# STAGE 1: Builder
# ─────────────────────────────────────────────────────────────────────
FROM python:3.10-slim as builder

WORKDIR /app

# Install build dependencies
# libgomp1 is required by faiss-cpu for OpenMP parallelism
# gcc and python3-dev are required for compiling some packages
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies into the virtual environment
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────
# STAGE 2: Runtime
# ─────────────────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# We need libgomp1 in the runtime environment for faiss-cpu
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security (Production Best Practice)
RUN useradd -m -s /bin/bash appuser

# Copy the pre-compiled virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy the rest of the application code
COPY --chown=appuser:appuser . .

# Switch to the non-root user
USER appuser

# Expose port 7860
EXPOSE 7860

# ─────────────────────────────────────────────────────────────────────
# Start the FastAPI service
# Hugging Face Spaces provides the PORT environment variable (default 7860).
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}