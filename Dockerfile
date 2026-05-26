# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend-build
WORKDIR /frontend
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ curl tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend

# CPU-only torch first — saves ~2 GB vs the default CUDA build
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY backend/ ./

# Copy built frontend into the path main.py expects: ../frontend/dist
COPY --from=frontend-build /frontend/dist ../frontend/dist

# Persistent directories — overridden by volume mounts in production
RUN mkdir -p data/cache models/checkpoints mlruns

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# Single worker — scheduler lives inside the uvicorn process
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
