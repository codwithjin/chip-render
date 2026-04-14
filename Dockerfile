FROM python:3.11.9-slim

# Cache-bust: 2026-04-14a
# System deps for MediaPipe + Node for React build
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# React build
COPY frontend/package*.json frontend/
RUN cd frontend && npm ci
COPY frontend/ frontend/
RUN cd frontend && npm run build

# Force cache invalidation — increment BUILD_ID to bust Railway's layer cache
ENV BUILD_ID=2026-04-14-02

# Copy models explicitly
COPY models/ models/
RUN echo "=== models ===" && ls -lh /app/models/ || echo "models dir missing"

# Copy rest of application
COPY . .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn server:app --workers 2 --timeout 300 --bind 0.0.0.0:${PORT:-8080}"]
