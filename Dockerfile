FROM python:3.11-slim

# System deps for MediaPipe + Node for React build
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    nodejs \
    npm \
    curl \
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

# Copy application
COPY . .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn server:app --workers 2 --timeout 300 --bind 0.0.0.0:${PORT:-8080}"]
