FROM python:3.11-slim

# Install system dependencies at runtime layer
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8080

# Start gunicorn
CMD gunicorn server:app \
    --workers 2 \
    --timeout 300 \
    --bind 0.0.0.0:$PORT
