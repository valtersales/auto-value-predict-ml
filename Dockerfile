FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install Poetry and dependencies (including dev dependencies for testing)
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

