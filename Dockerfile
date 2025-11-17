FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml poetry.lock* requirements.txt* ./

# Install Poetry if pyproject.toml exists, otherwise use pip
RUN if [ -f pyproject.toml ]; then \
        pip install poetry && \
        poetry config virtualenvs.create false && \
        poetry install --no-dev; \
    elif [ -f requirements.txt ]; then \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Default command (can be overridden)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

