# Use the official Python image as base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*

# Set working directory in the container
WORKDIR /app

# Copy and install requirements
COPY requirement.txt .
RUN pip install --no-cache-dir -r requirement.txt

# Copy the application code into the container
COPY . .

# Expose the port where FastAPI will run
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "127.0.0.0", "--port", "8000"]
