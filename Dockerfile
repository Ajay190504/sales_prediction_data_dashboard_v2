FROM python:3.11-slim

# Install system deps for common Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Allow choosing a requirements file at build time (default to the trimmed one)
ARG REQ=requirements-render.txt
COPY ${REQ} /app/requirements.txt

RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy application
COPY . /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Use gunicorn for production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
