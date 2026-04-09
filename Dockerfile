FROM python:3.11-slim

WORKDIR /app

# Install uv
RUN pip install uv

# Copy dependency files
COPY pyproject.toml uv.lock requirements.txt ./

# Sync dependencies
RUN uv sync

# Copy all files
COPY . .

EXPOSE 7860

CMD ["uv", "run", "safedig"]