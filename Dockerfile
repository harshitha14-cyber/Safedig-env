FROM python:3.11-slim

WORKDIR /app

# Copy everything from your project
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r server/requirements.txt

# Debug: Show what files are where
RUN echo "=== Root directory ===" && ls -la
RUN echo "=== Server directory ===" && ls -la server/

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 7860

# Run with python directly
CMD ["python", "server/app.py"]