FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install -r server/requirements.txt
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 7860

# Run the app
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
