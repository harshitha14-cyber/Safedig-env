FROM python:3.11-slim

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install -r server/requirements.txt
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app

# Run the app
CMD ["python", "server/app.py"]

EXPOSE 7860