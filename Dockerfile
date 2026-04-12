FROM python:3.11-slim

WORKDIR /app

# Copy requirements
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire server folder
COPY server/ ./server/

# Debug: Verify files are copied
RUN ls -la ./server/

# Set Python path
ENV PYTHONPATH=/app

EXPOSE 7860

CMD ["python", "server/app.py"]