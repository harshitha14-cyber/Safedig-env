FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r server/requirements.txt

ENV PYTHONPATH=/app/server

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]