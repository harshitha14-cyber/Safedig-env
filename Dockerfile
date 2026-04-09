FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD curl -f http://localhost:8000/info || exit 1
CMD ["sh", "-c", "python -m uvicorn app:app --host 0.0.0.0 --port 8000 & python gradio_ui.py"]