FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gpu_metrics_publisher.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "gpu_metrics_publisher.py"]
