FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY remote_inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY remote_inference/app.py ./app.py
COPY client/inference ./inference

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8100"]
