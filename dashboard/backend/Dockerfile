FROM python:3.11-slim

WORKDIR /app

COPY dashboard/backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard /app/dashboard
COPY common /app/common

ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "dashboard.backend.run"]
