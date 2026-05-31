FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    iproute2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY network_conditions_publisher.py .
COPY tc_controller.py .

ENV PYTHONUNBUFFERED=1

CMD ["python", "network_conditions_publisher.py"]
