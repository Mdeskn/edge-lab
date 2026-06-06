# NOTE: Docker is NOT the current deployment method for the Network VM (172.22.174.148).
# Docker is not installed on the Network VM. These scripts run as plain Python scripts:
#
#   cd /home/mae/network_load
#   python3 -m venv .venv && source .venv/bin/activate
#   pip install -r requirements.txt
#   python3 -u network_conditions_publisher.py --kafka-bootstrap-servers 172.22.174.149:9092
#   python3 -u tc_controller.py --kafka-bootstrap-servers 172.22.174.149:9092
#
# This Dockerfile is kept for reference / future containerised deployments only.

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
