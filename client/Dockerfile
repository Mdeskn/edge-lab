FROM python:3.11-slim

# System deps needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The client uses flat imports, so its own modules land directly in /app.
# `common` is shared with the dashboard backend and is imported as a package.
COPY client/ /app/
COPY common/ /app/common/

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
