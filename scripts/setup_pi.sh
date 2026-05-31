#!/bin/bash
# Run once on each Pi to install dependencies.
set -e
echo "Setting up Edge Lab on Raspberry Pi..."
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip libopencv-dev
python3.11 -m venv /opt/edge-lab-venv
/opt/edge-lab-venv/bin/pip install --upgrade pip
/opt/edge-lab-venv/bin/pip install -r /opt/edge-lab/client/requirements.txt
echo "Setup complete."
