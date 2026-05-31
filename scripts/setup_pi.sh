#!/bin/bash
# Run once on each Pi to install dependencies.
#
# Prerequisites: the repo must already be deployed to /opt/edge-lab before
# running this script. Clone or rsync it there first, for example:
#   git clone <repo-url> /opt/edge-lab
# or:
#   rsync -av --delete ./ pi@<pi-ip>:/opt/edge-lab/
set -e
echo "Setting up Edge Lab on Raspberry Pi..."
sudo apt update && sudo apt install -y python3.11 python3.11-venv python3-pip libopencv-dev
python3.11 -m venv /opt/edge-lab-venv
/opt/edge-lab-venv/bin/pip install --upgrade pip
/opt/edge-lab-venv/bin/pip install -r /opt/edge-lab/client/requirements.txt
echo "Setup complete."
