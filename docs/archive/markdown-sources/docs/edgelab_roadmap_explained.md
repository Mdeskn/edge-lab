# Edge Lab — Pi Setup & Integration Roadmap (Explained Edition)
**June 4 to June 11, 2026** (you are off June 12 to 15)

This version explains what you are doing and why, not just the commands to type. Read the "Concepts" section first. After that, each day tells you what to do, and most steps include a short "What this does" note.

---

## Part 1: Concepts you will meet (read this first)

You do not need to be an expert in any of these. You just need a working mental picture so the commands later make sense.

**The terminal and SSH.**
The terminal is a text window where you type commands instead of clicking buttons. SSH ("Secure Shell") is a way to open a terminal *on another computer* over the network. When you type `ssh mae@172.22.174.149`, you are logging into the machine at that address as the user "mae", and from then on every command you type runs on *that* machine, not your laptop. You will SSH into several machines in this project. Always glance at the start of the prompt line to remind yourself which machine you are on.

**The Raspberry Pi.**
A Raspberry Pi is a small, cheap, slow computer the size of a deck of cards. In this lab it plays the role of the "edge device": something close to where data is produced, but not very powerful. The whole point of the lab is to compare running an AI model on this weak local device versus sending the work to a powerful remote GPU server.

**The GPU server and Triton.**
The GPU server is a powerful machine with a graphics card (an NVIDIA H100) that can run AI models very fast. "Triton" is NVIDIA's software that loads an AI model and waits for requests. You send it an image, it runs the model, and it sends back the result. Think of Triton as a waiter at a counter: you hand over an image, it runs to the kitchen (the GPU), and brings back the answer.

**The model: YOLOv10n and ONNX.**
YOLOv10n is the AI model that finds objects in an image (in your case, a tennis ball) and returns its position. "ONNX" is just a file format for storing a trained model so different programs can load it. Your file `yolov10n.onnx` is the model. The "n" means "nano", the smallest and fastest version.

**Docker and containers.**
Docker lets you package a program together with everything it needs to run (its libraries, its settings) into a "container". A container runs the same way on any machine, so you avoid the classic "it works on my computer but not yours" problem. `docker compose up -d` starts a set of containers in the background. `docker compose ps` lists which containers are running. `docker compose down` stops them. The `-d` means "detached" (runs in the background instead of filling your terminal).

**Kafka and topics.**
Kafka is a message system. Programs can *publish* (send) small messages to a named channel called a "topic", and other programs can *subscribe* (listen) to that topic to receive them. In your lab, the GPU server publishes its load numbers to a topic, the network machine publishes delay numbers to another topic, and the Pi publishes its results to a third. The student's code subscribes to these topics to decide where to run inference. Kafka is the nervous system connecting all the parts.

**Grafana.**
Grafana is a tool that draws live graphs from data. You point it at your Kafka data and it shows charts of GPU load, network delay, latency, and so on. It is how students *see* what is happening during the experiment instead of staring at raw numbers.

**Traffic control (tc) and the network VM.**
"tc" is a built-in Linux tool that can deliberately slow down or disrupt network traffic: add delay, add jitter (random variation in delay), drop packets. The "network VM" (also called the router machine) sits between the Pi and the GPU server. All traffic from the Pi to the server passes through it. When you apply a tc rule on the network VM, you are artificially making the network worse so students can observe the effect. This is the heart of the "network load" part of the experiment.

**VPN.**
The three lab machines live inside the university network. From home, you cannot reach them directly. A VPN ("Virtual Private Network") creates a secure tunnel that makes your Pi behave as if it were inside the university network. Without the VPN connected, none of the `172.22.174.x` addresses will respond.

**The .env file.**
A file named `.env` holds configuration values (addresses, file paths, on/off switches) as simple `NAME=value` lines. Your program reads this file at startup. Keeping settings in `.env` means you can change behavior without touching the code. You will edit this file several times as you connect more pieces.

**SeQaM.**
SeQaM is your team's platform for orchestrating experiments. In this lab it does one job: at fixed times it runs commands over SSH to switch the experiment between phases (turn GPU load on, turn network delay on, and so on). Think of it as an automated stage manager following a script.

**The two "clients" (important, do not get confused).**
There are two completely separate things both called "client":
1. **Eldiyar's test client** (`/home/mae/client/client.py` on the SeQaM machine). This is a throwaway tool he wrote to check the server works. It sends a blank image to Triton in a loop and prints the latency. It is *not* part of the lab. You can ignore it entirely.
2. **Your client** (the `client/` folder in your `edge-lab` project). This is the real application: it reads the video, runs the model locally or remotely, scores the result, and publishes metrics. This is what runs on the Pi.

When this roadmap says "the client", it always means *your* client unless it explicitly says "Eldiyar's test client".

---

## Part 2: The machines

Keep this table open every day. These are the three lab machines and their jobs.

| Machine | IP address | What it does |
|---|---|---|
| GPU Server | 172.22.174.145 | Runs Triton with the YOLOv10n model (and ResNet50 for load) |
| Network VM (router) | 172.22.174.148 | Applies network delay/jitter; all Pi-to-server traffic passes through here |
| SeQaM Machine | 172.22.174.149 | Runs Kafka, Grafana, and the SeQaM orchestrator |

Plus your own hardware: the Raspberry Pi (at home with you) and your laptop.

---

## Part 3: One decision you must settle early (the port question)

This is the single most important technical detail in the whole setup, so here it is in plain terms.

For the "network load" part of the lab to work, traffic from the Pi to the GPU server **must pass through the network VM** (172.22.174.148). That is the only place the artificial delay is applied. If the Pi talks straight to the GPU server, the delay never happens and half the lab is meaningless.

So your client's `TRITON_URL` should point at the **network VM** (172.22.174.148), and the network VM forwards that traffic onward to Triton on the GPU server.

Now the port. Triton actually listens on two different "doors":
- Port **8000** for HTTP (this is what *your* client code uses).
- Port **8001** for gRPC (this is what *Eldiyar's test client* uses).

Eldiyar's instruction file says to connect to `172.22.174.148:8001`, but that is because his test client speaks gRPC. Your client speaks HTTP, so it needs an HTTP door.

**What you need to confirm on Day 2:** that the network VM forwards an HTTP port through to Triton's port 8000 on the GPU server. The cleanest arrangement is that the network VM forwards its own port 8000 to the GPU server's port 8000. Then you set:

```
TRITON_URL=172.22.174.148:8000
```

If Eldiyar only set up forwarding for the gRPC port (8001), you have two choices, and you will pick one with him:
- **Option A (easier for you):** ask Eldiyar to add a forward rule so the network VM's port 8000 also goes to the GPU server's port 8000. One line in his iptables setup.
- **Option B:** change your client to use gRPC. This is more work and not worth it unless Option A is impossible.

Aim for Option A. Note this as the first thing to check once the VPN is up.

---

## Day 1 — Wednesday June 4: Flash the Pi and get local inference working

**Goal:** The Pi is freshly set up, and your app runs *entirely on the Pi alone* (no server, no network), showing the video with the overlay. This proves the core application works before you add any complexity.

Why start local-only? Because if something breaks later, you will know it is an integration problem, not a problem with the app itself. Always get the simplest version working first.

### Morning: flash the Pi

"Flashing" means writing a fresh operating system onto the Pi's SD card, wiping whatever was there. You are starting clean because the old setup was broken.

1. On your laptop, open **Raspberry Pi Imager** (download it from raspberrypi.com if you do not have it).
2. Choose the OS: **Raspberry Pi OS (64-bit)** with the desktop. Pick the desktop version, not "Lite", because students need to see the video window, and a window needs a desktop.
3. Before writing, click the gear icon (settings) and set:
   - Hostname: `edgelab-pi` (this is the name you will use to find the Pi on the network)
   - Enable SSH, with password login
   - Username `mae` and a password you will remember
   - Your home Wi-Fi name and password
   - Your timezone
4. Insert the SD card, write, wait for it to finish, put the card in the Pi, and power it on.
5. Wait about 2 minutes for it to boot, then from your laptop terminal:

```bash
ssh mae@edgelab-pi.local
```

**What this does:** opens a terminal on the Pi. The first time, it asks you to confirm the connection; type `yes`. Then enter the password you set.

If `edgelab-pi.local` does not work, log into your home router's admin page (usually `192.168.1.1` in a browser) to find the Pi's IP address, and use that instead: `ssh mae@192.168.x.x`.

### Update the system

```bash
sudo apt update && sudo apt full-upgrade -y
sudo reboot
```

**What this does:** `apt` is the program that installs and updates software on this kind of Linux. `update` refreshes the list of available software; `full-upgrade` installs the newest versions of everything. `sudo` means "run this as administrator" (you may be asked for your password). `reboot` restarts the Pi. Wait two minutes and SSH back in.

### Install the building blocks

```bash
sudo apt install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libopencv-dev \
    git \
    build-essential \
    curl \
    nano
```

**What this does:** installs the tools your app needs. `python3-venv` lets you make isolated Python environments. `libgl1`, `libglib2.0-0`, and `libopencv-dev` are graphics libraries that OpenCV (the video-handling library) depends on. `nano` is a simple text editor for editing files in the terminal. `git` is for code, `curl` is for testing web connections.

### Midday: copy your project onto the Pi

Your code lives on your laptop. You need it on the Pi. Run this **on your laptop** (not in the SSH session), from inside your `edge-lab` project folder:

```bash
rsync -av --exclude='.git' --exclude='__pycache__' --exclude='venv' \
    --exclude='node_modules' --exclude='data/*.onnx' --exclude='data/*.mp4' \
    ./ mae@edgelab-pi.local:/home/mae/edge-lab/
```

**What this does:** `rsync` copies files from one place to another over SSH. The `--exclude` parts skip junk you do not need to copy (the version-control folder, cached files, the old environment, and the big data files which you will copy separately). The `./` means "this folder", and the part after it is the destination on the Pi.

Now copy the big data files (they were excluded above because they are large binary files):

```bash
scp data/yolov10n.onnx mae@edgelab-pi.local:/home/mae/edge-lab/data/
scp data/test_video.mp4 mae@edgelab-pi.local:/home/mae/edge-lab/data/
scp data/ground_truth.csv mae@edgelab-pi.local:/home/mae/edge-lab/data/
```

**What this does:** `scp` ("secure copy") copies a single file over SSH. These three files are the model, the video, and the answer key (ground truth) for scoring.

### Create the Python environment on the Pi

Back in your SSH session (on the Pi):

```bash
cd /home/mae/edge-lab
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r client/requirements.txt
```

**What this does:**
- `cd` changes directory (moves into the project folder).
- `python3 -m venv venv` creates a "virtual environment" named `venv`: a private, isolated space for this project's Python libraries so they do not clash with anything else on the Pi.
- `source venv/bin/activate` switches into that environment. You will see `(venv)` appear at the start of your prompt. Whenever you work on this project, activate it first.
- `pip install -r client/requirements.txt` reads the list of required libraries and installs them.

This step takes 10 to 15 minutes on the Pi because one library (`onnxruntime`, which actually runs the model) is large. Be patient. If it fails, run the same `pip install` command again; sometimes it just needs a second try.

### Afternoon: configure and run local-only

Create your settings file:

```bash
cd /home/mae/edge-lab
cp .env.example .env
nano .env
```

**What this does:** `cp` copies the example settings file to a real one named `.env`. `nano` opens it for editing. Set these values (use arrow keys to move, type normally, then press Ctrl+O then Enter to save, and Ctrl+X to exit):

```
GROUP_ID=1
VIDEO_PATH=/home/mae/edge-lab/data/test_video.mp4
GROUND_TRUTH_PATH=/home/mae/edge-lab/data/ground_truth.csv
MODEL_PATH=/home/mae/edge-lab/data/yolov10n.onnx
TRITON_URL=
KAFKA_BROKERS=
DISPLAY_OUTPUT=true
AUTO_STOP=false
LOG_LEVEL=INFO
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
DASHBOARD_ENABLED=false
```

**Why these values:** `TRITON_URL` and `KAFKA_BROKERS` are left empty, which tells the app "there is no server, run everything locally." `DISPLAY_OUTPUT=true` shows the video window. `AUTO_STOP=false` means the app runs until you stop it (instead of waiting for an experiment phase signal, which does not exist yet). `TARGET_CLASS_ID=32` is the code number for "sports ball" in the model, so it ignores people and other objects.

Now run it. You need a screen connected to the Pi for the video window to appear. If you only have SSH access right now, temporarily set `DISPLAY_OUTPUT=false` to confirm the pipeline runs without a window.

```bash
cd /home/mae/edge-lab
source venv/bin/activate
cd client
python main.py
```

**What you should see:** a banner of startup info, then a stream of log lines as it processes frames. If `DISPLAY_OUTPUT=true` and you have a screen, a window appears showing the video with a green dot (the true ball position) and a red dot (the model's guess). Let it run 30 seconds, then press Ctrl+C to stop. It prints a summary with displacement numbers (how far off the guesses were).

**If it crashes saying "FileNotFoundError":** a path in `.env` is wrong. Check the files exist: `ls -la /home/mae/edge-lab/data/`.

### Measure how slow the Pi is

```bash
cd /home/mae/edge-lab
source venv/bin/activate
python scripts/benchmark_inference.py \
    --mode local \
    --model data/yolov10n.onnx \
    --video data/test_video.mp4 \
    --runs 50
```

**What this does:** runs the model 50 times on the Pi and reports the average time per run. This number matters: it is the "local" speed that students will compare against the remote GPU. Write it down. You want it somewhere around 200 to 500 milliseconds. If it is under 100ms, the Pi is too fast for the lab to be interesting, and you would slow it down with `--threads 1` or use a heavier model. The script prints a recommendation about this.

**End of Day 1 checklist:**
- [ ] Pi freshly flashed and reachable over SSH
- [ ] Project copied to `/home/mae/edge-lab`
- [ ] Virtual environment created, dependencies installed
- [ ] App runs local-only without crashing, video overlay visible
- [ ] Local latency number recorded

---

## Day 2 — Thursday June 5: Connect to the GPU server through the network VM

**Goal:** The Pi can reach the lab machines over VPN, and remote inference works through the network VM. By end of day you have both a "local" and a "remote" speed number.

### Morning: get on the VPN

The Pi needs to reach the `172.22.174.x` machines, which means the VPN must be running on the Pi. Set up the same VPN client you use on your laptop. The exact steps depend on which VPN software FHDO uses; if you are unsure, ask Jaime, since he raised VPN in the meetings.

Once connected, test that each machine answers:

```bash
ping 172.22.174.149   # SeQaM machine
ping 172.22.174.148   # Network VM
ping 172.22.174.145   # GPU server
```

**What this does:** `ping` sends a tiny "are you there?" message. Each should reply with timing lines. Press Ctrl+C to stop each ping. If a machine does not reply, the VPN is not routing to it; do not continue until all three answer.

### Settle the port question (from Part 3)

First check whether Triton's HTTP door is reachable through the network VM:

```bash
curl http://172.22.174.148:8000/v2/health/live
```

**What this does:** `curl` fetches a web address. This one asks Triton "are you alive?" through the network VM on the HTTP port. A reply like `{"live":true}` (or an empty success) means the forwarding is set up correctly and you can use port 8000. If you get "Connection refused" or it hangs, the network VM is probably not forwarding the HTTP port yet. Message Eldiyar (Option A from Part 3): ask him to forward the network VM's port 8000 to the GPU server's port 8000.

While you wait for his reply, confirm Triton itself is alive by checking it directly on the GPU server:

```bash
curl http://172.22.174.145:8000/v2/health/live
curl http://172.22.174.145:8000/v2/models/yolov10n/ready
```

**What this does:** the first checks Triton is running on the GPU server. The second checks your specific model is loaded. The model check should return success (HTTP 200). If it returns "not found", the model file is missing on the server. SSH in and check:

```bash
ssh mae@172.22.174.145
ls /home/mae/server/model_repository/yolov10n/1/
```

There should be a file called `model.onnx`. If it is missing, copy yours up (run this from your laptop):

```bash
scp data/yolov10n.onnx mae@172.22.174.145:/home/mae/server/model_repository/yolov10n/1/model.onnx
```

Then restart Triton on the GPU server:

```bash
ssh mae@172.22.174.145
cd /home/mae/server
docker compose down
docker compose up -d
```

**What this does:** stops and restarts the Triton container so it picks up the new model file.

### Midday: point your client at the network VM

Edit the Pi's settings:

```bash
nano /home/mae/edge-lab/.env
```

Set (assuming the HTTP forward is confirmed working):

```
TRITON_URL=172.22.174.148:8000
```

**Why the network VM and not the GPU server directly:** so traffic passes through the network VM, where delay will later be applied. This is the whole reason the network VM exists. Port 8000 because your client speaks HTTP.

### Compare local versus remote speed

```bash
cd /home/mae/edge-lab
source venv/bin/activate
python scripts/benchmark_inference.py \
    --mode both \
    --model data/yolov10n.onnx \
    --video data/test_video.mp4 \
    --triton-url 172.22.174.148:8000 \
    --runs 50
```

**What this does:** runs the model 50 times locally on the Pi and 50 times remotely on the GPU server, then prints both averages and a recommendation. The remote number should be *lower* (faster) than the local number under these clean conditions. If remote is somehow slower than local, something is wrong (wrong port, network problem, or the traffic is not reaching the GPU). Fix it before moving on, because the entire lab depends on remote being the faster option when conditions are good.

### Afternoon: run the full app with the remote server

```bash
cd /home/mae/edge-lab/client
python main.py
```

Watch for a log line like `RemoteClient connected to Triton at 172.22.174.148:8000`. Let it run two minutes, then Ctrl+C. This confirms the real application (not just the benchmark) can use the remote server.

Also confirm the support services are alive on the SeQaM machine:

```bash
ssh mae@172.22.174.149
cd /home/mae/grafana-kafka
docker compose ps
```

**What this does:** lists the running containers. You should see Kafka and Grafana. From your laptop browser, open the Kafka admin page at `http://172.22.174.149:8080` and Grafana at `http://172.22.174.149:3000` to confirm they load.

**End of Day 2 checklist:**
- [ ] VPN working on the Pi; all three machines answer ping
- [ ] HTTP port forwarding through the network VM confirmed (or fix requested from Eldiyar)
- [ ] Triton alive and yolov10n model loaded
- [ ] Remote latency measured and faster than local
- [ ] Full app runs with the remote server
- [ ] Kafka and Grafana reachable in the browser

---

## Day 3 — Friday June 6: Wire up Kafka so all the pieces talk

**Goal:** The Pi publishes its results to Kafka, and it receives GPU load, network delay, and phase messages from the other machines. Data shows up in Grafana.

Recall: Kafka is the message system. Each kind of data has its own "topic" (named channel). You will confirm three publishers are sending data, then point the Pi at Kafka so it both sends and receives.

### Morning: confirm the GPU server is publishing its load numbers

```bash
ssh mae@172.22.174.145
cd /home/mae/server
docker compose ps
```

Eldiyar said the GPU metrics publisher is set up. If it is not in the running list, start everything with `docker compose up -d`.

Now check the data is actually arriving. Open the Kafka admin page in your browser: `http://172.22.174.149:8080`. Find the Topics section and look for `/edgelab/server/metrics`. Click into it and you should see recent messages containing fields like `gpu_utilization_pct` and `triton_requests_per_sec`.

**What this proves:** the GPU server is measuring itself and broadcasting those numbers every second. Students will use these to decide whether the GPU is too busy to send work to.

### Confirm the network VM is publishing delay numbers

```bash
ssh mae@172.22.174.148
docker ps
```

**What this does:** `docker ps` lists running containers (a shorter form of `docker compose ps`). Look for a network conditions publisher. If there is none, you need to deploy yours. From your laptop:

```bash
scp -r publishers/network_conditions/ mae@172.22.174.148:/home/mae/edge-lab/publishers/
scp docker-compose.netvm.yml mae@172.22.174.148:/home/mae/edge-lab/
scp .env mae@172.22.174.148:/home/mae/edge-lab/
```

Then on the network VM, edit the settings so it knows where Kafka is:

```bash
ssh mae@172.22.174.148
cd /home/mae/edge-lab
nano .env
```

Set:
```
KAFKA_BROKERS=172.22.174.149:9092
NETWORK_INTERFACE=eth0
PHASE_FILE=/host-tmp/edgelab_phase
```

**What these mean:** `KAFKA_BROKERS` is the address of Kafka (port 9092 is Kafka's standard port). `NETWORK_INTERFACE` is the name of the network card that traffic flows through, usually `eth0`; you can confirm with the command `ip link` and looking for the main interface. `PHASE_FILE` is a small file that SeQaM will write the current phase name into, which this publisher watches.

Start the publisher:
```bash
docker compose -f docker-compose.netvm.yml up -d
```

Check the Kafka admin page for messages on `/edgelab/network/metrics` and `/edgelab/server/events/phase`.

### Midday: point the Pi at Kafka

```bash
nano /home/mae/edge-lab/.env
```

Add:
```
KAFKA_BROKERS=172.22.174.149:9092
AUTO_STOP=false
```

Run the app and watch the startup log:

```bash
cd /home/mae/edge-lab/client
python main.py
```

You should see lines confirming it connected to Kafka and subscribed to the three incoming topics (server metrics, network metrics, phase events). Let it run two minutes. Then in the Kafka admin page, open the topic `/edgelab/app/metrics/group1`. You should see the Pi's own per-frame results arriving there.

**If Kafka will not connect,** test whether the port is reachable from the Pi:

```bash
nc -zv 172.22.174.149 9092
```

**What this does:** `nc` ("netcat") checks if a port is open. Success means Kafka is reachable; a timeout means the VPN or a firewall is blocking port 9092, which is a question for Jaime.

### Afternoon: see the data in Grafana, and test the delay manually

Open Grafana (`http://172.22.174.149:3000`, login usually admin/admin). You may not have a dashboard yet; that comes on Day 5. For now just confirm data exists.

Then test that traffic control actually works. SSH into the network VM and apply a 100ms delay:

```bash
ssh mae@172.22.174.148
cd /home/mae/network_load
sudo ./tc_control.sh netem_tbf 100ms 20ms 1gbit 2mbit 50ms 60
```

**What this does:** tells Linux to add 100 milliseconds of delay (with 20ms of random jitter) to outgoing traffic for 60 seconds. While it is active, re-run the remote benchmark from the Pi and confirm the remote latency has jumped up by roughly 100ms. Then clear the rule:

```bash
sudo ./tc_control.sh clear
```

Confirm the latency drops back. This proves the network-load mechanism works.

**End of Day 3 checklist:**
- [ ] GPU metrics flowing to `/edgelab/server/metrics`
- [ ] Network metrics flowing to `/edgelab/network/metrics`
- [ ] Phase events topic exists
- [ ] Pi publishing to `/edgelab/app/metrics/group1`
- [ ] Applying a tc delay raises remote latency; clearing it restores it
- [ ] Grafana reachable and showing some data

---

## Day 4 — Monday June 9: Automate the experiment with SeQaM

**Goal:** SeQaM runs the four-phase experiment on a timer, the Pi notices each phase change, and one full 120-second cycle works start to finish.

The experiment has four phases, 30 seconds each: baseline (nothing), gpu_load (GPU busy), network_load (network slow), combined (both). SeQaM's job is to flip these switches at the right times by running commands over SSH.

### Morning: make the SeQaM script match the real machines

Your project's SeQaM files were written with placeholder IP addresses and script names that do not match Eldiyar's actual setup. You need to correct them.

First, fix the SSH target addresses:

```bash
nano seqam/ScenarioConfig.json
```

Replace the host addresses with the real ones (network VM 172.22.174.148, GPU server 172.22.174.145):

```json
{
  "distributed": {
    "router": [
      {
        "name": "net-vm",
        "description": "Edge Lab network VM",
        "host": "172.22.174.148",
        "component_type": "distributed_event_manager",
        "ssh_user": "mae",
        "ssh_port": 22
      },
      {
        "name": "gpu-server",
        "description": "Edge Lab GPU server",
        "host": "172.22.174.145",
        "component_type": "distributed_event_manager",
        "ssh_user": "mae",
        "ssh_port": 22
      }
    ]
  }
}
```

Next, find out the exact script names and paths on the real machines, because Eldiyar's may differ from your placeholders:

```bash
ssh mae@172.22.174.145
ls /home/mae/server/          # look for the GPU load script (e.g. load.sh)
```

```bash
ssh mae@172.22.174.148
ls /home/mae/network_load/    # look for the tc script (tc_control.sh)
```

**Why this matters:** SeQaM runs these exact commands. If the path or filename is wrong, the command silently fails and the phase does nothing. Eldiyar's GPU script is likely called `load.sh` and uses a setting called `CONCURRENCY_RANGE` (like `1:3:1`) rather than a single number, and his tc script is `tc_control.sh` with the longer argument list you saw on Day 3.

Now update the experiment timeline to use the real commands:

```bash
nano seqam/scenario.json
```

```json
{
  "experiment_name": "edge-computing-lab-loop",
  "execute_immediately": true,
  "eventList": [
    {
      "command": "ssh src_device_type:router src_device_name:net-vm command:'sudo /home/mae/network_load/tc_control.sh clear'",
      "executionTime": 0
    },
    {
      "command": "ssh src_device_type:router src_device_name:net-vm command:'echo baseline > /tmp/edgelab_phase'",
      "executionTime": 0
    },
    {
      "command": "ssh src_device_type:router src_device_name:net-vm command:'echo gpu_load > /tmp/edgelab_phase'",
      "executionTime": 30000
    },
    {
      "command": "ssh src_device_type:router src_device_name:gpu-server command:'bash /home/mae/server/load.sh'",
      "executionTime": 30000
    },
    {
      "command": "ssh src_device_type:router src_device_name:net-vm command:'sudo /home/mae/network_load/tc_control.sh netem_tbf 100ms 20ms 1gbit 2mbit 50ms 30'",
      "executionTime": 60000
    },
    {
      "command": "ssh src_device_type:router src_device_name:net-vm command:'echo network_load > /tmp/edgelab_phase'",
      "executionTime": 60000
    },
    {
      "command": "ssh src_device_type:router src_device_name:net-vm command:'echo combined > /tmp/edgelab_phase'",
      "executionTime": 90000
    },
    {
      "command": "ssh src_device_type:router src_device_name:gpu-server command:'bash /home/mae/server/load.sh'",
      "executionTime": 90000
    },
    {
      "command": "ssh src_device_type:router src_device_name:net-vm command:'sudo /home/mae/network_load/tc_control.sh clear'",
      "executionTime": 119000
    },
    {
      "command": "exit",
      "executionTime": 120000
    }
  ]
}
```

**How to read this:** `executionTime` is milliseconds from the start. At 0ms it clears any old delay and writes "baseline" into the phase file. At 30000ms (30s) it switches the phase to gpu_load and starts the GPU load. At 60s it adds network delay and switches to network_load. At 90s it switches to combined and loads the GPU again. At 119s it clears the delay, and at 120s it ends. The `echo baseline > /tmp/edgelab_phase` commands write the phase name into the file that the network VM's publisher watches, which then broadcasts it on Kafka so the Pi knows.

### Give SeQaM passwordless SSH access

SeQaM logs into the other machines automatically, so it cannot stop to type a password. You set up "key-based" login, where a cryptographic key replaces the password.

```bash
ssh mae@172.22.174.149

ls ~/.ssh/                                  # check for an existing key
ssh-keygen -t ecdsa -b 256 -f ~/.ssh/id_ecdsa -N ""   # make one if needed
ssh-copy-id mae@172.22.174.148              # install it on the network VM
ssh-copy-id mae@172.22.174.145              # install it on the GPU server
ssh mae@172.22.174.148 echo "router ok"     # test: should print without a password
ssh mae@172.22.174.145 echo "gpu ok"        # test: should print without a password
```

**What this does:** `ssh-keygen` creates the key pair. `ssh-copy-id` puts the public half on each target machine. After that, logging in from the SeQaM machine needs no password, which is what SeQaM requires.

### Midday: test each command by hand before trusting SeQaM

From the SeQaM machine, run each phase command manually to be sure they work:

```bash
ssh mae@172.22.174.149

ssh mae@172.22.174.148 "echo baseline > /tmp/edgelab_phase"
# check the Kafka phase topic shows "baseline"

ssh mae@172.22.174.148 "sudo /home/mae/network_load/tc_control.sh netem_tbf 100ms 20ms 1gbit 2mbit 50ms 10"
ssh mae@172.22.174.148 "sudo /home/mae/network_load/tc_control.sh clear"

ssh mae@172.22.174.145 "bash /home/mae/server/load.sh"
```

Each must run without asking for a password and without errors.

**If the tc command asks for a sudo password,** allow it to run without one:

```bash
ssh mae@172.22.174.148
sudo visudo
```

Add this single line at the bottom, then save (Ctrl+O, Enter, Ctrl+X):

```
mae ALL=(ALL) NOPASSWD: /home/mae/network_load/tc_control.sh
```

**What this does:** tells the system that user "mae" may run that one specific script as administrator without a password. This is safe because it is limited to that one script.

### Afternoon: run one full cycle

On the Pi, turn on auto-stop so it waits for the experiment to start:

```bash
nano /home/mae/edge-lab/.env      # set AUTO_STOP=true
cd /home/mae/edge-lab/client
python main.py
```

It will print `Waiting for experiment phase to start...` and pause.

Now from the SeQaM machine, start the experiment once:

```bash
SEQAM_API_URL=http://172.22.174.149:8000 ./scripts/run_scenario_loop.sh
```

Watch the Pi's console. You should see it begin at baseline, then announce each phase change at 30, 60, and 90 seconds, then stop after 120 seconds with a per-phase summary. The latency should visibly rise during gpu_load and network_load. That is the entire experiment working end to end.

**End of Day 4 checklist:**
- [ ] ScenarioConfig.json has the correct IP addresses
- [ ] scenario.json uses the real script names, paths, and tc syntax
- [ ] Passwordless SSH from SeQaM to both machines works
- [ ] Each phase command tested by hand
- [ ] One full 120-second cycle completes on the Pi
- [ ] Phase changes appear in the Pi log; latency rises under load

---

## Day 5 — Tuesday June 10: Build the dashboard and tune the difficulty

**Goal:** Grafana shows a clear live picture of the experiment, and the load levels are tuned so the effects are obvious but not extreme.

### Morning: build the Grafana dashboard

Open Grafana (`http://172.22.174.149:3000`). Create a new dashboard and add one panel per metric. A "panel" is a single graph. For each, you choose the Kafka topic as the data source and the field to plot:

1. GPU Utilization, from `/edgelab/server/metrics`, field `gpu_utilization_pct`
2. Triton Inference Time (ms), same topic, field `triton_inference_duration_ms`
3. Triton Request Rate, same topic, field `triton_requests_per_sec`
4. Network Delay (ms), from `/edgelab/network/metrics`, field `delay_ms`
5. Network Jitter (ms), same topic, field `jitter_ms`
6. Client Latency (ms), from `/edgelab/app/metrics/group1`, field `latency_ms`
7. Cumulative Displacement (the score), same topic, field `cumulative_displacement_px`
8. Current Phase, from `/edgelab/server/events/phase`, field `phase`

If Grafana cannot read Kafka directly, it needs the Kafka data source plugin:

```bash
ssh mae@172.22.174.149
cd /home/mae/grafana-kafka
docker exec -it <grafana-container-name> grafana-cli plugins install hamedkarbasi93-kafka-datasource
docker restart <grafana-container-name>
```

Find `<grafana-container-name>` by running `docker compose ps` and copying the Grafana container's name. **What this does:** installs a plugin that lets Grafana subscribe to Kafka topics, then restarts Grafana so it loads the plugin.

### Midday: tune the GPU load

The lab is only interesting if the load clearly slows things down. Test different load levels and watch the effect. On the GPU server:

```bash
ssh mae@172.22.174.145
CONCURRENCY_RANGE=1:3:1 ./load.sh
```

**What "concurrency" means:** how many requests are sent at the same time. Higher concurrency means the GPU is juggling more work, so each request waits longer. While the load runs, measure remote latency from the Pi (the remote benchmark from Day 2). You want loaded latency to be clearly higher than unloaded, ideally around twice as high. The H100 is extremely powerful, so you may need a high concurrency range to see an effect. Once you find a value that works, put it in the GPU command in `scenario.json`.

### Tune the network delay

Same idea for the network. On the network VM, try a few delay values and measure the effect from the Pi:

```bash
ssh mae@172.22.174.148
sudo /home/mae/network_load/tc_control.sh netem_tbf 50ms 10ms 1gbit 2mbit 50ms 30
# measure from Pi, then try 100ms, then 200ms
```

You want the delay big enough that, during network_load, sending to the remote server becomes *slower* than just running locally on the Pi. That is what gives students a reason to switch to local. Once you find the right value, put it in `scenario.json`.

### Afternoon: run a couple of full cycles and sanity-check the scores

Run the experiment twice. After each, check that the per-phase summary makes sense: baseline should have the best (lowest) displacement, and combined the worst. Confirm the Grafana panels move during each phase and the phase label changes at the right times. Fix anything that looks off. Common issues: a panel showing nothing usually means the topic name does not exactly match; phase changes not appearing usually means the phase file path is wrong.

**End of Day 5 checklist:**
- [ ] Grafana dashboard shows all eight metrics
- [ ] GPU load level tuned and its latency effect confirmed
- [ ] Network delay tuned and its latency effect confirmed
- [ ] Two clean full cycles
- [ ] Per-phase scores look sensible

---

## Day 6 — Wednesday June 11: Prepare the student experience and finalize

**Goal:** A second Pi works identically, students have clear instructions, and the experiment runs continuously so anyone can connect at any time.

### Morning: clone the Pi

Now that one Pi works perfectly, the fastest way to make the others is to copy its SD card. Power down the Pi, take out the card, and put it in your laptop's card reader.

```bash
diskutil list                                              # find the card's device name (macOS)
sudo dd if=/dev/disk4 of=~/edgelab-pi-ready.img bs=4M status=progress    # copy card to a file
# swap in a blank card, check its device name again, then:
sudo dd if=~/edgelab-pi-ready.img of=/dev/disk5 bs=4M status=progress     # write the file to the new card
```

**What this does:** `dd` makes an exact byte-for-byte copy. The first command copies the working card into an image file on your laptop; the second writes that image onto a fresh card. Be very careful with the device names: writing to the wrong disk could erase your laptop. Double-check with `diskutil list` each time.

On the new Pi, after first boot, change its group number so its data does not collide with Pi 1:

```bash
nano /home/mae/edge-lab/.env     # set GROUP_ID=2
```

Run a full cycle with both Pis on at once and confirm each publishes to its own topic (`group1` and `group2`) without interfering.

### Midday: write the student handout

Students need one short, clear document. Keep it to roughly two pages. It should cover:

1. What the lab is about, in two short paragraphs.
2. How to connect to their Pi and start the app.
3. Where to watch the live data (the Grafana URL).
4. The one file they edit: `client/student/sp_agent.py`, and that `decide()` must return either `"local"` or `"remote"`.
5. What information `decide()` can use: the current phase, recent latency, GPU load, and network conditions (all already available to them as properties).
6. How scoring works: cumulative displacement, lower is better.
7. How a run works: four phases, 30 seconds each, 120 seconds total.
8. What they submit or present at the end.

You already have a thorough README in the project. Pull the student-relevant parts out and trim them down. Remember your audience: keep it simple and concrete, with exact commands they can copy.

### Run the experiment continuously

The plan is for the experiment to loop 24/7 so students can connect whenever they like:

```bash
ssh mae@172.22.174.149
SEQAM_API_URL=http://172.22.174.149:8000 nohup ./edge-lab/scripts/run_scenario_loop.sh > /tmp/scenario_loop.log 2>&1 &
tail -f /tmp/scenario_loop.log
```

**What this does:** `nohup ... &` runs the loop in the background so it keeps going even after you log out. The `tail -f` shows you the live log; you should see the cycle number increase every two minutes. Press Ctrl+C to stop watching (the loop keeps running).

### Afternoon: final checks

Reset `client/student/sp_agent.py` to the default (always returns `"local"`) and run one experiment to confirm the starting point students get is sane. Then write a quick test version that always returns `"remote"` and run again: confirm "remote" scores better in baseline and worse in network_load. This proves the lab rewards good decisions, which is the entire point.

Finally, pretend you are a student: starting from a clean Pi, follow your own handout and time yourself. If any step is confusing or assumes knowledge a student would not have, fix it. Then push all your changes (the corrected scenario files, any fixes, the handout) to the repository, and confirm `.env` is still git-ignored so no secrets leak.

**End of Day 6 checklist:**
- [ ] Second Pi cloned and tested
- [ ] Both Pis run at once without interfering
- [ ] Student handout written and proofread
- [ ] Experiment looping continuously on the SeQaM machine
- [ ] Scoring validated (good decisions score better)
- [ ] All changes pushed; `.env` not committed

---

## Part 4: If something goes wrong

**VPN will not work on the Pi.** Confirm the exact VPN client with Jaime before Day 2. Fallback: connect the Pi by ethernet cable to a lab machine that already has the VPN, and share that connection.

**tc commands fail when SeQaM runs them.** Almost always the passwordless-sudo rule (Day 4) is missing. Add it.

**GPU load does not slow anything down.** The H100 is very powerful. Keep raising the concurrency. As a last resort, use the heavier ResNet50 model (already loaded on the server) as the load generator instead of YOLO.

**Kafka topic names do not match.** Check the exact names in the Kafka admin page before wiring anything. Your code expects `/edgelab/server/metrics`, `/edgelab/network/metrics`, and `/edgelab/server/events/phase`. If a publisher uses a different name, change the matching variable in the Pi's `.env`.

**SeQaM is broken or unreachable.** You do not actually need SeQaM; it only fires SSH commands on a timer. Use the manual script below instead and the lab still works.

### Manual experiment script (SeQaM replacement)

Save this on the SeQaM machine as `run_manual_scenario.sh`. It does exactly what SeQaM would do, using plain SSH and `sleep`:

```bash
#!/bin/bash
NET_VM=172.22.174.148
GPU_SERVER=172.22.174.145
TC=/home/mae/network_load/tc_control.sh

echo "Phase: baseline (30s)"
ssh mae@$NET_VM "sudo $TC clear"
ssh mae@$NET_VM "echo baseline > /tmp/edgelab_phase"
sleep 30

echo "Phase: gpu_load (30s)"
ssh mae@$NET_VM "echo gpu_load > /tmp/edgelab_phase"
ssh mae@$GPU_SERVER "CONCURRENCY_RANGE=1:5:1 bash /home/mae/server/load.sh" &
sleep 30

echo "Phase: network_load (30s)"
ssh mae@$NET_VM "sudo $TC netem_tbf 100ms 20ms 1gbit 2mbit 50ms 30"
ssh mae@$NET_VM "echo network_load > /tmp/edgelab_phase"
sleep 30

echo "Phase: combined (30s)"
ssh mae@$NET_VM "echo combined > /tmp/edgelab_phase"
ssh mae@$GPU_SERVER "CONCURRENCY_RANGE=1:5:1 bash /home/mae/server/load.sh" &
sleep 30

echo "Clearing rules"
ssh mae@$NET_VM "sudo $TC clear"
echo "Cycle complete."
```

Make it runnable and loop it:

```bash
chmod +x run_manual_scenario.sh
while true; do ./run_manual_scenario.sh; done
```

**What this does:** `chmod +x` makes the file executable. The `while true` loop runs the four-phase cycle over and over forever. This gives you the whole experiment without depending on SeQaM at all.
