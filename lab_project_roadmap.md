# Edge Computing Lab — Project Roadmap

**Timeline:** May 23 to June 15 (24 days, then 2 days of buffer before June 17 deadline)

**Your current resources:**
- 1 Raspberry Pi 5 (at home with you)
- 3 VMs from Amin (Network VM, SeQaM Central, GPU Server)
- All application code (`edge-lab` project)
- Your laptop
- The team (Jaime, Yuri, Eldiyar, Yvan) reachable by message

**What's still needed (your blockers to resolve):**
- A YOLOv10n ONNX model file (you can produce this on your laptop in 5 minutes)
- A test video (you can record one in 5 minutes)
- Ground truth CSV (you generate this from the video and model)
- Triton status confirmation from Jaime or Eldiyar
- Yuri's per-service Kafka topic changes (1-day job, status to confirm)
- Network access between the Pi and the 3 VMs (VPN setup)

---

## High-level phases

**Week 1 (May 23-31): Pi standalone**
Get the Pi number. Get the remote number if Triton is already up. Decide on the model variant. You can do almost all of this alone.

**Week 2 (June 1-7): Infrastructure**
Stand up the three VMs. Verify all the publishers, Triton, tc rules, and Kafka. This week needs team coordination.

**Week 3 (June 8-15): Integration and student materials**
SeQaM scenarios, Grafana dashboards, image the other 3 Pis, write student-facing docs.

---

## Communication to do today (Saturday May 23)

Send these three messages right now, before doing anything else. They unblock everything else in the timeline.

### Message 1: to Jaime
> Hi Jaime, two questions to unblock the benchmark numbers:
> 1. Is Triton already running on the FHDO server with our YOLOv10n model loaded? If yes, what's the URL and port?
> 2. If not, who do I coordinate with to bring it up? Eldiyar mentioned he wanted to test before his holiday. Did that happen?
>
> Once I have the Pi number and the Triton number I can confirm whether YOLOv10n at 4 threads is the right setup or whether we need to slow the Pi down or move up to YOLOv10s.

### Message 2: to Yuri
> Hi Yuri, quick check: did the per-service Kafka topic publishing go in? I'll need it before students can subscribe to their own group's metrics without seeing the other groups' data.

### Message 3: to Amin (or whoever provisioned the VMs)
> Hi, can you confirm I have SSH access to the three VMs (Network VM, SeQaM Central, GPU Server)? Their IP addresses and any access keys would help.

Send those before going further. They take 2 minutes total and replies will arrive while you work on the Pi.

---

## Week 1: May 23 to May 31 (Pi standalone work)

### Saturday May 24 (today)

After sending the three messages above:

**Set up the Pi.** Follow the detailed Phase 1 instructions I sent you earlier. Specifically:

1. Flash Raspberry Pi OS Lite 64-bit onto the SD card using Raspberry Pi Imager
2. Configure WiFi, SSH, hostname during the imager step
3. Boot the Pi, SSH in from your Mac
4. Update the system with `sudo apt update && sudo apt upgrade`
5. Install system dependencies: `sudo apt install -y python3-pip python3-venv libgl1 libglib2.0-0 git build-essential python3-dev`
6. SCP the entire `edge-lab` folder from your laptop to the Pi at `/home/mae/edge-lab`
7. Create a venv: `cd ~/edge-lab && python3 -m venv venv && source venv/bin/activate`
8. Install Python dependencies: `pip install -r client/requirements.txt`

By end of Saturday: Pi is up, code is on it, venv is ready.

### Sunday May 25

**Get model, video, ground truth onto the Pi.**

On your laptop:

1. Create a working folder: `mkdir -p ~/yolo-export && cd ~/yolo-export`
2. Create a venv and install ultralytics: `python3 -m venv venv && source venv/bin/activate && pip install ultralytics`
3. Export the model: `yolo export model=yolov10n.pt format=onnx` (produces `yolov10n.onnx`)
4. Record or find a 30-second test video, save as `test_video.mp4` in this folder
5. Copy the ground truth script: `cp /Users/maede/Library/CloudStorage/OneDrive-FHDortmund/Work/Apps/edge-lab/ground_truth/generate_ground_truth.py .`
6. Generate ground truth: `python generate_ground_truth.py --video test_video.mp4 --model yolov10n.onnx --output ground_truth.csv --conf 0.3`
7. SCP all three files to the Pi: `scp yolov10n.onnx test_video.mp4 ground_truth.csv mae@edgelab-pi.local:/home/mae/edge-lab/`

By end of Sunday: Pi has model, video, ground truth ready to use.

### Monday May 26

**Run the benchmark. Get the Pi number.**

On the Pi via SSH:

1. Create the `.env` file (copy from `.env.example`, set paths, leave TRITON_URL and KAFKA_BROKERS empty)
2. Activate venv: `source venv/bin/activate`
3. Run the benchmark:
   ```bash
   python scripts/benchmark_inference.py \
       --mode local \
       --model yolov10n.onnx \
       --video test_video.mp4 \
       --runs 50 \
       --warmup 5
   ```
4. **Record the mean latency.** This is Pi number 1.

Interpret per the rules from Jaime:
- Under 100ms: too fast, lab won't work. Try `--threads 1`. If still under 200ms, switch to YOLOv10s and repeat the export + ground truth + benchmark.
- 200-500ms: ideal, you're done.
- 500-800ms: workable but slow.
- Over 800ms: something's wrong, investigate.

By end of Monday: you have the Pi number and you know which model variant to use.

### Tuesday May 27

**Get the remote number (if Triton is available).**

By now Jaime should have replied. Two scenarios:

**Scenario A: Triton is running on FHDO.**
1. Get the URL and port from Jaime
2. From the Pi: `python scripts/benchmark_inference.py --mode remote --model yolov10n.onnx --triton-url <fhdo-url>:<port>`
3. Record the remote mean latency. This is number 2.
4. Run `--mode both` for the comparison and recommendation
5. Send both numbers to Jaime, ask if they look right

**Scenario B: Triton is not running yet.**
1. Defer the remote benchmark to next week (Week 2)
2. Skip ahead to Wednesday's task

### Wednesday May 28

**Run the full client app on the Pi (local-only mode).**

This verifies the application works end to end, even if you can't test remote yet.

On the Pi:
1. Make sure `.env` has TRITON_URL and KAFKA_BROKERS empty, DISPLAY_OUTPUT=false
2. Run: `cd ~/edge-lab && source venv/bin/activate && python client/main.py`
3. Let it run for 60-90 seconds. Watch the logs.
4. Press Ctrl+C. Verify the final summary prints.
5. Check `results.csv`: should have ~600-900 rows with `processing_mode=local`, real latency numbers, real displacement numbers.

If something doesn't work, debug it now. This is the last week you can do it alone before depending on infrastructure.

By end of Wednesday: the full client app works on the Pi in local-only mode.

### Thursday May 29

**Visual verification (optional but recommended).**

If you have a monitor and keyboard you can connect to the Pi:
1. Shut down the Pi: `sudo shutdown -h now`
2. Plug in monitor, keyboard, power
3. Log in directly
4. Run: `cd ~/edge-lab && source venv/bin/activate && DISPLAY_OUTPUT=true python client/main.py`
5. Confirm the two OpenCV windows appear: input frame and output with overlay
6. See the green dot (ground truth), red dot (prediction), yellow line, HUD text
7. Press `q` to quit

This is your "this thing actually works" moment. Take a screenshot or short video for the project records.

### Friday May 30 - Sunday May 31

**Buffer / polish / preparation for Week 2.**

Use this time for:
1. Any debugging you skipped
2. Re-running benchmarks if you changed something
3. Reading the team's responses and acting on them
4. Preparing for Week 2: list out what each VM needs

If everything worked smoothly, you have free time. Use it for the open questions:
- Did Yuri finish the Kafka topic work?
- Did you get the remote number?
- Is everyone aligned on the model variant?

---

## Week 2: June 1 to June 7 (Infrastructure)

Eldiyar and Yvan are back this week. Coordinate with them.

### Monday June 1

**Sync with the team.**

1. Get an update from everyone: where is Triton? Where is Kafka? Is the GPU stressor script ready?
2. Confirm SSH access to all three VMs
3. Get the IP addresses written down and tested (`ping` and `ssh` each one)
4. Verify VPN works from your Pi to all three VMs

### Tuesday June 2

**Bring up the SeQaM Central VM.**

This is the most important one because everything else depends on Kafka being reachable.

1. SSH into the SeQaM Central VM
2. Install Docker and Docker Compose: `sudo apt install -y docker.io docker-compose-v2`
3. Get the SeQaM platform code (clone from wherever the team hosts it)
4. Verify Yuri's per-service Kafka topic changes are merged
5. Start the SeQaM stack: `docker compose up -d`
6. This should bring up Kafka (port 9092), ClickHouse, the SeQaM API, and Grafana (port 3000)
7. Verify Kafka is reachable from your laptop: `nc -zv <seqam-ip> 9092`
8. From your laptop, install Kafka tools: `pip install confluent-kafka`
9. Test publishing:
   ```python
   from confluent_kafka import Producer
   p = Producer({'bootstrap.servers': '<seqam-ip>:9092'})
   p.produce('test', b'hello')
   p.flush()
   ```
10. Test subscribing on the same topic to see your message arrive
11. Open `http://<seqam-ip>:3000` in your browser to access Grafana. Default login is `admin` / `admin`

By end of Tuesday: Kafka is reachable, Grafana is reachable, SeQaM is responding.

### Wednesday June 3

**Bring up the GPU Server VM.**

1. SSH in, install Docker and NVIDIA Container Toolkit
2. Verify GPU is detected: `nvidia-smi` (should show the H100)
3. Get the YOLOv10n ONNX file onto the VM
4. Copy it to the model repository: `mkdir -p triton/model_repository/yolov10n/1 && cp yolov10n.onnx triton/model_repository/yolov10n/1/model.onnx`
5. Edit `.env` to set `KAFKA_BROKERS=<seqam-ip>:9092`
6. Start the stack: `docker compose -f docker-compose.gpu-server.yml up -d`
7. Test Triton is alive: `curl http://localhost:8000/v2/health/live` (should return 200)
8. From the Pi (over VPN): `curl http://<gpu-vm-ip>:8000/v2/health/live` (should also return 200)
9. Verify GPU metrics publisher is working: from your laptop, subscribe to the `gpu.metrics` Kafka topic, should see one JSON message per second
10. Get Eldiyar's GPU stressor bash script onto the VM (or write one if he hasn't delivered it)
11. Test the stressor manually: `bash gpu_stressor.sh yolov10n 100 30` (should send 100 concurrent requests for 30 seconds)
12. **Now run the remote benchmark from the Pi:** `python scripts/benchmark_inference.py --mode remote --model yolov10n.onnx --triton-url <gpu-vm-ip>:8000`
13. **Record the remote number.** Send it to Jaime alongside the Pi number from Week 1.

By end of Wednesday: Triton works, GPU metrics flowing to Kafka, you have number 2.

### Thursday June 4

**Bring up the Network VM.**

1. SSH in, install Docker
2. Verify `tc` is installed: `which tc` should show `/usr/sbin/tc`. If not: `sudo apt install -y iproute2`
3. Edit `.env` to set `KAFKA_BROKERS=<seqam-ip>:9092` and `NETWORK_INTERFACE=eth0`
4. Start the publisher: `docker compose -f docker-compose.netvm.yml up -d`
5. Verify network conditions publisher is sending to Kafka. Subscribe to `network.conditions` topic, should see 0/0/0 every 2 seconds
6. Test applying tc rules: `bash scripts/tc_apply.sh eth0 100 20 2`
7. Verify the Kafka topic immediately shows new conditions (100ms delay, 20ms jitter, 2% loss)
8. Clear: `bash scripts/tc_clear.sh eth0`
9. Verify Kafka topic shows zeros again

**Set up traffic routing through the Network VM.** This is the trickiest part. You need traffic from the Pi destined for the GPU server to actually go through the Network VM so that tc rules affect it.

Two approaches:

**Approach 1: Simple HTTP proxy.** Run a small HTTP forwarder on the Network VM that listens on port 8000 and forwards to the GPU server's port 8000. The tc rules on the Network VM's interface then affect the traffic. This is the simplest approach.

**Approach 2: IP routing / NAT.** Configure the Network VM as a router so that the Pi sends Triton traffic to the Network VM's IP, and the Network VM forwards it to the GPU server. Needs iptables NAT rules and IP forwarding enabled. More "correct" but more setup.

Ask Jaime or Yuri which approach they prefer. Approach 1 is faster to set up.

By end of Thursday: Network VM applies tc rules, publishes to Kafka, and routes traffic from Pi to GPU server.

### Friday June 5

**End-to-end test.**

1. On the Pi, set `TRITON_URL=<network-vm-ip>:8000` (or whatever port the proxy uses)
2. From the Pi: `python scripts/benchmark_inference.py --mode both` and verify it works
3. On the Network VM, apply tc rules: `bash tc_apply.sh eth0 100 20 2`
4. Re-run the benchmark from the Pi: remote latency should be higher now (because of the network delay)
5. Clear tc rules, verify latency goes back down
6. Run the full client app on the Pi with TRITON_URL and KAFKA_BROKERS both set
7. Verify the SP-Agent's mode actually changes by setting LOG_LEVEL=DEBUG
8. Verify Kafka topics are receiving:
   - `app.metrics.group1` from the Pi
   - `gpu.metrics` from the GPU server
   - `network.conditions` from the Network VM
9. Use a Kafka consumer on your laptop to peek at each topic

By end of Friday: all four machines are talking to each other, data is flowing through Kafka.

### Saturday June 6 - Sunday June 7

**Buffer for debugging.** Inevitably something will break this week. Use the weekend to fix it before Week 3.

Likely issues:
- VPN doesn't allow some port
- tc rules don't survive reboot
- Triton has trouble with the ONNX model (might need different config)
- Kafka has trouble with multiple subscribers
- Network VM forwarding doesn't work for HTTPS but works for HTTP
- GPU metrics publisher polls too slowly or too fast

Document anything you fix in a `KNOWN_ISSUES.md` file in the repo.

---

## Week 3: June 8 to June 15 (Integration and student materials)

### Monday June 8 - Tuesday June 9

**SeQaM scenario integration.**

1. Upload `seqam/scenario.json` to the SeQaM web interface (or however SeQaM is operated)
2. Confirm SeQaM can SSH into the GPU server and Network VM. Set up SSH keys if not already done.
3. Trigger the scenario manually
4. Watch the four Kafka topics:
   - `experiment.phase` should show phase strings: baseline, gpu_load, network_load, combined
   - `gpu.metrics` should show GPU load rising during gpu_load and combined phases
   - `network.conditions` should show delay/jitter/loss applied during network_load and combined phases
   - `app.metrics.group1` should show latency rising during the bad phases
5. Run the client app on the Pi during a full 120-second loop. Watch the SP-Agent (with default "always local" decide()) and observe how cumulative displacement accumulates.

By end of Tuesday: full scenario runs end-to-end, all topics get the right data.

### Wednesday June 10 - Thursday June 11

**Grafana dashboards.**

1. Open Grafana at `http://<seqam-ip>:3000`
2. Add ClickHouse as a data source (plugin may need installing first)
3. Create a new dashboard with these panels:
   - **End-to-end latency over time** (line chart): from `app.metrics.group{N}` filtered by group_id, plotting latency_ms over timestamp
   - **Cumulative displacement** (single stat or line chart): same source, plotting cumulative_displacement_px
   - **GPU utilization** (gauge): from `gpu.metrics`, plotting gpu_utilization_pct
   - **Triton queue duration** (line chart): from `gpu.metrics`, plotting triton_queue_duration_ms
   - **Network delay/jitter/loss** (three small stats): from `network.conditions`
   - **Current phase** (banner): from `experiment.phase`, displays current phase name in large text
4. Export the dashboard JSON: Settings → JSON Model → copy to `seqam/grafana_dashboard.json` in your repo
5. Test by running the scenario and watching the dashboard in real time

By end of Thursday: Grafana shows everything students need to see during their lab session.

### Friday June 12

**Image the other 3 Pis.**

If you have access to 3 more Pi 5s now:

1. Use the SD card you have in your current Pi as a template
2. Clone it to 3 new cards using `dd` or Pi Imager's "Clone" feature
3. Boot each new Pi with its own hostname (edgelab-pi-2, edgelab-pi-3, edgelab-pi-4)
4. Edit each one's `.env` to set its own `GROUP_ID` (1, 2, 3, 4)
5. Test all 4 Pis can reach the VMs

If you don't have the other Pis yet: do this as soon as they arrive.

### Saturday June 13 - Sunday June 14

**Student materials.**

Write a short student-facing guide (3-4 pages max) covering:
1. What the lab is and what they're trying to optimize
2. How to access their Pi (VPN instructions, SSH credentials)
3. Where to edit code: `client/student/sp_agent.py` only
4. What metrics are available to their `decide()` method
5. How to view their Grafana dashboard
6. How to submit results
7. Scoring: lowest `cumulative_displacement_px` across one full 120-second loop wins

This is separate from the technical README. Students don't need to know about Triton or tc or Kafka internals. They need to know "look at these numbers, write this function."

### Monday June 15

**Final integration test and handoff.**

1. Run the lab end-to-end one more time with all 4 Pis simultaneously (if available)
2. Verify each group sees only their own data in Grafana
3. Verify scenario runs through 2-3 full loops without errors
4. Send the project to Rolf with a summary email:
   - Architecture overview
   - How to start/stop the experiment
   - Known limitations
   - Student instructions

---

## Communication checkpoints throughout

Send weekly updates to Jaime, including:
- What numbers you measured
- What's blocking you
- What you need from the team

Daily during Week 2 and Week 3, post quick status to your group chat:
- What you did today
- What you'll do tomorrow
- Who you need help from

---

## What to do right now, in this exact order

1. **Send the three messages** (Jaime, Yuri, Amin) at the top of this document
2. **Open Raspberry Pi Imager on your Mac** and flash the SD card
3. **Boot the Pi** while you go make coffee
4. **SSH in** when it's up

Everything else flows from there.

You can absolutely have the Pi number by Monday night. Don't let perfect be the enemy of good. Get something running this weekend.
