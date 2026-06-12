# Edge Lab Complete 6-Day Roadmap (SeQaM Included)

This is the full plan: every node wired, every command placed, SeQaM installed and driving the experiment, and the phase signal flowing end to end into the app. It treats SeQaM as required, not optional.

The hardest single thing in these 6 days is the SeQaM platform install (Days 3-4). The rest is mechanical wiring. So the plan front-loads everything that does not depend on SeQaM (app fixes, data pipeline, manual verification) on Days 1-2, then installs SeQaM with a clean foundation underneath, then connects them on Days 4-5. That way if SeQaM hits a snag you can still see exactly where in the pipeline the problem is.

---

## The full system at a glance

Five machines, three roles:

**SeQaM machine (172.22.174.149)** is the brain. It runs the full SeQaM platform stack: the API (edpapi on port 8000), Kafka (9092), Grafana (host port 8300), redis, rabbitmq, plus SigNoz for telemetry. You POST experiments here. It SSHes outward to the workers.

**Network VM (172.22.174.148)** is the path shaper. The Pi sends Triton requests to its port 8001, which it forwards to the GPU server. Eldiyar's `tc_control.sh` adds delay/jitter to that forward path. It also holds `/tmp/edgelab_phase` and runs the phase publisher and network conditions publisher. SeQaM SSHes in to trigger tc and write the phase.

**GPU server (172.22.174.145)** is the worker. Triton serves yolov10n (for client requests) and resnet50_full (as a load target). Eldiyar's GPU metrics publisher runs here. SeQaM does not SSH-trigger anything here during the cycle, but the machine is still registered so the distributed component can collect its metrics.

**GPU-loader VM (pending from Amin, call its IP `<loader-ip>`)** is the stress generator. It runs `load.sh` which fires perf_analyzer at the GPU server at 145:8001. Must be a separate machine; running load on the GPU server itself would corrupt your measurements. SeQaM SSHes in to start the load during gpu_load and combined phases.

**Raspberry Pi (one per group, at students' homes via VPN)** runs your app. It pulls frames, runs local YOLO, sends to 148:8001 for remote inference, subscribes to Kafka for GPU metrics, network metrics, and phase signal. The student's `sp_agent.py` decides local vs remote per frame.

---

## How the signals flow

There are three independent data flows. Understanding all three is the whole game.

**Flow 1: control (SeQaM to workers, outbound SSH only).** SeQaM holds a private SSH key (`ecdsa`, generated during install). At scheduled moments it SSHes into a worker (looking up the host, user, and port in `ScenarioConfig.json`) and runs a shell command. That is it. SSH is one-way: SeQaM to worker, never the reverse.

**Flow 2: phase (the bridge between SeQaM and the app).** SeQaM SSHes into the network VM and writes a word (`baseline`, `gpu_load`, `network_load`, or `combined`) into `/tmp/edgelab_phase`. A small Python script called `phase_publisher.py` on the network VM watches that file. Whenever the word changes, it publishes a JSON message to the Kafka topic `edgelab.phase`. The Pi's SP agent subscribes to that topic and updates `self.experiment_phase`. So the entire bridge between SeQaM's timeline and your app is that one file plus that one publisher.

**Flow 3: data (everyone publishes into Kafka).** Eldiyar's GPU publisher on 145 sends GPU metrics to `dnn_partition.server_metrics`. The network publisher on 148 sends current tc settings to `edgelab.network.metrics`. The phase publisher sends phase changes to `edgelab.phase`. Your Pi app publishes per-frame results to `dnn_partition.client_metrics` (or whatever group-specific topic you settle on with Eldiyar). The Pi's SP agent subscribes to the first three topics so the student's `decide()` function has live signals to react to.

**The inference path is its own thing**, not a SeQaM concern: Pi sends a frame to 148:8001 (router), router forwards to 145:8001 (Triton), Triton answers, response comes back the same way. The router shapes that path with tc but never touches the request content.

---

## Day 1: Fix the app, verify every node, request what you need from Yuriy

Today's deliverable: the app is configured for the real broker and the real topic names, you have SSHed into every machine and know exactly what is running, and Yuriy has been pinged for the SeQaM tarball and registry password.

**1.1 Update the app config.** In your app's `.env` or constants file, set:

```
TRITON_URL=172.22.174.148:8001
KAFKA_BROKERS=172.22.174.149:9092
GROUP_ID=group1
SERVER_METRICS_TOPIC=dnn_partition.server_metrics
NETWORK_METRICS_TOPIC=edgelab.network.metrics
PHASE_TOPIC=edgelab.phase
APP_METRICS_TOPIC=dnn_partition.client_metrics
```

Switch the Triton client from HTTP to gRPC if you have not already. The router only forwards 8001 which is gRPC. HTTP will not work through the router.

**1.2 Fix the SP agent parser.** Eldiyar's message is nested, not flat. Update your SP agent base class to read it like this:

```python
def on_server_metrics(self, message):
    server = message["server"]
    totals = message["totals"]
    yolo = next((m for m in message["models"] if m["model_name"] == "yolov10n"), None)

    self.gpu_util_pct = server["gpu_util_percent"]
    self.gpu_temp_c = server["gpu_temp_c"]
    self.cpu_util_pct = server["cpu_util_percent"]
    self.total_rps = totals["total_rps"]
    self.total_pending = totals["total_pending_requests"]
    if yolo:
        self.yolo_queue_ms = yolo["avg_queue_time_ms"]
        self.yolo_infer_ms = yolo["avg_compute_infer_ms"]
        self.yolo_pending = yolo["pending_requests"]

def on_phase(self, message):
    self.experiment_phase = message["phase"]

def on_network_metrics(self, message):
    self.net_delay_ms = message.get("delay_ms", 0)
    self.net_jitter_ms = message.get("jitter_ms", 0)
    self.net_bandwidth = message.get("bandwidth", "unknown")
```

Update any sample student `decide()` to use these new field names.

**1.3 SSH into every machine and inventory.** Run these on each:

```bash
ssh mae@172.22.174.149       # SeQaM machine
docker ps                     # confirm Eldiyar's kafka, grafana running
ls ~                          # see what folders exist

ssh mae@172.22.174.145       # GPU server
docker ps                     # triton container should be running
ps aux | grep kafka_metrics   # GPU publisher should be running
curl http://localhost:8000/v2/models/yolov10n | head    # model metadata

ssh mae@172.22.174.148       # Network VM
ls /home/mae/network_load/   # tc_control.sh should be there
sudo /home/mae/network_load/tc_control.sh clear    # should run cleanly
```

**1.4 Open the Kafka UI and confirm topics.** Visit `http://172.22.174.149:8080`. You should see `dnn_partition.server_metrics` with messages. Click in, confirm the nested structure matches what your parser expects. If something looks different, fix the parser today, not later.

**1.5 Message Yuriy.** You need three things from him to install SeQaM on Day 3:

- The platform binary distribution tarball (the latest one)
- The docker registry password for user `pigovsky`
- Confirmation that the SeQaM API port is 8000 (or whatever it actually is)

Send the message today so the wait does not block you mid-week.

**1.6 Message Eldiyar.** Two questions:

- Is there already a network conditions publisher, or do you need to write one?
- What field names does he publish for network metrics?

End of day: app config is correct, parser is fixed, you have SSH into every machine, and Yuriy and Eldiyar know what you need from them.

---

## Day 2: Get the Pi working end to end, write the phase publisher, write the network publisher

Today's deliverable: the Pi runs your app, sees inference results coming back through the router, and the phase pipeline is built and tested manually (no SeQaM yet). By tonight, writing a word into `/tmp/edgelab_phase` on the network VM makes the Pi's `self.experiment_phase` update.

**2.1 Prepare the Pi.** Flash Raspberry Pi OS Desktop (64-bit). Set up VPN to the university network. Clone your repo. Install dependencies. Drop in the video file, ground_truth.csv, and yolov10n.onnx.

Test reachability:

```bash
nc -vz 172.22.174.149 9092    # Kafka
nc -vz 172.22.174.148 8001    # Triton via router
```

If either fails, escalate to Amin or IT today. Routing through VPN cannot be diagnosed from your laptop.

**2.2 Run the app, watch it work.** Start the app. You should see the video window, the green ground truth dot, the red prediction dot, latency, mode. Let it run a few minutes. Confirm both `local` mode and `remote` mode work. Open the Kafka UI and check `dnn_partition.client_metrics` is getting your messages.

**2.3 Write the phase publisher.** Save on the network VM as `/home/mae/network_load/phase_publisher.py`:

```python
import time, json
from kafka import KafkaProducer

PHASE_FILE = "/tmp/edgelab_phase"
TOPIC = "edgelab.phase"
BROKERS = "172.22.174.149:9092"

producer = KafkaProducer(
    bootstrap_servers=BROKERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

last = None
while True:
    try:
        with open(PHASE_FILE) as f:
            phase = f.read().strip()
        if phase and phase != last:
            producer.send(TOPIC, {"phase": phase, "timestamp": time.time()})
            producer.flush()
            print(f"[phase_publisher] {phase}")
            last = phase
    except FileNotFoundError:
        pass
    time.sleep(1)
```

Install kafka-python:

```bash
ssh mae@172.22.174.148
pip install kafka-python --user
nohup python3 /home/mae/network_load/phase_publisher.py > /tmp/phase_publisher.log 2>&1 &
disown
```

**2.4 Write the network conditions publisher.** Save as `/home/mae/network_load/network_publisher.py`:

```python
import time, json, subprocess, re
from kafka import KafkaProducer

TOPIC = "edgelab.network.metrics"
BROKERS = "172.22.174.149:9092"
IFACE = "eth0"   # change if your interface is different

producer = KafkaProducer(
    bootstrap_servers=BROKERS,
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

def read_tc():
    out = subprocess.run(["tc", "-s", "qdisc", "show", "dev", IFACE],
                         capture_output=True, text=True).stdout
    delay_ms = 0.0
    jitter_ms = 0.0
    bw = "none"
    m = re.search(r"delay\s+(\d+\.?\d*)ms(?:\s+(\d+\.?\d*)ms)?", out)
    if m:
        delay_ms = float(m.group(1))
        if m.group(2):
            jitter_ms = float(m.group(2))
    m = re.search(r"rate\s+(\S+)", out)
    if m:
        bw = m.group(1)
    return {"delay_ms": delay_ms, "jitter_ms": jitter_ms, "bandwidth": bw,
            "timestamp": time.time()}

while True:
    msg = read_tc()
    producer.send(TOPIC, msg)
    producer.flush()
    time.sleep(1)
```

Run it the same way:

```bash
nohup python3 /home/mae/network_load/network_publisher.py > /tmp/network_publisher.log 2>&1 &
disown
```

**2.5 Test the phase pipeline manually.** From the SeQaM machine or your laptop:

```bash
ssh mae@172.22.174.148 "echo baseline > /tmp/edgelab_phase"
# in the Kafka UI, open topic edgelab.phase, should see one message
ssh mae@172.22.174.148 "echo gpu_load > /tmp/edgelab_phase"
# another message appears
```

On the Pi, with the app running, your SP agent's `self.experiment_phase` should update. Print it from inside `decide()` to confirm.

End of day: Pi works, phase signal works without SeQaM in the loop, network conditions show up in Kafka.

---

## Day 3: Install the full SeQaM platform on the SeQaM machine

Today's deliverable: the SeQaM API is running on 172.22.174.149:8000, the central stack is up, and the SSH key `ecdsa` has been generated.

Eldiyar's standalone Kafka and Grafana on 149 will collide with the platform's bundled ones. Before installing, settle this:

```bash
ssh mae@172.22.174.149
docker ps                          # note Eldiyar's kafka and grafana containers
cd /home/mae/grafana-kafka         # his compose
docker compose down                # stop his standalone stack
```

The platform brings its own Kafka and Grafana. Eldiyar's publishers will continue working because they connect to `172.22.174.149:9092`, and that port will still serve Kafka, just from the platform's compose instead of his.

**3.1 Unpack the platform tarball Yuriy sent you.** Use scp to copy it over:

```bash
# from your laptop
scp <platform-distribution>.tar.gz mae@172.22.174.149:~/

# on the SeQaM machine
ssh mae@172.22.174.149
mkdir -p ~/seqam-platform
cd ~/seqam-platform
tar zxvf ~/<platform-distribution>.tar.gz
./scripts/install-docker.sh
```

After this you should see `api/`, `bare-composes/`, `scripts/` inside `~/seqam-platform`.

**3.2 Clone and patch SigNoz.**

```bash
cd ~
git clone https://github.com/SigNoz/signoz.git
cd signoz
git apply ../seqam-platform/apply-me-on-new-signoz.diff || echo "patch may already be applied"
```

**3.3 Run the install step.** This creates the config folder and generates the ecdsa key.

```bash
cd ~/seqam-platform
./api/bin/install.sh
ls ~/.seqam_fh_dortmund_project_emulate/
# should see: ModuleConfig.json, ScenarioConfig.json, ecdsa, ecdsa.pub, env, ExperimentConfig.json
```

**3.4 Set the central IP in env.**

```bash
sed -i -e "s/PLEASE_CHANGE_ME/172.22.174.149/g" ~/.seqam_fh_dortmund_project_emulate/env
nano ~/.seqam_fh_dortmund_project_emulate/env
```

Confirm these lines are correct:

```
PRIVATE_DOCKER_REGISTRY="pigovsky"
SEQAM_CENTRAL_HOST=172.22.174.149
CLUSTER_HOST="$SEQAM_CENTRAL_HOST"
DATABASE_ENDPOINT="signoz-clickhouse"
OTLP_URL="signoz-otel-collector:4317"
API_HOST="$SEQAM_CENTRAL_HOST"
API_PORT=8000
```

**3.5 Docker login to the private registry.**

```bash
docker login -u pigovsky
# paste the password Yuriy sent
```

**3.6 Generate compose files and bring up the stack.**

```bash
cd ~/seqam-platform/bare-composes
./generate-docker-composes.sh

cd ~/signoz/deploy/docker
docker compose up -d

cd ~/seqam-platform/bare-composes/seqam-central
docker compose up -d
```

Wait a couple of minutes for everything to settle.

**3.7 Verify.**

```bash
docker ps    # expect edpapi, kafka, grafana, redis, rabbitmq, signoz containers
curl http://172.22.174.149:8000/config/ScenarioConfig.json
```

The curl should return JSON. If it does, the API is alive. Open `http://172.22.174.149:8300` for Grafana (platform-bundled). Open `http://172.22.174.149:8080` for the Kafka UI (Eldiyar's, if still up, or the platform's).

Also confirm your publishers still work: they should still be writing to `172.22.174.149:9092` because the platform brought its own Kafka on the same port. Check the Kafka UI for fresh messages.

If at any point the install errors out, the most common causes are: forgetting `./api/bin/install.sh` before `generate-docker-composes.sh`, forgetting `docker login`, or a stale SigNoz patch. Re-run from the failed step.

End of day: SeQaM API answers, ecdsa key exists, your publishers still feed Kafka.

---

## Day 4: Wire the workers (SSH key, ScenarioConfig, distributed components, first SeQaM SSH test)

Today's deliverable: SeQaM can SSH from the central machine into each worker and run a command, and all three workers are registered with the platform.

**4.1 Install the SeQaM public key on every worker.**

```bash
ssh mae@172.22.174.149
ssh-copy-id -i ~/.seqam_fh_dortmund_project_emulate/ecdsa.pub mae@172.22.174.148    # net-vm
ssh-copy-id -i ~/.seqam_fh_dortmund_project_emulate/ecdsa.pub mae@172.22.174.145    # gpu-server
ssh-copy-id -i ~/.seqam_fh_dortmund_project_emulate/ecdsa.pub <loader-user>@<loader-ip>   # loader (once Amin provisions it)
```

Test passwordless access:

```bash
ssh -i ~/.seqam_fh_dortmund_project_emulate/ecdsa mae@172.22.174.148 "echo net-ok"
ssh -i ~/.seqam_fh_dortmund_project_emulate/ecdsa mae@172.22.174.145 "echo gpu-ok"
ssh -i ~/.seqam_fh_dortmund_project_emulate/ecdsa <loader-user>@<loader-ip> "echo loader-ok"
```

All three must print their message with no password prompt.

**4.2 Write ScenarioConfig.json.** Edit `~/.seqam_fh_dortmund_project_emulate/ScenarioConfig.json` on the SeQaM machine:

```json
{
  "distributed": {
    "server": [
      {
        "name": "net-vm",
        "description": "Network VM: traffic control + phase file",
        "component_type": "network_event_manager",
        "host": "172.22.174.148",
        "ssh_user": "mae",
        "ssh_port": 22
      },
      {
        "name": "gpu-server",
        "description": "Triton GPU server",
        "component_type": "distributed_event_manager",
        "host": "172.22.174.145",
        "ssh_user": "mae",
        "ssh_port": 22
      }
    ],
    "ue": [
      {
        "name": "gpu-loader",
        "description": "Separate VM running load.sh against the GPU server",
        "component_type": "distributed_event_manager",
        "host": "<loader-ip>",
        "ssh_user": "<loader-user>",
        "ssh_port": 22
      }
    ]
  }
}
```

Note the device types: `server` for the network VM and the GPU server, `ue` for the loader. Not `router`. Jaime explicitly said he did not recognize that framing.

Copy this into the running stack's config folder and restart the API so it picks up the change:

```bash
cp ~/.seqam_fh_dortmund_project_emulate/ScenarioConfig.json \
   ~/seqam-platform/bare-composes/seqam-central/config/ScenarioConfig.json

cd ~/seqam-platform/bare-composes/seqam-central
docker compose restart edpapi
curl http://172.22.174.149:8000/config/ScenarioConfig.json    # confirm all three appear
```

**4.3 Passwordless sudo on the network VM.** SeQaM cannot type a password. On 148:

```bash
ssh mae@172.22.174.148
sudo visudo
# add at the bottom:
mae ALL=(ALL) NOPASSWD: /home/mae/network_load/tc_control.sh
```

**4.4 Install the distributed components on the workers.** Each worker gets the matching agent. From the SeQaM machine:

```bash
# Network VM (network event manager)
scp -r ~/seqam-platform/bare-composes/seqam-network-event-manager mae@172.22.174.148:~/
ssh mae@172.22.174.148
cd ~/seqam-network-event-manager
./load-image.sh seqam-network-event-manager.tar.gz   # adjust to actual filename
nano docker-compose.yaml
# set: SEQAM_DEVICE_TYPE=server SEQAM_DEVICE_NAME=net-vm SEQAM_DEVICE_HOST=172.22.174.148
#      API_HOST=172.22.174.149 API_PORT=8000 OTLP_URL=172.22.174.149:4317
#      NETWORK_EVENT_MANAGER_PORT=9003
docker compose up -d

# GPU server (distributed event manager)
ssh mae@172.22.174.149
scp -r ~/seqam-platform/bare-composes/seqam-distributed-event-manager mae@172.22.174.145:~/
ssh mae@172.22.174.145
cd ~/seqam-distributed-event-manager
./load-image.sh seqam-distributed-event-manager.tar.gz
nano docker-compose.yaml
# set: SEQAM_DEVICE_TYPE=server SEQAM_DEVICE_NAME=gpu-server SEQAM_DEVICE_HOST=172.22.174.145
#      API_HOST=172.22.174.149 API_PORT=8000 OTLP_URL=172.22.174.149:4317
#      DISTRIBUTED_EVENT_MANAGER_PORT=9001
docker compose up -d

# GPU loader (distributed event manager)
ssh mae@172.22.174.149
scp -r ~/seqam-platform/bare-composes/seqam-distributed-event-manager <loader-user>@<loader-ip>:~/
ssh <loader-user>@<loader-ip>
cd ~/seqam-distributed-event-manager
./load-image.sh seqam-distributed-event-manager.tar.gz
nano docker-compose.yaml
# set: SEQAM_DEVICE_TYPE=ue SEQAM_DEVICE_NAME=gpu-loader SEQAM_DEVICE_HOST=<loader-ip>
#      API_HOST=172.22.174.149 API_PORT=8000 OTLP_URL=172.22.174.149:4317
#      DISTRIBUTED_EVENT_MANAGER_PORT=9011
docker compose up -d
```

These agents report metrics to the platform's own SigNoz/Grafana, which is a separate pipeline from the Kafka topics your students consume. Jaime wanted them installed, so they are.

**4.5 First SeQaM-triggered SSH test.** Create a tiny experiment that just writes a phase and exits. Save on the SeQaM machine as `~/test_one_event.json`:

```json
{
  "experiment_name": "first-ssh-test",
  "execute_immediately": true,
  "eventList": [
    {
      "command": "ssh src_device_type:server src_device_name:net-vm \"echo seqam_test > /tmp/edgelab_phase\"",
      "executionTime": 0
    }
  ]
}
```

POST it:

```bash
curl -X POST http://172.22.174.149:8000/config/ExperimentConfig.json \
  -H "Content-Type: application/json" \
  -d @~/test_one_event.json
```

Then check:

```bash
ssh mae@172.22.174.148 "cat /tmp/edgelab_phase"   # should print: seqam_test
```

And open the Kafka UI, check `edgelab.phase` got a message with `"phase": "seqam_test"`.

This single command end to end is the proof that SeQaM is wired correctly. If this works, the rest is just writing more events.

End of day: ScenarioConfig is registered, distributed components are running, and one SSH command went all the way from a POST to the API to a Kafka message the Pi can see.

---

## Day 5: Full four-phase experiment via SeQaM, GPU load, looping

Today's deliverable: SeQaM runs the full 120-second cycle, the Pi sees all four phases, GPU load runs during gpu_load and combined, and the cycle loops unattended.

**5.1 Install load.sh on the loader VM.** If Amin has provisioned it:

```bash
ssh <loader-user>@<loader-ip>
nano load.sh    # paste Eldiyar's section 7.1 script verbatim
chmod +x load.sh

# quick test
CONCURRENCY_RANGE=1 ./load.sh
```

Open the Kafka UI during the test and watch `dnn_partition.server_metrics`. The `gpu_util_percent` and `total_rps` and `yolov10n` queue times should jump from 0 to something high.

If the loader VM still is not provisioned, put load.sh temporarily on the SeQaM machine and SSH to itself for the load command. Note this is a stopgap; move it to the dedicated VM the moment Amin delivers.

**5.2 Write the full experiment.** Save as `~/experiment.json` on the SeQaM machine:

```json
{
  "experiment_name": "edge-lab-cycle",
  "execute_immediately": true,
  "eventList": [
    {
      "command": "ssh src_device_type:server src_device_name:net-vm \"sudo /home/mae/network_load/tc_control.sh clear; echo baseline > /tmp/edgelab_phase\"",
      "executionTime": 0
    },
    {
      "command": "ssh src_device_type:ue src_device_name:gpu-loader \"pkill -f perf_analyzer; true\"",
      "executionTime": 0
    },
    {
      "command": "ssh src_device_type:server src_device_name:net-vm \"echo gpu_load > /tmp/edgelab_phase\"",
      "executionTime": 30000
    },
    {
      "command": "ssh src_device_type:ue src_device_name:gpu-loader \"nohup env CONCURRENCY_RANGE=1:5:1 bash /home/<loader-user>/load.sh > /tmp/load.log 2>&1 &\"",
      "executionTime": 30000
    },
    {
      "command": "ssh src_device_type:server src_device_name:net-vm \"sudo /home/mae/network_load/tc_control.sh netem_tbf 0.5ms 1ms 1gbit 2mbit 50ms 30; echo network_load > /tmp/edgelab_phase\"",
      "executionTime": 60000
    },
    {
      "command": "ssh src_device_type:server src_device_name:net-vm \"echo combined > /tmp/edgelab_phase\"",
      "executionTime": 90000
    },
    {
      "command": "ssh src_device_type:ue src_device_name:gpu-loader \"nohup env CONCURRENCY_RANGE=1:5:1 bash /home/<loader-user>/load.sh > /tmp/load.log 2>&1 &\"",
      "executionTime": 90000
    },
    {
      "command": "ssh src_device_type:server src_device_name:net-vm \"sudo /home/mae/network_load/tc_control.sh clear\"",
      "executionTime": 119000
    },
    {
      "command": "ssh src_device_type:ue src_device_name:gpu-loader \"pkill -f perf_analyzer; true\"",
      "executionTime": 119000
    }
  ]
}
```

What each piece does in order:
- 0ms: clear network, kill any leftover load, write baseline
- 30s: write gpu_load, start load on the loader (detached so it does not block)
- 60s: apply tc jitter+delay, write network_load
- 90s: write combined, kick load again
- 119s: clear network, stop load so the next cycle starts clean

**5.3 POST it and watch.**

```bash
curl -X POST http://172.22.174.149:8000/config/ExperimentConfig.json \
  -H "Content-Type: application/json" \
  -d @~/experiment.json
```

From the network VM, watch the phase file:

```bash
ssh mae@172.22.174.148
watch -n 1 cat /tmp/edgelab_phase
```

You should see baseline, gpu_load, network_load, combined cycling.

On the Pi, with the app running, watch the overlay. During gpu_load the remote latency should rise (Triton is busy). During network_load the remote latency should rise (jitter and delay). During combined both effects appear. During baseline everything is calm.

Tune the values if the differences are too subtle. Increase the tc jitter, or push concurrency higher. Eldiyar warns about going too high; start with the values above and only nudge up.

**5.4 Loop it 24/7.** Save on the SeQaM machine as `~/run_edgelab_loop.sh`:

```bash
#!/bin/bash
API=http://172.22.174.149:8000
EXP=/home/mae/experiment.json

while true; do
  curl -s -X POST "$API/config/ExperimentConfig.json" \
    -H "Content-Type: application/json" \
    -d @$EXP > /dev/null
  echo "[$(date)] cycle started"
  sleep 120
done
```

```bash
chmod +x ~/run_edgelab_loop.sh
nohup ~/run_edgelab_loop.sh > ~/edgelab_loop.log 2>&1 &
disown
tail -f ~/edgelab_loop.log
```

To stop later: `pkill -f run_edgelab_loop.sh`.

**5.5 Write the fallback in case SeQaM stalls.** Even though SeQaM is now your primary, keep a manual cycle script ready in case it dies overnight while you are away. Save as `~/manual_cycle.sh` for emergencies. (Same content as Day 4 of the earlier roadmap; you have it.)

End of day: full cycle loops, Pi sees every phase, displacement varies meaningfully across phases.

---

## Day 6: Student materials and final dry run

Today's deliverable: a student you have never met can clone your repo, follow the README, and have their first scored run within 30 minutes.

**6.1 Write the student README.** Cover:

- What the lab is: minimize cumulative displacement of the predicted dot vs ground truth dot over time.
- How to get on the network: VPN setup, which IPs they reach.
- How to clone and install: dependencies, paths, model file.
- The one file they edit: `client/student/sp_agent.py`, specifically the `decide()` method. Return `"local"` or `"remote"`.
- What live signals they have inside `decide()`: list every `self.*` attribute the SP agent base provides, with example values and units.
- How to run the app: command, what they see.
- How they are scored: cumulative displacement, lower is better, shown in the Grafana dashboard.
- One worked example: a naive always-remote agent vs a smart one that switches on `self.experiment_phase == "network_load"`.

Keep it under 5 pages. Students do not read longer.

**6.2 Dry run as a student.** Wipe a Pi or use a fresh user. Follow your own README exactly. Time yourself. Note every place you stumbled and fix the docs. If something requires a manual step the student cannot do (VPN access, group ID assignment), put it in the docs and identify who they ask.

**6.3 Confirm the dashboard.** Open Grafana at `http://172.22.174.149:8300`. Set up at minimum one panel: cumulative displacement per group from the `dnn_partition.client_metrics` topic (or your agreed app metrics topic). Even a basic line chart is enough for the contest feel.

**6.4 Set everything to run unattended.** Before you go on holiday June 12:

- `run_edgelab_loop.sh` running in nohup, `tail -f` showed at least 5 cycles cleanly
- `phase_publisher.py` running on the network VM (nohup or systemd)
- `network_publisher.py` running on the network VM
- All distributed components running
- SeQaM central up
- Eldiyar's GPU publisher still running
- Triton still serving

Confirm with one final test: connect from the Pi to the running system without touching anything else. If your app sees phases changing and remote latency changing, the lab is alive.

End of day: students inherit a working, looping, scored lab.

---

## What to never cut, even under time pressure

- Pi-to-Triton path working through the router
- The phase signal reaching the Pi via Kafka
- App metrics getting published so students can be scored
- A reproducible cycle that runs unattended

If anything else goes wrong, you can still finish the lab. If any of these four breaks, you cannot.

---

## Coordination notes

Send today (Day 1):
- Yuriy: tarball, registry password, API port confirmation
- Eldiyar: confirm network publisher exists or not, confirm exact topic and field names

Pending (chase whenever):
- Amin: GPU-loader VM, IP, user

You are off June 12 to 15. Whatever is running by end of Day 6 (June 11) is what students get unless you fix it June 16. Treat June 16 as buffer for fixes that emerged while you were away, not for original work.
