EdgeLab machines

VM 1 — Kafka / Grafana / SeQaM target
IP: 172.22.174.149
Hostname: ubuntu
User: mae
Password: <VM1_PASSWORD>
Main role: Kafka broker, Kafka UI, Grafana, Prometheus
Important ports:
  Kafka: 9092
  Kafka UI: 8080
  Grafana: 3000
  Prometheus: 9090

VM 2 — GPU Server / Triton
IP: 172.22.174.145
Hostname: edgelab-GPU
User: mae
Password: <VM2_PASSWORD>
Main role: Triton server with yolov10n and resnet50_full
Important ports:
  Triton HTTP: 8000
  Triton gRPC: 8001
  Triton metrics: 8002
Main path:
  /home/mae/server

VM 3 — Network / Router / TC
IP: 172.22.174.148
Hostname: ubuntu
User: mae
Password: <VM3_PASSWORD>
Main role: forwards client traffic to GPU server and applies tc network shaping
Main path:
  /home/mae/network_load
Important files:
  tc_control.sh
  tc_controller.py
  network_conditions_publisher.py

VM 4 — GPU-load machine / LC1
IP: 172.22.232.19
Hostname: LC1
User: lc1
Password: <LC1_PASSWORD>
Main role: runs external GPU load against GPU Server
Target for load:
  172.22.174.145:8001
Important note:
  This machine should hit the GPU server directly, not through the router.

Student / Raspberry Pi client
IP: <student-pi-ip>
Hostname: <pi-hostname>
User: <pi-user>
Password: <PI_PASSWORD>
Main role: runs the client app, local/remote decision logic, publishes client metrics
Remote inference target:
  172.22.174.148:8001
Kafka:
  172.22.174.149:9092