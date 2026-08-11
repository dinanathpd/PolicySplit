# Mininet Deployment Guide for Split-Mamba

## Overview

This guide walks through deploying the Split-Mamba IDS on a Mininet testbed.
The architecture runs:

- **P4 data plane**: BMv2 `simple_switch_grpc` with stateful feature extraction
- **SDN controller**: Python process running BiMamba+Entropy via ONNX Runtime
- **Traffic replay**: `tcpreplay` injecting CICIoT2023 PCAPs

## Prerequisites

On the Mininet machine you need:

- Ubuntu 20.04 or 22.04
- Python 3.10+
- Mininet installed (`sudo apt-get install mininet`)
- BMv2 with P4Runtime (`simple_switch_grpc`)
- p4c compiler
- tcpreplay (`sudo apt-get install tcpreplay`)
- CICIoT2023 PCAP files (download separately)

### Install BMv2 + p4c (if not already installed)

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y automake cmake libjudy-dev libgmp-dev \
    libpcap-dev libboost-dev libboost-system-dev libboost-thread-dev \
    libevent-dev libssl-dev libgrpc++-dev protobuf-compiler-grpc \
    python3-pip git

# Clone and build behavioral-model (BMv2)
git clone https://github.com/p4lang/behavioral-model.git
cd behavioral-model
./install_deps.sh
./autogen.sh
./configure --with-pi --with-thrift --enable-grpc
make -j$(nproc)
sudo make install
sudo ldconfig
cd ..

# Clone and build p4c compiler
git clone --recursive https://github.com/p4lang/p4c.git
cd p4c
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install
cd ../..
```

## Step 1: Transfer Files to Mininet Machine

On the **development machine** (where you trained models):

```bash
cd /path/to/enn_ddos

# Pack deployment files into a tarball
bash scripts/pack_for_mininet.sh

# Transfer to Mininet machine
scp enn_ddos_mininet.tar.gz user@mininet-host:~/
```

The tarball (~1 MB) contains:

- `controller/` - SDN controller + ONNX inference engine
- `p4/` - P4 source code
- `topology/` - Mininet topology
- `models/onnx/bimamba_ent.onnx` - BiMamba+Entropy ONNX model (816 KB)
- `scripts/` - Environment setup, benchmarks, traffic replay
- `config/` - Experiment config

## Step 2: Setup on Mininet Machine

```bash
ssh user@mininet-host
tar xzf enn_ddos_mininet.tar.gz
cd enn_ddos

# Create lightweight Python venv (no PyTorch needed)
bash scripts/setup_mininet_env.sh
source venv/bin/activate
```

This installs only: numpy, onnxruntime, scapy, grpcio (~80 MB total).

## Step 3: Compile P4 Program

```bash
# Compile the P4 program for BMv2
mkdir -p p4/generated
p4c-bm2-ss \
    --p4v 16 \
    --p4runtime-files p4/generated/feature_extraction.p4info.txt \
    -o p4/generated/feature_extraction.json \
    p4/feature_extraction.p4

echo "Compiled. Output: p4/generated/feature_extraction.json"
```

## Step 4: Download PCAP Files (if not already available)

The CICIoT2023 PCAPs are needed for traffic replay. If you already have
them on the Mininet machine, skip this step.

```bash
# Option A: Copy from development machine
scp -r user@dev-machine:/path/to/dataset_cic_iot_2023/PCAP/ dataset/PCAP/

# Option B: Download directly (large, ~100 GB total)
# See: https://www.unb.ca/cic/datasets/iotdataset-2023.html
# Download a subset for testing:
mkdir -p dataset/PCAP
python scripts/download_pcap.py --output dataset/PCAP --subset small
```

For a quick test, you only need a few PCAPs:

- One benign file
- One DDoS file (e.g., DDoS-SYN_Flood)
- One DoS file (e.g., DoS-TCP_Flood)

## Step 5: Start the Mininet Topology

```bash
# Start Mininet with the P4 switch
sudo python topology/mininet_topo.py \
    --p4-json p4/generated/feature_extraction.json
```

This creates:

```
h1 (10.0.1.1) - attacker1  \
h2 (10.0.1.2) - attacker2  --- [P4 Switch s1] --- h4 (10.0.2.1) victim
h3 (10.0.1.3) - benign     /         |
                               h5 (10.0.3.1) controller
```

You will get a `mininet>` prompt.

## Step 6: Start the SDN Controller

In the Mininet CLI:

```bash
# Start the controller on h5 (background)
mininet> h5 cd /home/user/enn_ddos && source venv/bin/activate && \
         python controller/sdn_controller.py \
         --model models/onnx/bimamba_ent.onnx \
         --mode onnx \
         --listen-port 9999 &

# Verify controller is running
mininet> h5 ps aux | grep sdn_controller
```

The controller:

1. Listens for telemetry packets on UDP port 9999
2. Runs BiMamba+Entropy inference via ONNX Runtime
3. Pushes drop rules back to the P4 switch via P4Runtime gRPC

## Step 7: Replay Traffic

```bash
# Replay benign traffic from h3
mininet> h3 tcpreplay --intf1=h3-eth0 --mbps=10 \
         dataset/PCAP/benign_sample.pcap &

# Replay DDoS attack from h1
mininet> h1 tcpreplay --intf1=h1-eth0 --mbps=10 \
         dataset/PCAP/ddos_syn_sample.pcap &

# Replay DoS attack from h2
mininet> h2 tcpreplay --intf1=h2-eth0 --mbps=10 \
         dataset/PCAP/dos_tcp_sample.pcap &
```

Or use the automated replay script:

```bash
mininet> h1 python scripts/replay_attack.py \
         --attack dataset/PCAP/DDoS-SYN_Flood/ \
         --interface h1-eth0 --speed 10 &

mininet> h3 python scripts/replay_attack.py \
         --benign dataset/PCAP/Benign/ \
         --interface h3-eth0 --speed 10 &
```

## Step 8: Monitor and Collect Results

```bash
# Check controller logs
mininet> h5 tail -f /tmp/sdn_controller.log

# Check P4 switch counters
mininet> sh simple_switch_CLI --thrift-port 9090 <<< "counter_read flow_counter 0"

# Check if drop rules are installed
mininet> sh simple_switch_CLI --thrift-port 9090 <<< "table_dump attack_drop"

# Test connectivity (should work for benign, blocked for attacks)
mininet> h3 ping h4 -c 5
mininet> h1 ping h4 -c 5
```

## Step 9: Run ONNX Benchmark on the Mininet Machine

To measure inference latency on the Mininet machine's hardware:

```bash
# Exit Mininet first, or open a new terminal
source venv/bin/activate
python scripts/benchmark_onnx.py --full-pipeline
```

This reports:

- Single-sample ONNX inference latency
- Batch throughput
- Full pipeline timing (parse + infer + rule push)

## Step 10: Clean Up

```bash
mininet> exit
sudo mn -c   # Clean up Mininet state
```

---

## Quick Reference

| Command                                                                                | Description                            |
| -------------------------------------------------------------------------------------- | -------------------------------------- |
| `bash scripts/pack_for_mininet.sh`                                                     | Pack deployment tarball on dev machine |
| `bash scripts/setup_mininet_env.sh`                                                    | Setup Python venv on Mininet machine   |
| `p4c-bm2-ss -o p4/generated/feature_extraction.json p4/feature_extraction.p4`          | Compile P4                             |
| `sudo python topology/mininet_topo.py --p4-json p4/generated/feature_extraction.json`  | Start topology                         |
| `python controller/sdn_controller.py --model models/onnx/bimamba_ent.onnx --mode onnx` | Start controller                       |
| `python scripts/benchmark_onnx.py --full-pipeline`                                     | Benchmark ONNX on Mininet hardware     |

## Troubleshooting

**"simple_switch_grpc not found"**

- BMv2 is not installed or not in PATH. Install BMv2 with gRPC support.

**"Mininet not installed"**

- Run: `sudo apt-get install mininet`

**"p4c-bm2-ss not found"**

- The p4c compiler is not installed. Follow the p4c build instructions above.

**Controller crashes with "No module named onnxruntime"**

- Activate the venv: `source venv/bin/activate`

**P4 compilation errors**

- Ensure you are using p4c version compatible with the v1model architecture.
  The P4 code targets BMv2 v1model.

**ONNX model not found**

- Verify `models/onnx/bimamba_ent.onnx` exists. If missing, re-export from
  the development machine:
  ```bash
  conda run -n spike312 python -c "
  import torch
  from src.models.bimamba_controller import BiMambaController
  model = BiMambaController(num_classes=8, d_model=64, n_layers=4,
      d_state=16, d_conv=4, expand=1, n_entropy_features=6,
      use_entropy=True, input_dim=3, dropout=0.15)
  ckpt = torch.load('models/bimamba_ent_best.pt', map_location='cpu')
  model.load_state_dict(ckpt.get('model_state_dict', ckpt))
  model.eval()
  seq = torch.randn(1, 100, 3)
  ent = torch.randn(1, 6)
  torch.onnx.export(model, (seq, ent), 'models/onnx/bimamba_ent.onnx',
      input_names=['sequences', 'entropy'], output_names=['logits'],
      dynamic_axes={'sequences': {0: 'batch'}, 'entropy': {0: 'batch'},
                    'logits': {0: 'batch'}})
  "
  ```

**Traffic not flowing through switch**

- Check ARP: `mininet> h1 arp -a`
- Add static ARP entries if needed:
  ```
  mininet> h1 arp -s 10.0.2.1 00:00:00:00:02:01
  mininet> h4 arp -s 10.0.1.1 00:00:00:00:01:01
  ```

**No telemetry packets reaching controller**

- Verify the P4 program's clone session is configured:
  ```
  simple_switch_CLI --thrift-port 9090 <<< "mirroring_add 500 5"
  ```
  (where 5 is the port connected to h5, the controller host)
