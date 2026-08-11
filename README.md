# PolicySplit

A split intrusion detection pipeline for IoT networks, divided between a
programmable switch and an SDN controller.

The switch keeps five statistics per flow in registers: byte count,
exponentially weighted averages of packet length and inter-arrival time, and the
smallest and largest packet seen. When a flow's tenth packet arrives it quantises
those five values and looks the cell up in a match-action table. A hit drops the
flow; a miss is mirrored to the controller as a 44-byte UDP report, and either
verdict is remembered for the rest of the flow. Two bitmap sketches also track
how many distinct sources address each victim and how many ports each source
contacts, which is what makes a spoofed flood visible when no single flow is long
enough to classify.

The table is not hand written. A compiler maps traffic classes onto the two
actions the switch can take, allow or drop, pools the training evidence of every
class that implies the same action, and emits entries only for cells whose drop
decision meets an error target measured on held-out flows. It never emits an
allow entry, so anything uncertain is deferred to the controller rather than
forwarded.

This repository holds the P4 program, the policy compiler, the controller, the
PCAP feature extraction, and the training and evaluation code.

## Requirements

- Python 3.10 or newer
- Docker, for the P4 compiler and the BMv2 software switch
- A CUDA GPU is optional. It is only needed to retrain the sequence model, and
  the configuration assumes about 6 GB. Everything else runs on CPU.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

CICIoT2023 is distributed by the Canadian Institute for Cybersecurity and is not
redistributed here. Register at
[unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html),
then download the captures with your own session cookie:

```bash
export CIC_COOKIE="Token=..."           # from your authenticated browser session
python scripts/download_pcap.py --cookie "$CIC_COOKIE" --out dataset_cic_iot_2023
```

Do not commit that cookie. Build the flow corpus from the raw captures:

```bash
PYTHONPATH=. python scripts/pcap_to_entropy_fast.py \
    --pcap-dir dataset_cic_iot_2023/PCAP \
    --output data/sequences/entropy_flows_v2.h5 \
    --workers 10 --window 10.0
```

This writes one HDF5 file holding a 100 x 3 packet series per flow, six context
features, labels, and flow identifiers.

## Usage

```bash
# Compile the drop policy and emit the P4 table entries and quantiser
PYTHONPATH=. python scripts/train_rf.py

# Sequence models and the baseline suite
PYTHONPATH=. python scripts/run_experiment.py --stage all
PYTHONPATH=. python scripts/train_cnn.py

# Aggregate every run into summary.json, tables, and figures
PYTHONPATH=. python scripts/plot_results.py
```

`scripts/train_rf.py` writes `p4/generated/classifier_entries.txt` and
`p4/generated/bucketise.p4inc`. The second file is the quantiser that the P4
program includes, so the switch and the compiler always share cut points.

`scripts/run_experiment.py` is resumable and skips finished cells. Use
`--stage ablation`, `--stage context-controls`, or `--stage protocol` to run part
of the grid, `--dry-run` to print the commands, and `--force` to recompute.

Split rules, train-only normalisation, subsampling, and the context presentation
modes all live in `src/preprocessing/protocol.py`, so a run is described by its
flags rather than by edits to a script.

## Data plane

The P4 toolchain runs in Docker, so no system packages are required.

```bash
# Compile for BMv2
docker run --rm -v "$PWD/p4:/w" -w /w p4lang/p4c \
    p4c-bm2-ss --target bmv2 --arch v1model \
    -o generated/feature_extraction.json \
    --p4runtime-files generated/feature_extraction.p4info.txtpb \
    feature_extraction.p4

# Runtime image; adds iproute2, tcpreplay, and scapy to the behavioral model
docker build -t policysplit/bmv2 -f topology/Dockerfile.bmv2 topology

# Start the switch, replay captures through it, record telemetry and mitigation
docker run --rm --privileged -v "$PWD:/work" -w /work policysplit/bmv2 \
    python3 scripts/replay_attack.py --out results/bmv2_dataplane.json

# Controller, listening for telemetry reports
python controller/sdn_controller.py --model path/to/model.onnx
```

`scripts/replay_attack.py --collect --no-classifier` records the in-switch
statistics of every flow instead of running the per-capture trials, which is how
the source-diversity measurements are produced.

## Layout

```
p4/            P4_16 headers, parser, feature extractor, generated table entries
controller/    SDN controller, telemetry parser, ONNX inference, P4Runtime client
src/
  models/         BiMamba encoder, data-plane CNN, Mamba controller model
  preprocessing/  PCAP parser, context features, protocol.py, quantiser
  utils/          metrics and plotting helpers
scripts/       extraction, training, policy compilation, evaluation, replay
topology/      Mininet topology and the BMv2 runtime image
config/        YAML configurations
results/       one JSON record per run
```

Datasets, model checkpoints, and manuscript sources are not tracked; see
`.gitignore`.

## Class mapping

`src/preprocessing/feature_extractor.py` groups the 33 CICIoT2023 scenarios into
eight classes:

| Class      | Scenarios                                                        |
| ---------- | ---------------------------------------------------------------- |
| Benign     | BenignTraffic                                                    |
| DDoS       | SYN, UDP, TCP, ICMP, HTTP floods, fragmentation, SlowLoris       |
| DoS        | SYN, UDP, TCP, HTTP floods                                       |
| Mirai      | greeth_flood, greip_flood, udpplain                              |
| Recon      | HostDiscovery, OSScan, PingSweep, PortScan                       |
| Web        | XSS, SqlInjection, CommandInjection, Uploading, BrowserHijacking |
| Spoofing   | DNS_Spoofing, MITM-ArpSpoofing                                   |
| BruteForce | DictionaryBruteForce                                             |

## Citation

```bibtex
@article{padhya2026policysplit,
  title  = {PolicySplit: Policy-Aligned Fast-Path Admission for P4-SDN IoT
            Intrusion Detection},
  author = {Padhya, Dinanath and Sapkota, Binod and Dawadi, Babu R.},
  year   = {2026}
}
```

Please also cite the dataset: E. C. P. Neto, S. Dadkhah, R. Ferreira,
A. Zohourian, R. Lu, and A. A. Ghorbani, "CICIoT2023: A real-time dataset and
benchmark for large-scale attacks in IoT environment," Sensors, vol. 23, no. 13,
p. 5941, 2023.

## License

MIT. See [LICENSE](LICENSE).
