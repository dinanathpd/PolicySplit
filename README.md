# PolicySplit

Research code for **Policy-aligned fast-path admission for hybrid P4-SDN
intrusion prevention in IoT networks**.

A P4 switch applies an admission table compiled from the operator's action
map. Uncertain flows are reported to an SDN controller, and per-victim
alerts report surges of short flows that do not reach the decision packet.

## Repository

| Path | Contents |
| --- | --- |
| `p4/` | BMv2 program, generated admission tables and Tofino 2 port |
| `controller/` | Report decoding and controller decisions |
| `scripts/` | Data preparation, experiments, replay and plotting |
| `src/` | Feature extraction and model implementations |
| `topology/` | Switch test-bed setup |
| `results/` | Saved experiment records |

## Setup

Use Python 3.10 or newer. Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH=.
```

Switch replay requires Docker and BMv2. The Tofino model requires Open P4
Studio. A CUDA GPU is needed only for retraining the sequence models.

## Data

Download the datasets from their original sources:

- [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html)
- [IoT-23](https://www.stratosphereips.org/datasets-iot23)
- [UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

Datasets, checkpoints and replay traces are not included in this repository.
To prepare the main flow dataset from downloaded CICIoT2023 captures:

```bash
python scripts/pcap_to_entropy_fast.py --pcap-dir dataset_cic_iot_2023/PCAP \
    --output data/sequences/entropy_flows_v2.h5 --workers 1
```

## Main commands

Compile the deployed admission table and run the admission study:

```bash
python scripts/train_rf.py --seed 30
python scripts/train_rf.py --study
```

Regenerate tables and plot data from the saved results:

```bash
python scripts/plot_results.py --tables out/tables --figures out/figures
```

For replay options and additional experiments, use the relevant script's help:

```bash
python scripts/train_rf.py --help
python scripts/replay_attack.py --help
```

## Citation and license

See [CITATION.cff](CITATION.cff) for citation details and [LICENSE](LICENSE)
for the MIT license. Please also cite any datasets used.
