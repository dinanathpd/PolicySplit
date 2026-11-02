# PolicySplit

Policy-aligned fast-path admission for hybrid P4-SDN intrusion prevention in
IoT networks: the P4 pipeline, the compiler that builds its admission table,
the controller that adjudicates what the switch defers, the BMv2 and Tofino
model test beds, and the result records behind every number in the paper.

The switch detects on two paths. The per-flow path keeps five registers per
flow (byte count, running means of packet length and inter-arrival time, the
smallest and largest packet) and, when a flow's tenth packet arrives,
quantises them and looks the cell up in a match-action table. A hit applies
the entry's action (drop, rate limit through a per-flow meter, or redirect to
an inspection port); a miss defers the flow, which is forwarded and reported
to the controller as a 70-byte UDP datagram. The per-victim path counts the
distinct sources and the new flows that address each destination and reports
the surges of short flows that never reach a tenth packet.

The table is compiled, not hand written. `scripts/train_rf.py` maps the eight
traffic classes onto the actions an operator chooses, pools the training
evidence of every class that implies the same action, scores each cell by a
Wilson lower bound on that pooled share, and selects each action's threshold
on held-out flows with a bound that accounts for dependent flows. It never
emits an allow entry, so an uncertain cell defers to the controller, which
scores the report with a boosted-tree adjudicator
(`controller/report_adjudicator.py`, model in
`p4/generated/controller_model.json`) and installs mitigation entries.

## What is here

| Path | Contents |
| --- | --- |
| `p4/` | P4_16 program for BMv2 (`feature_extraction.p4`, `headers.p4`, `parsers.p4`), the compiled program and the generated tables under `p4/generated/`, and the Tofino Native Architecture port under `p4/tna/` with the compiler's resource logs |
| `scripts/train_rf.py` | the compiler and every admission study |
| `scripts/replay_attack.py` | the BMv2 test bed, the bit-exact emulator of the ingress pipeline, the closed loop with the controller, the capture replays, the new-flow alert check and the controller saturation study |
| `scripts/tofino_model.py`, `topology/tofino_model.sh` | control plane and packet jobs for the TNA port on Intel's Tofino 2 reference model |
| `scripts/replicate_unsw.py` | the admission study on UNSW-NB15 |
| `scripts/run_experiment.py`, `train_bimamba.py`, `train_cnn.py`, `benchmark.py` | the offline diagnostics of Appendix A (sequence model, estimator suite, CPU cost) |
| `scripts/download_pcap.py`, `pcap_to_entropy_fast.py`, `src/preprocessing/` | fetching CICIoT2023 and extracting the flow corpus |
| `scripts/plot_results.py` | regenerates every table, figure data file and quoted number of the manuscript from `results/` |
| `results/` | one JSON record per run; `confirm/` holds the confirmation seeds 60 to 89, `dev/` the development-seed record |

Datasets, extracted flow files, model checkpoints and replay traces are not
tracked (see `.gitignore`); the commands below regenerate them, and the
reviewer artifact that accompanies the submission ships them.

## Requirements

- Python 3.10 or newer; `pip install -r requirements.txt`
- Docker, for the P4 compiler and the BMv2 software switch
- A CUDA GPU only to retrain the sequence model of Appendix A
- Open P4 Studio (bf-p4c and the Tofino model) only for `p4/tna/`

Every command runs from the repository root with `PYTHONPATH=.`.

## Data

CICIoT2023 is distributed by the Canadian Institute for Cybersecurity and is
not redistributed here. Register at
[unb.ca/cic/datasets/iotdataset-2023.html](https://www.unb.ca/cic/datasets/iotdataset-2023.html),
then fetch the captures with your own session cookie (do not commit it):

```bash
export CIC_COOKIE="Token=..."
python scripts/download_pcap.py --cookie "$CIC_COOKIE" --pcap-dir dataset_cic_iot_2023/PCAP
python scripts/pcap_to_entropy_fast.py --pcap-dir dataset_cic_iot_2023/PCAP \
    --output data/sequences/entropy_flows_v2.h5 --workers 10 --window 10.0
```

The extractor labels a capture by the directory it sits in (the CIC layout),
keeps flows of at least ten packets, and writes one HDF5 file with the packet
series, labels and flow identifiers of every flow. `results/log_extraction_v3.txt`
is its log for the corpus of the paper.

Three further sets of captures are read as they are, not added to the corpus:

- The first 60 MB of the two benign captures the corpus does not read
  (`--file PCAP/Benign_Final/BenignTraffic2.pcap --file
  PCAP/Benign_Final/BenignTraffic3.pcap --head-mb 60`), extracted to
  `data/sequences/benign_unseen{2,3}.h5`.
- The first 60 MB of one capture per attack family, the capture heads of
  Section 6.5.4 (`--head-mb 60`; the file names are listed in
  `scripts/replay_attack.py`).
- The three benign IoT-23 captures (Philips Hue, Amazon Echo, Somfy door
  lock) from the [Stratosphere Laboratory](https://www.stratosphereips.org/datasets-iot23).
  Place each under a `Benign_Final/` directory and run the extractor on it to
  obtain `data/sequences/iot23_{hue,echo,somfy}.h5`.

UNSW-NB15 (`dataset_unsw_nb15/`) comes from
[UNSW Canberra](https://research.unsw.edu.au/projects/unsw-nb15-dataset).

## Compiling the admission table

```bash
python scripts/train_rf.py --seed 30                         # the deployed two-action table
python scripts/train_rf.py --seed 30 --deploy-map 4-action   # four-action table, p4/generated/tiered
```

This writes `p4/generated/classifier_entries.txt` (the entries), 
`p4/generated/bucketise.p4inc` (the quantiser the P4 program includes, so the
switch and the compiler share cut points) and the adjudicator
`p4/generated/controller_model.json`. `p4/generated/tiered_functional/` is the
four-action map compiled with `--deploy-calibration block`, whose rate-limit
and redirect entries exercise every switch action; `p4/generated/spoof_redirect/`
keeps the drop for every class but spoofing, which goes to inspection
(`--deploy-map spoof-redirect --tiered-out-dir p4/generated/spoof_redirect`).

## Reproducing the paper

Table, figure and section numbers refer to the revised manuscript. The
admission studies report seeds 30 to 59; seeds 0 to 29 were used for
development (`results/dev/`). Add `--study-first-seed 60` and an output path
under `results/confirm/` to repeat a study on the confirmation seeds 60 to 89.

| Result | Command | Record |
| --- | --- | --- |
| Action gate against label gate, richer maps, calibration (Tables 3, 5, 6; Figures 3, 5) | `scripts/train_rf.py --study` | `results/policy_study.json` |
| Flood-inspection and flood-rate-limit maps (Tables 5, 6) | `scripts/train_rf.py --study --study-maps flood-inspect flood-ratelimit --study-out results/policy_study_grouped.json` | `results/policy_study_grouped.json` |
| Binary, multi-task, cost-sensitive and tree rules (Table 4) | `scripts/train_rf.py --baselines` | `results/baselines_study.json` |
| Models fitted on the action labels (Section 6.2) | `scripts/train_rf.py --direct-action` | `results/direct_action_study.json` |
| Action-map study and the selection rule (Figure 4) | `scripts/train_rf.py --map-study` | `results/map_study.json` |
| Where each gate's calibration walk stops (Section 6.4) | `scripts/train_rf.py --calibration-walks` | `results/calibration_walks.json` |
| Switch plus controller (Table 10) | `scripts/train_rf.py --two-tier` | `results/two_tier_study.json` |
| Adjudicator over tree counts and leaf limits (Appendix C) | `scripts/train_rf.py --controller-sensitivity` | `results/controller_sensitivity.json` |
| Unseen benign captures, offline (Section 6.5.3) | `scripts/train_rf.py --benign-refresh` | `results/benign_refresh.json` |
| IoT-23 benign captures (Section 6.5.3) | `scripts/train_rf.py --benign-refresh --benign-captures data/sequences/iot23_echo.h5 data/sequences/iot23_hue.h5 data/sequences/iot23_somfy.h5 --benign-refresh-out results/benign_iot23.json` | `results/benign_iot23.json` |
| UNSW-NB15 admission, map study and context controls (Section 6.3, Table A2) | `scripts/replicate_unsw.py --policy`, `--map-study`, no flag | `results/unsw_nb15/` |
| Sequence-model grid and tuning (Appendix A) | `scripts/run_experiment.py --stage all`, `--tune 12` | `results/controlled/` |
| Estimator suite and CPU cost (Table A4) | `scripts/train_cnn.py`; `scripts/benchmark.py --device cpu --output results/benchmark_cpu.json` | `results/controlled/baselines.json`, `results/benchmark_cpu.json` |
| Timed trace, closed loop and load (Table 7) | see below | `results/bmv2_trace_replay.json`, `results/bmv2_closed_loop.json`, `results/online_reconciliation*.json`, `results/bmv2_meter_check.json` |
| Capture slices and unseen benign captures on BMv2 (Table 8, Section 6.5.3) | see below | `results/capture_replay*.json`, `results/capture_benign_unseen*.json`, `results/bmv2_*` |
| Capture heads and the new-flow alert (Table 9) | see below | `results/flood/`, `results/capture/heads/`, `results/bmv2/heads/` |
| Controller under load (Table 11) | see below | `results/controller_saturation.json`, `results/bmv2/saturation/`, `results/bmv2_saturation.json` |
| Source diversity in the switch (Table A3) | `scripts/replay_attack.py --collect --no-classifier [--interleave-seed N]` in the container | `results/causal/` |
| Tofino 2 port and model (Appendix D) | see below | `p4/tna/compiled/`, `results/tofino_model.json`, `results/tofino_model_trace.json` |

Then regenerate the tables, the figure data and the macros the manuscript
quotes:

```bash
python scripts/plot_results.py --tables out/tables --figures out/figures
```

## The data plane on BMv2

The P4 toolchain and the switch run in Docker.

```bash
docker run --rm -v "$PWD/p4:/w" -w /w p4lang/p4c \
    p4c-bm2-ss --target bmv2 --arch v1model \
    -o generated/feature_extraction.json \
    --p4runtime-files generated/feature_extraction.p4info.txtpb feature_extraction.p4
docker build -t policysplit/bmv2 -f topology/Dockerfile.bmv2 topology
```

`scripts/replay_attack.py` has host modes, which prepare traces and compare
records, and container modes, which drive the switch. The container is
started as

```bash
docker run --rm --privileged -v "$PWD:/work" -w /work policysplit/bmv2 \
    python3 scripts/replay_attack.py <container mode>
```

| Experiment | Host | Container |
| --- | --- | --- |
| Timed trace of held-out flows (Table 7) | `--build-trace`, then `--reconcile` | `--replay-trace`, and `--replay-trace --closed-loop` |
| Four-action table, functional | the same with `--entries p4/generated/tiered_functional/classifier_entries.txt --action-map 4-action` | |
| Rate-limit meter | `--meter-check --trace-out results/trace/meter_check.pcap` | `--replay-trace --trace results/trace/meter_check.pcap` |
| Capture slices (Table 8) | `--capture-trace --pcap dataset/PCAP_slices/*.pcap`, then `--capture-reconcile`; add `--min-gap-us 2000` for the paced copy | `--replay-trace --trace results/capture/slices_merged.pcap [--closed-loop]` |
| Unseen benign captures | `--capture-trace --pcap <BenignTraffic2> <BenignTraffic3>` | the same, with each table's entries |
| Capture heads at recorded timing (Table 9) | `--flood-check --pcap <heads> --out results/flood/heads.json`, the same over the benign captures to check the alert level, and `--capture-trace --pcap <head> --capture-out results/capture/heads/<head>.json` per head; for the live check add `--head-packets 30000 --min-gap-us 1000` | `--replay-trace --closed-loop --trace <head trace>` |
| Controller under load (Table 11) | `--saturation`; `--build-trace --trace-rate 100 --trace-cut-s 600 --trace results/trace/heldout_s30_r100.pcap --manifest results/trace/heldout_s30_r100.json`, then `--saturation-live --closed-loop-runs results/bmv2/saturation/*.json` | `--replay-trace --closed-loop --controller-queue 16 [--controller-rate-cap 30] --trace results/trace/heldout_s30_r100.pcap` |

The emulator reimplements the ingress state machine, including the slot
hashes, and every record compares it with the switch flow by flow. The
closed loop scores each deferred flow's report with the adjudicator and
installs entries through the switch's runtime interface; the controller
model, the queue and the report-rate cap are options of the same script.

## The Tofino 2 port

`p4/tna/policysplit.p4` is the same pipeline for the Tofino Native
Architecture. With Open P4 Studio installed (`SDE` pointing at its install
directory):

```bash
p4/tna/build.sh tofino2 tofino            # compiler logs under p4/tna/compiled/
sudo bash topology/tofino_model.sh up      # hugepages, veths, model, switchd, packet agent
python scripts/tofino_model.py setup && python scripts/tofino_model.py buckets
python scripts/tofino_model.py selftest    # results/tofino_model.json
python scripts/tofino_model.py run JOB.json && python scripts/tofino_model.py compare MODEL.json BMV2.json
sudo bash topology/tofino_model.sh down
```

The control commands need the SDE's BF-RT Python client (built for Python
3.10); `scripts/tofino_model.py --help` lists every command, and the file's
docstring states the interpreter and `PYTHONPATH` to use.

## Class mapping

`src/preprocessing/feature_extractor.py` groups the CICIoT2023 scenarios into
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
  title  = {Policy-aligned fast-path admission for hybrid {P4-SDN} intrusion
            prevention in {IoT} networks},
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
