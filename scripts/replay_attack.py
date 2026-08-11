"""
replay_attack.py - BMv2 data-plane test bed.

Runs inside the splitmamba/bmv2 container. Brings up a behavioral-model
switch with three virtual ports, installs the forwarding, classification
and mitigation state, replays real captures through the pipeline and
records what the switch actually did:

  * whether escalation reports are emitted and parse as well-formed UDP,
  * whether the per-flow statistics they carry match the offline
    definition of the same statistics,
  * the latency from the packet that triggers escalation to the report
    arriving on the controller port,
  * the effect of installing a mitigation rule, measured by the direct
    counter on the mitigation table.

Invoked from the host as:

    docker run --rm --privileged -v "$PWD:/work" -w /work \
        splitmamba/bmv2 python3 scripts/replay_attack.py \
            --pcap dataset/PCAP_slices/ddos_syn.pcap
"""

import argparse
import json
import logging
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

THRIFT_PORT = 9090
REPORT_PORT = 9999
REPORT_BYTES = 44
REPORT_FORMAT = "!IIIHHBBBBIIHHIIHH"
CLONE_SESSION = 100
MAX_FRAME = 1514

# port number -> (switch-side veth, test-side veth)
PORTS = {0: ("veth1", "veth0"), 1: ("veth3", "veth2"), 2: ("veth5", "veth4")}
CONTROLLER_PORT = 2

# Capture slice -> eight-class label, used only by the collection mode.
SLICE_LABEL = {
    "benign": 0,
    "ddos_syn": 1,
    "dos_tcp": 2,
    "mirai": 3,
    "recon": 4,
    "spoofing": 6,
    "spoofing_arp": 6,
    "spoofing_dns": 6,
    "bruteforce": 7,
}


def sh(cmd, check=True, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), check=check,
                          capture_output=True, text=True, **kw)


def setup_veths():
    for sw, host in PORTS.values():
        sh(f"ip link del {sw}", check=False)
    for sw, host in PORTS.values():
        sh(f"ip link add {host} type veth peer name {sw}")
        for i in (sw, host):
            sh(f"ip link set {i} up")
            sh(f"ip link set {i} mtu 9000")
            # Offloads reorder and coalesce frames, which would corrupt
            # the per-packet timing the switch measures.
            sh(f"ethtool -K {i} tx off rx off tso off gso off gro off",
               check=False)
            sh(f"sysctl -w net.ipv6.conf.{i}.disable_ipv6=1", check=False)
    logger.info("virtual ports up: %s",
                ", ".join(f"{p}@{sw}" for p, (sw, _) in PORTS.items()))


def start_switch(json_path, log_path):
    args = ["simple_switch", "--device-id", "0",
            "--thrift-port", str(THRIFT_PORT),
            "--log-file", log_path, "--log-level", "info", "--log-flush"]
    for p, (sw, _) in PORTS.items():
        args += ["-i", f"{p}@{sw}"]
    args.append(json_path)
    proc = subprocess.Popen(args, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    for _ in range(60):
        time.sleep(0.5)
        if sh(f"simple_switch_CLI --thrift-port {THRIFT_PORT} <<< 'help'",
              check=False, executable="/bin/bash").returncode == 0:
            logger.info("simple_switch ready (pid %d)", proc.pid)
            return proc
    raise RuntimeError("simple_switch did not come up")


def cli(commands):
    text = "\n".join(commands) + "\n"
    r = subprocess.run(["simple_switch_CLI", "--thrift-port",
                        str(THRIFT_PORT)], input=text,
                       capture_output=True, text=True)
    return r.stdout


def install_state(entries_file):
    cmds = [
        # Escalated clones leave on the controller port.
        f"mirroring_add {CLONE_SESSION} {CONTROLLER_PORT}",
        # Everything that is not mitigated is forwarded to port 1.
        "table_add SplitMambaIngress.ipv4_lpm forward 0.0.0.0/0 => 1",
    ]
    rules = [ln.strip() for ln in Path(entries_file).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    out = cli(cmds + rules)
    installed = out.count("Entry has been added")
    logger.info("control state installed: %d classifier entries, "
                "1 route, 1 mirroring session", installed - 1)
    return installed - 1


def read_counter(name, index):
    # BMv2 prints counters as "<name>[<i>]= (<n> bytes, <m> packets)".
    out = cli([f"counter_read {name} {index}"])
    m = re.search(r"(\d+)\s+packets", out)
    return int(m.group(1)) if m else 0


def parse_report(payload):
    """Decode the escalation report emitted by the egress pipeline."""
    import struct
    (flow_id, src, dst, sport, dport, proto, cls, conf, reason,
     pkts, byts, diversity, fanout, iat, mean_len, min_len,
     max_len) = struct.unpack(
        REPORT_FORMAT, payload[:REPORT_BYTES])
    return {"flow_id": flow_id, "src": src, "dst": dst,
            "src_port": sport, "dst_port": dport, "protocol": proto,
            "dp_class": cls, "dp_confidence": conf, "reason": reason,
            "pkt_count": pkts, "byte_count": byts,
            "src_diversity": diversity, "dstport_fanout": fanout,
            "mean_iat_us": iat, "mean_len": mean_len,
            "min_len": min_len, "max_len": max_len}


def run_trial(pcap, limit, rate):
    from scapy.all import AsyncSniffer, Ether, IP, UDP, PcapReader, conf

    sniffer = AsyncSniffer(iface=PORTS[CONTROLLER_PORT][1],
                           filter=f"udp port {REPORT_PORT}", store=True)
    sniffer.start()
    time.sleep(1.0)

    # A persistent layer-2 socket avoids re-opening a raw socket for every
    # frame, which otherwise dominates the offered rate.
    sock = conf.L2socket(iface=PORTS[0][1])
    sent, oversize, sent_at = 0, 0, {}
    t_start = time.time()
    with PcapReader(pcap) as reader:
        for pkt in reader:
            if IP not in pkt:
                continue
            frame = Ether(src="00:00:00:00:00:01",
                          dst="00:00:00:00:00:02") / pkt[IP]
            # Captures contain segments larger than the link MTU because
            # of host offloads at capture time; they cannot be replayed.
            if len(frame) > MAX_FRAME:
                oversize += 1
                continue
            sock.send(frame)
            sent += 1
            sent_at[sent] = time.time()
            if rate:
                time.sleep(1.0 / rate)
            if limit and sent >= limit:
                break
    sock.close()
    replay_seconds = time.time() - t_start

    time.sleep(2.0)
    captured = sniffer.stop()

    reports, latencies = [], []
    last_send = max(sent_at.values()) if sent_at else t_start
    for frame in captured:
        if UDP not in frame:
            continue
        payload = bytes(frame[UDP].payload)
        if len(payload) < REPORT_BYTES:
            continue
        reports.append(parse_report(payload))
        # A report is produced by the packet that triggered it, so the
        # nearest preceding transmission is its cause.
        arrival = float(frame.time)
        prior = [t for t in sent_at.values() if t <= arrival]
        if prior:
            latencies.append((arrival - max(prior)) * 1e3)

    return {
        "packets_sent": sent,
        "packets_skipped_oversize": oversize,
        "replay_seconds": round(replay_seconds, 3),
        "offered_pps": round(sent / replay_seconds, 1) if replay_seconds else 0,
        "reports_received": len(reports),
        "reports": reports[:20],
        "escalation_latency_ms": {
            "n": len(latencies),
            "median": round(sorted(latencies)[len(latencies) // 2], 3)
            if latencies else None,
            "p95": round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 3)
            if latencies else None,
            "mean": round(sum(latencies) / len(latencies), 3)
            if latencies else None,
        },
        "last_send_epoch": last_send,
    }


def _flow_key(src, sport, dst, dport, proto):
    """Direction-independent key matching the switch's canonical 5-tuple."""
    a, b = (str(src), int(sport)), (str(dst), int(dport))
    if a > b:
        a, b = b, a
    return (a[0], a[1], b[0], b[1], int(proto))


def run_collection(pcaps, labels, limit, rate):
    """
    Replay several captures interleaved and collect every escalation report.

    Interleaving matters. Replaying one capture at a time would leave the
    per-victim sketch observing a single attack family, which is the same
    condition that makes the offline windowed features a proxy for capture
    identity. Round-robin interleaving puts concurrent traffic from several
    families on the link, so the source-diversity estimate a flow receives
    is the one a switch would compute on a shared link.

    Returns the per-flow records, each carrying the statistics the switch
    itself measured and the label of the capture the flow came from.
    """
    from scapy.all import AsyncSniffer, Ether, IP, UDP, PcapReader, conf

    readers = {}
    for pcap in pcaps:
        name = Path(pcap).stem
        readers[name] = PcapReader(pcap)

    sniffer = AsyncSniffer(iface=PORTS[CONTROLLER_PORT][1],
                           filter=f"udp port {REPORT_PORT}", store=True)
    sniffer.start()
    time.sleep(1.0)

    sock = conf.L2socket(iface=PORTS[0][1])
    key_label, ambiguous = {}, set()
    sent, oversize, per_capture = 0, 0, {n: 0 for n in readers}
    # Ground truth the switch's sketch is trying to estimate, and the
    # premise the whole source-diversity argument rests on.
    victim_srcs = defaultdict(set)
    capture_srcs = defaultdict(set)
    true_div_at_send = {}
    t_start = time.time()

    active = list(readers)
    while active and (not limit or sent < limit):
        for name in list(active):
            try:
                pkt = next(readers[name])
            except StopIteration:
                active.remove(name)
                readers[name].close()
                continue
            if IP not in pkt:
                continue
            frame = Ether(src="00:00:00:00:00:01",
                          dst="00:00:00:00:00:02") / pkt[IP]
            if len(frame) > MAX_FRAME:
                oversize += 1
                continue

            ip = pkt[IP]
            sport = int(getattr(ip.payload, "sport", 0) or 0)
            dport = int(getattr(ip.payload, "dport", 0) or 0)
            key = _flow_key(ip.src, sport, ip.dst, dport, int(ip.proto))
            prior = key_label.get(key)
            if prior is not None and prior != labels[name]:
                ambiguous.add(key)
            key_label[key] = labels[name]

            victim_srcs[ip.dst].add(ip.src)
            capture_srcs[name].add(ip.src)
            true_div_at_send[key] = len(victim_srcs[ip.dst])

            sock.send(frame)
            sent += 1
            per_capture[name] += 1
            if rate:
                time.sleep(1.0 / rate)
            if limit and sent >= limit:
                break
    sock.close()
    for r in readers.values():
        try:
            r.close()
        except Exception:
            pass

    replay_seconds = time.time() - t_start
    time.sleep(3.0)
    captured = sniffer.stop()

    records, unmatched, dropped_ambiguous = [], 0, 0
    for frame in captured:
        if UDP not in frame:
            continue
        payload = bytes(frame[UDP].payload)
        if len(payload) < REPORT_BYTES:
            continue
        rep = parse_report(payload)
        src = ".".join(str((rep["src"] >> s) & 0xFF) for s in (24, 16, 8, 0))
        dst = ".".join(str((rep["dst"] >> s) & 0xFF) for s in (24, 16, 8, 0))
        key = _flow_key(src, rep["src_port"], dst, rep["dst_port"],
                        rep["protocol"])
        if key in ambiguous:
            dropped_ambiguous += 1
            continue
        label = key_label.get(key)
        if label is None:
            unmatched += 1
            continue
        rep["label"] = label
        rep["true_victim_sources"] = true_div_at_send.get(key, 0)
        records.append(rep)

    logger.info("collection: sent=%d reports=%d matched=%d unmatched=%d "
                "ambiguous=%d in %.1fs",
                sent, len(captured), len(records), unmatched,
                dropped_ambiguous, replay_seconds)

    return {
        "packets_sent": sent,
        "packets_skipped_oversize": oversize,
        "packets_per_capture": per_capture,
        "replay_seconds": round(replay_seconds, 3),
        "offered_pps": round(sent / replay_seconds, 1) if replay_seconds else 0,
        "reports_matched": len(records),
        "reports_unmatched": unmatched,
        "reports_ambiguous_key": dropped_ambiguous,
        # The premise the source-diversity argument rests on: how many
        # distinct addresses each capture actually contains.
        "distinct_sources_per_capture": {k: len(v)
                                         for k, v in capture_srcs.items()},
        "distinct_victims": len(victim_srcs),
        "records": records,
    }


def main():
    ap = argparse.ArgumentParser(description="BMv2 data-plane test bed")
    ap.add_argument("--json", default="p4/generated/feature_extraction.json")
    ap.add_argument("--entries", default="p4/generated/classifier_entries.txt")
    ap.add_argument("--pcap", nargs="+",
                    default=["dataset/PCAP_slices/ddos_syn.pcap"])
    ap.add_argument("--limit", type=int, default=3000,
                    help="Packets to replay per capture")
    ap.add_argument("--rate", type=float, default=2000.0,
                    help="Offered packets per second (0 = as fast as possible)")
    ap.add_argument("--out", default="results/bmv2_dataplane.json")
    ap.add_argument("--collect", action="store_true",
                    help="Interleave the captures and record the in-switch "
                         "features of every escalated flow instead of "
                         "running the per-capture trials")
    ap.add_argument("--no-classifier", action="store_true",
                    help="Install no classifier entries, so every flow with "
                         "enough packets escalates and is recorded")
    args = ap.parse_args()

    setup_veths()
    log_path = "/tmp/bmv2.log"
    proc = start_switch(args.json, log_path)

    try:
        n_entries = install_state(
            os.devnull if args.no_classifier else args.entries)

        if args.collect:
            labels = {Path(p).stem: SLICE_LABEL[Path(p).stem]
                      for p in args.pcap}
            result = run_collection(args.pcap, labels, args.limit, args.rate)
            result["environment"] = {
                "switch": sh("simple_switch --version").stdout.strip(),
                "target": "bmv2 v1model, simple_switch",
                "classifier_entries": n_entries,
                "note": "features are the values the switch itself computed "
                        "from packets that had already arrived",
            }
            result["labels"] = labels
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=2))
            logger.info("wrote %d labelled in-switch records to %s",
                        result["reports_matched"], args.out)
            return

        trials = {}
        for pcap in args.pcap:
            name = Path(pcap).stem
            logger.info("replaying %s", name)
            before = {"fwd": read_counter("pipeline_events", 0),
                      "drop": read_counter("pipeline_events", 1),
                      "esc": read_counter("pipeline_events", 2)}
            trial = run_trial(pcap, args.limit, args.rate)
            after = {"fwd": read_counter("pipeline_events", 0),
                     "drop": read_counter("pipeline_events", 1),
                     "esc": read_counter("pipeline_events", 2)}
            trial["pipeline_events"] = {
                k: after[k] - before[k] for k in before}
            trial["class_verdicts"] = [
                read_counter("class_verdicts", i) for i in range(8)]
            trials[name] = trial
            logger.info("  sent=%d reports=%d forwarded=%d dropped=%d "
                        "escalated=%d",
                        trial["packets_sent"], trial["reports_received"],
                        trial["pipeline_events"]["fwd"],
                        trial["pipeline_events"]["drop"],
                        trial["pipeline_events"]["esc"])

        # Mitigation: adjudicate the busiest reported source and install a
        # drop rule, then replay the same capture and read the direct
        # counter to confirm the rule is enforced in the pipeline.
        mitigation = {}
        first = next(iter(trials.values()))
        if first["reports"]:
            src = first["reports"][0]["src"]
            dotted = ".".join(str((src >> s) & 0xFF) for s in (24, 16, 8, 0))
            t0 = time.time()
            cli([f"table_add SplitMambaIngress.mitigation drop_packet "
                 f"{dotted} 0.0.0.0&&&0.0.0.0 0&&&0 => 10"])
            install_ms = (time.time() - t0) * 1e3
            replay = run_trial(args.pcap[0], args.limit, args.rate)
            hits = read_counter("SplitMambaIngress.mitigation_hits", 0)
            mitigation = {
                "blocked_source": dotted,
                "rule_install_ms": round(install_ms, 3),
                "packets_replayed_after_rule": replay["packets_sent"],
                "mitigation_hits_after_rule": hits,
                "reports_after_rule": replay["reports_received"],
            }
            logger.info("mitigation rule for %s installed in %.1f ms; "
                        "hits=%d, reports after rule: %d",
                        dotted, install_ms, hits, replay["reports_received"])

        result = {
            "environment": {
                "switch": sh("simple_switch --version").stdout.strip(),
                "target": "bmv2 v1model, simple_switch",
                "ports": {str(p): sw for p, (sw, _) in PORTS.items()},
                "note": "the behavioral model is functionally faithful but "
                        "is not a timing model of switching hardware",
            },
            "control_state": {
                "classifier_entries": n_entries,
                "clone_session": CLONE_SESSION,
                "controller_port": CONTROLLER_PORT,
            },
            "trials": trials,
            "mitigation": mitigation,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        logger.info("results written to %s", args.out)
    finally:
        proc.terminate()
        proc.wait(timeout=10)


if __name__ == "__main__":
    main()
