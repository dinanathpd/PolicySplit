"""
replay_attack.py - BMv2 data-plane test bed.

Runs inside the policysplit/bmv2 container. Brings up a behavioral-model
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
        policysplit/bmv2 python3 scripts/replay_attack.py \
            --pcap dataset/PCAP_slices/ddos_syn.pcap
"""

import argparse
import contextlib
import io
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

THRIFT_PORT = 9090
REPORT_PORT = 9999
REPORT_BYTES = 70
REPORT_FORMAT = "!IIIHHBBBBIIHHIIHHHIIIIIHBB"
CLONE_SESSION = 100
MAX_FRAME = 1514

# port number -> (switch-side veth, test-side veth)
PORTS = {0: ("veth1", "veth0"), 1: ("veth3", "veth2"), 2: ("veth5", "veth4"),
         3: ("veth7", "veth6")}
CONTROLLER_PORT = 2
INSPECT_PORT = 3
# Rate-limit action: per-flow two-rate meter, packets per microsecond and
# burst in packets (committed 50 packets/s, peak 100 packets/s).
METER_RATES = "0.00005:5 0.0001:10"
PIPELINE_EVENTS = ("fwd", "drop", "esc", "rate_limit_pass", "rate_limit_drop",
                   "redirect")

# Capture slice -> eight-class label, used only by the collection mode.
SLICE_LABEL = {
    "benign": 0,
    # Benign captures the corpus extractor never read.
    "BenignTraffic2": 0,
    "BenignTraffic3": 0,
    "ddos_syn": 1,
    "dos_tcp": 2,
    "mirai": 3,
    "recon": 4,
    "spoofing": 6,
    "spoofing_arp": 6,
    "spoofing_dns": 6,
    "bruteforce": 7,
    # 60 MB heads of one capture per attack family (download_pcap.py
    # --head-mb 60); the numbered files are ones the extractor never read.
    "DDoS-SYN_Flood1": 1,
    "DoS-TCP_Flood1": 2,
    "Mirai-udpplain1": 3,
    "Recon-PortScan": 4,
    "SqlInjection": 5,
    "VulnerabilityScan": 5,
    "DNS_Spoofing": 6,
    "MITM-ArpSpoofing1": 6,
    "DictionaryBruteForce": 7,
}
# Flow-length bins of the capture summaries, in packets.
LENGTH_BINS = (("1", 1, 1), ("2-9", 2, 9), ("10-31", 10, 31), ("32+", 32, None))


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


def start_switch(json_path, log_path, quiet=False):
    args = ["simple_switch", "--device-id", "0",
            "--thrift-port", str(THRIFT_PORT)]
    if not quiet:
        args += ["--log-file", log_path, "--log-level", "info", "--log-flush"]
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
        f"meter_array_set_rates flow_meter {METER_RATES}",
    ]
    rules = [ln.strip() for ln in Path(entries_file).read_text().splitlines()
             if ln.strip() and not ln.startswith("#")]
    out = cli(cmds + rules)
    installed = out.count("Entry has been added")
    logger.info("control state installed: %d classifier entries, "
                "1 route, 1 mirroring session, meters %s", installed - 1,
                METER_RATES)
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
     max_len, fwd_pkts, fwd_bytes, iat_min, iat_max, duration, len_dev,
     first_len, action, level) = struct.unpack(
        REPORT_FORMAT, payload[:REPORT_BYTES])
    return {"flow_id": flow_id, "src": src, "dst": dst,
            "src_port": sport, "dst_port": dport, "protocol": proto,
            "dp_class": cls, "dp_confidence": conf, "reason": reason,
            "pkt_count": pkts, "byte_count": byts,
            "src_diversity": diversity, "dstport_fanout": fanout,
            "mean_iat_us": iat, "mean_len": mean_len,
            "min_len": min_len, "max_len": max_len,
            "fwd_pkts": fwd_pkts, "fwd_bytes": fwd_bytes,
            "iat_min_us": iat_min, "iat_max_us": iat_max,
            "duration_us": duration, "len_dev": len_dev,
            "first_len": first_len, "dp_action": action,
            "alert_level": level}


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


def run_collection(pcaps, labels, limit, rate, interleave_seed=None):
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
    # A seed replaces strict round-robin with a random capture per packet.
    rng = None
    if interleave_seed is not None:
        import random
        rng = random.Random(interleave_seed)
    while active and (not limit or sent < limit):
        for name in ([rng.choice(active)] if rng else list(active)):
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


# ---------------------------------------------------------------------------
# Timing-faithful replay of held-out flows, and a bit-exact switch emulator.
# The trace is rebuilt from the stored per-flow packet series, so the switch
# sees the same flows, sizes and gaps the offline compiler scored; only
# interleaving, slot collisions and scheduling jitter differ.
# ---------------------------------------------------------------------------

FLOW_SLOTS = 16384
FLOW_TIMEOUT_US = 60_000_000
DECISION_PACKETS = 10
TS_BASE_US = 1_000_000          # a zero timestamp means "no previous packet"
# Per-victim new-flow counter of feature_extraction.p4: slot claims towards
# one destination address in a tumbling window, reported to the controller
# when the count reaches a power of two from FLOOD_LEVEL_MIN up.
VICTIM_SLOTS = 4096
NEW_FLOW_WINDOW_US = 1_000_000
FLOOD_LEVEL_MIN = 8             # log2: 256 new flows per window
FLOOD_LEVELS = frozenset(1 << b for b in range(FLOOD_LEVEL_MIN, 16))
FEATURE_ORDER = ("byte_bucket", "len_bucket", "iat_bucket", "minlen_bucket",
                 "maxlen_bucket")


def _host_imports():
    here = Path(__file__).resolve()
    for p in (str(here.parent), str(here.parents[1])):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import train_rf as T
    from src.preprocessing import protocol as P
    return np, T, P


def policy_tables(results_json, entries_file):
    """
    Cut points, key radix and admitted cells (cell -> action) of a compiled
    policy. Entries whose confidence byte is under the P4 guard (179) are
    left out, because the switch defers them.
    """
    q = json.loads(Path(results_json).read_text())["quantisation"]
    edges = [[int(c) for c in q["edges"][n]] for n in FEATURE_ORDER]
    radix = int(q.get("levels", 6))
    admitted = {}
    for ln in Path(entries_file).read_text().splitlines():
        if ln.startswith("table_add"):
            head, tail = ln.split("=>")
            idx = 0
            for tok in head.split()[3:8]:
                idx = idx * radix + int(tok.split("&&&")[0])
            vals = [int(v) for v in tail.split()]
            if vals[1] >= CONF_GUARD:
                admitted[idx] = vals[2] if len(vals) >= 4 else 1
    return edges, admitted, radix


CONF_GUARD = 179


def _cell(vals, edges, radix):
    idx = 0
    for v, cuts in zip(vals, edges):
        idx = idx * radix + sum(v >= c for c in cuts)
    return idx


def _orient(src, dst, sport, dport):
    return 1 if (src < dst or (src == dst and sport <= dport)) else 0


def _flow_hash(src, dst, sport, dport, proto):
    """The switch's CRC32 over the canonical 13-byte key (verified on BMv2)."""
    import struct
    import zlib
    if _orient(src, dst, sport, dport):
        b = struct.pack("!IIHHB", src, dst, sport, dport, proto)
    else:
        b = struct.pack("!IIHHB", dst, src, dport, sport, proto)
    return zlib.crc32(b)


def _flow_slots(src, dst, sport, dport, proto):
    """First slot, second slot (ports-first key order) and stored tag."""
    import struct
    import zlib
    if not _orient(src, dst, sport, dport):
        src, dst, sport, dport = dst, src, dport, sport
    h = zlib.crc32(struct.pack("!IIHHB", src, dst, sport, dport, proto))
    h2 = zlib.crc32(struct.pack("!HHIIB", sport, dport, src, dst, proto))
    return h % FLOW_SLOTS, h2 % FLOW_SLOTS, h % 0xFFFFFFFF


def _crc16(data):
    """CRC-16/ARC, the behavioral model's HashAlgorithm.crc16."""
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def _victim_slot(dst):
    """The switch's victim index: crc16 of the destination address."""
    import struct
    return _crc16(struct.pack("!I", dst)) % VICTIM_SLOTS


def held_out_flows(corpus, seed, max_packets):
    """Test flows of one seed's flow-isolated partition, as packet lists."""
    np, T, P = _host_imports()
    X_seq, X_ent, y, fid = P.load_corpus(corpus)
    keep = P.stratified_subsample(y, 12000, seed)
    _, _, te = P.split_indices(y[keep], fid[keep], X_ent[keep],
                               "flow-isolated", seed)
    idx = keep[te]
    X = X_seq[idx, :max_packets]
    valid = X[:, :, 0] != 0
    lens = np.rint(np.expm1(X[:, :, 0])).astype(np.int64)
    gaps = np.rint(np.expm1(X[:, :, 1]) * 1000).astype(np.int64)
    dirs = X[:, :, 2] > 0.5
    raw10 = T.switch_raw(X_seq[idx, :DECISION_PACKETS], X_ent[idx])
    flows = [(lens[i][valid[i]].tolist(), gaps[i][valid[i]].tolist(),
              dirs[i][valid[i]].tolist()) for i in range(len(idx))]
    return flows, y[idx], raw10


def schedule(flows, rate, rng_seed, keys=None):
    """Poisson flow arrivals, recorded gaps inside each flow; time-sorted."""
    import numpy as np
    rng = np.random.RandomState(rng_seed)
    starts = TS_BASE_US + np.cumsum(rng.exponential(1e6 / rate, len(flows)))
    if keys is None:
        keys = [(0x0A000000 + i, 0xAC100000 + i, 10000 + i % 50000, 443, 17)
                for i in range(len(flows))]
    events = []
    for i, (lens, gaps, dirs) in enumerate(flows):
        t = int(starts[i])
        for k, (L, g, d) in enumerate(zip(lens, gaps, dirs)):
            if k:
                t += g
            events.append((t, i, k, L, d))
    events.sort()
    return events, keys


def emulate(events, keys, edges, admitted, n_flows, radix, ways=2,
            alerts=None, claims=None, dropped=None, controller=None):
    """
    Replicate the ingress flow state and verdict logic of the P4 program.

    ``events`` are (time_us, flow, index, length, direction) with direction
    1 when the packet runs from the flow's initiator. ``ways`` 2 is the
    two-choice table of feature_extraction.p4; 1 is the earlier one-way
    index, kept for the load comparison. Verdicts: 0 undecided, 1 defer,
    2 drop, 3 rate-limit, 4 redirect. ``fields`` holds the twelve report
    registers at the decision.

    When ``alerts`` is a list, the per-victim new-flow counter is run on
    every slot claim and each flood alert the switch would report is
    appended as (time_us, flow, index, level, src, dst, dst_port). When
    ``claims`` is a list, every slot claim is appended as
    (time_us, flow, index, direction). When ``dropped`` is a list, the
    position in ``events`` of every packet the switch verdict acts on is
    appended. With a ``controller`` (EntryModel) the loop is closed: a
    packet an installed entry matches is stopped before it touches any
    state, flood alerts and deferred decisions are handed to the
    controller, and ``controller.mech`` records what stopped each packet.
    """
    S = FLOW_SLOTS
    slots = [_flow_slots(*k) for k in keys]
    korient = [_orient(*k[:4]) for k in keys]
    tag, pk, by, ts, ie, le, mn, mx, dec = ([0] * S for _ in range(9))
    io, fts, fl, fp, fb, imn, imx, ld = ([0] * S for _ in range(8))
    verdict = [0] * n_flows
    k_at = [-1] * n_flows
    fields = [None] * n_flows
    acted = [0] * n_flows
    nf_cnt, nf_ts, vslot = [0] * VICTIM_SLOTS, [0] * VICTIM_SLOTS, {}
    if controller is not None and alerts is None and controller.alerts_enabled:
        alerts = []
    for n, (t, i, k, L, d) in enumerate(events):
        src, dst, sp, dp, _ = keys[i]
        if not d:
            src, dst, sp, dp = dst, src, dp, sp
        if controller is not None:
            m = controller.hit(t, src, dst, dp)
            if m:
                controller.mech[n] = m
                continue
        sa, sb, fid = slots[i]
        if ways == 1 or tag[sa] == fid:
            s = sa
        elif tag[sb] == fid:
            s = sb
        elif tag[sa] == 0 or (ts[sa] and t - ts[sa] > FLOW_TIMEOUT_US):
            s = sa
        elif tag[sb] == 0 or (ts[sb] and t - ts[sb] > FLOW_TIMEOUT_US):
            s = sb
        else:
            s = sa if ts[sa] <= ts[sb] else sb
        o = korient[i] if d else 1 - korient[i]
        if tag[s] != fid or (ts[s] and t - ts[s] > FLOW_TIMEOUT_US):
            tag[s] = fid
            pk[s] = by[s] = ts[s] = ie[s] = le[s] = mn[s] = mx[s] = dec[s] = 0
            fp[s] = fb[s] = imn[s] = imx[s] = ld[s] = 0
            if claims is not None:
                claims.append((t, i, k, d))
            if alerts is not None:
                v = vslot.get(dst)
                if v is None:
                    v = vslot[dst] = _victim_slot(dst)
                if nf_ts[v] == 0 or t - nf_ts[v] > NEW_FLOW_WINDOW_US:
                    nf_cnt[v], nf_ts[v] = 0, t
                nf_cnt[v] += 1
                if nf_cnt[v] in FLOOD_LEVELS:
                    level = nf_cnt[v].bit_length() - 1
                    alerts.append((t, i, k, level, src, dst, dp))
                    if controller is not None:
                        controller.on_alert(t, i, level, src, dst, dp)
        pk[s] += 1
        by[s] += L
        if pk[s] == 1:
            io[s], fts[s], fl[s] = o, t, L
        if o == io[s]:
            fp[s] += 1
            fb[s] += L
        if mn[s] == 0 or L < mn[s]:
            mn[s] = L
        if L > mx[s]:
            mx[s] = L
        dv = abs(L - le[s])
        if pk[s] > 1:
            ld[s] = dv if ld[s] == 0 else ld[s] - (ld[s] >> 3) + (dv >> 3)
        le[s] = L if le[s] == 0 else le[s] - (le[s] >> 3) + (L >> 3)
        if ts[s]:
            dl = t - ts[s]
            ie[s] = dl if ie[s] == 0 else ie[s] - (ie[s] >> 3) + (dl >> 3)
            if pk[s] == 2 or dl < imn[s]:
                imn[s] = dl
            if dl > imx[s]:
                imx[s] = dl
        ts[s] = t
        if dec[s] == 0 and pk[s] >= DECISION_PACKETS:
            a = 0
            if pk[s] == DECISION_PACKETS:
                a = admitted.get(_cell((by[s], le[s], ie[s], mn[s], mx[s]),
                                       edges, radix), 0)
            dec[s] = a + 1 if a else 1
            if verdict[i] == 0:
                verdict[i], k_at[i] = dec[s], k
                fields[i] = (by[s], le[s], ie[s], mn[s], mx[s], fp[s], fb[s],
                             imn[s], imx[s], t - fts[s], ld[s], fl[s])
                if controller is not None and dec[s] == 1:
                    controller.on_defer(t, i, src, dst, sp, dp, fields[i])
        if dec[s] >= 2:
            acted[i] += 1
            if dropped is not None:
                dropped.append(n)
            if controller is not None:
                controller.mech[n] = 1
    return verdict, k_at, fields, acted


def _ipv4_checksum(hdr):
    s = sum(int.from_bytes(hdr[j:j + 2], "big") for j in range(0, 20, 2))
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def _frame(src, dst, sport, dport, length):
    import struct
    length = max(length, 28)
    ip = bytearray(struct.pack("!BBHHHBBHII", 0x45, 0, length, 0, 0, 64, 17,
                               0, src, dst))
    ip[10:12] = _ipv4_checksum(ip).to_bytes(2, "big")
    udp = struct.pack("!HHHH", sport, dport, length - 20, 0)
    return (b"\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x01\x08\x00"
            + bytes(ip) + udp + bytes(length - 28))


def write_pcap(path, events, keys):
    import struct
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for t, i, k, L, d in events:
            src, dst, sp, dp, _ = keys[i]
            fr = (_frame(src, dst, sp, dp, L) if d else
                  _frame(dst, src, dp, sp, L))
            f.write(struct.pack("<IIII", t // 1_000_000, t % 1_000_000,
                                len(fr), len(fr)))
            f.write(fr)


def read_pcap(path):
    import struct
    out = []
    with open(path, "rb") as f:
        f.read(24)
        while True:
            h = f.read(16)
            if len(h) < 16:
                break
            sec, usec, incl, _ = struct.unpack("<IIII", h)
            out.append((sec * 1_000_000 + usec, f.read(incl)))
    return out


def build_trace(args):
    """Write a pcap of held-out flows that fit the link MTU, and a manifest."""
    np, T, _ = _host_imports()
    edges, admitted, radix = policy_tables(args.policy_results, args.entries)
    flows, y, raw10 = held_out_flows(
        args.corpus, args.seed, args.trace_packets)
    offline = [admitted.get(_cell(r.astype(int).tolist(), edges, radix), 0)
               for r in raw10]
    ok = [i for i, f in enumerate(flows) if max(f[0]) <= MAX_FRAME - 14]
    rng = np.random.RandomState(args.seed)
    pick = sorted(rng.choice(
        ok, min(args.trace_flows, len(ok)), replace=False))
    sub = [flows[i] for i in pick]
    events, keys = schedule(sub, args.trace_rate, args.seed)
    if args.trace_cut_s:
        # Drop the idle tail: packets after the cut, which long flows with
        # minute-long gaps would otherwise stretch the replay with.
        limit = events[0][0] + int(args.trace_cut_s * 1e6)
        events = [e for e in events if e[0] <= limit]
    per_flow = [0] * len(sub)
    for _, j, _, _, _ in events:
        per_flow[j] += 1
    write_pcap(args.trace, events, keys)
    manifest = {"seed": args.seed, "rate_flows_per_s": args.trace_rate,
                "packets": len(events), "duration_s":
                (events[-1][0] - events[0][0]) / 1e6,
                "cut_s": args.trace_cut_s or None,
                "flows": [{"key": keys[j], "label": int(y[i]),
                           "offline_drop": bool(offline[i] == 1),
                           "offline_action": int(offline[i]),
                           "packets": per_flow[j]}
                          for j, i in enumerate(pick)]}
    Path(args.manifest).write_text(json.dumps(manifest))
    logger.info("trace: %d flows, %d packets, %.0f s, %d offline drops",
                len(pick), len(events), manifest["duration_s"],
                sum(f["offline_drop"] for f in manifest["flows"]))


def replay_trace(pcap, rate_scale=1.0, on_report=None):
    """Send a pcap on its own timestamps and collect every report."""
    import socket
    tx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
    tx.bind((PORTS[0][1], 0))
    rx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))
    rx.bind((PORTS[CONTROLLER_PORT][1], 0))
    rx.settimeout(0.5)
    reports, stop = [], threading.Event()

    def listen():
        while not stop.is_set():
            try:
                fr = rx.recv(65535)
            except socket.timeout:
                continue
            if len(fr) < 42 + REPORT_BYTES or fr[12:14] != b"\x08\x00" \
                    or fr[23] != 17:
                continue
            u = 14 + (fr[14] & 0x0F) * 4
            if int.from_bytes(fr[u + 2:u + 4], "big") != REPORT_PORT:
                continue
            rep = parse_report(fr[u + 8:u + 8 + REPORT_BYTES])
            rep["t"] = time.time()
            reports.append(rep)
            if on_report is not None:
                on_report(rep)

    th = threading.Thread(target=listen, daemon=True)
    th.start()
    packets = read_pcap(pcap)
    base = packets[0][0]
    t0 = time.perf_counter() + 1.0
    late = []
    for ts_us, fr in packets:
        target = t0 + (ts_us - base) / 1e6 / rate_scale
        while True:
            dt = target - time.perf_counter()
            if dt <= 0:
                break
            if dt > 0.002:
                time.sleep(dt - 0.001)
        late.append(-dt)
        tx.send(fr)
    time.sleep(3.0)
    stop.set()
    th.join()
    late.sort()
    return reports, {
        "packets_sent": len(packets),
        "send_lateness_ms": {"p50": round(1e3 * late[len(late) // 2], 3),
                             "p99": round(1e3 * late[int(.99 * (len(late) - 1))], 3),
                             "max": round(1e3 * late[-1], 3)}}


def _dotted(a):
    return ".".join(str((a >> s) & 0xFF) for s in (24, 16, 8, 0))


def start_frame_counter(iface):
    """Count frames leaving the switch on ``iface``; returns a reader."""
    import socket
    rx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, socket.ntohs(3))
    rx.bind((iface, 0))
    rx.settimeout(0.5)
    seen = {"frames": 0, "stop": False}

    def loop():
        while not seen["stop"]:
            try:
                rx.recv(65535)
                seen["frames"] += 1
            except socket.timeout:
                continue
    th = threading.Thread(target=loop, daemon=True)
    th.start()

    def read():
        seen["stop"] = True
        th.join()
        return seen["frames"]
    return read


class LiveController:
    """
    Scores each deferred flow's report and installs drop rules for blocks.

    Flood alerts (reason 4) are served ahead of waiting deferred reports.
    ``maxsize`` bounds the deferred reports waiting (0 is unbounded): a
    deferred report that arrives when that many are waiting is lost and
    counted. ``rate_cap`` slows the worker to at most that many reports per
    second, the device used to study the loop under a controller that
    cannot keep up.
    """

    def __init__(self, model_path, maxsize=0, rate_cap=None):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        sys.path.append("/root/p4-python-venv/lib/python3.12/site-packages")
        import runtime_CLI
        import sswitch_CLI
        from controller.report_adjudicator import ReportAdjudicator
        pre = runtime_CLI.PreType.SimplePreLAG
        services = runtime_CLI.RuntimeAPI.get_thrift_services(pre)
        services.extend(sswitch_CLI.SimpleSwitchAPI.get_thrift_services())
        std, mc, ssw = runtime_CLI.thrift_connect("localhost", THRIFT_PORT,
                                                  services)
        runtime_CLI.load_json_config(std, None)
        self.api = sswitch_CLI.SimpleSwitchAPI(pre, std, mc, ssw)
        self.adj = ReportAdjudicator(model_path)
        self.queue = queue.PriorityQueue()
        self.maxsize, self.rate_cap = maxsize, rate_cap
        self.seq = 0
        self.offered = self.lost = 0
        self.log = []
        self.flood_sources = defaultdict(set)
        self.victim_entries = set()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def _cmd(self, line):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.api.onecmd(line)
        return buf.getvalue()

    def _add(self, entry, src, dst, dport):
        """Install one drop entry; None wildcards a field."""
        s = f"{_dotted(src)}&&&0xFFFFFFFF" if src is not None else "0.0.0.0&&&0"
        d = f"{_dotted(dst)}&&&0xFFFFFFFF" if dst is not None else "0.0.0.0&&&0"
        p = f"{dport}&&&0xFFFF" if dport is not None else "0&&&0"
        out = self._cmd(f"table_add SplitMambaIngress.mitigation drop_packet "
                        f"{s} {d} {p} => 10")
        m = re.search(r"handle (\d+)", out)
        if m:
            entry["handles"].append(int(m.group(1)))

    def _work(self):
        while True:
            _, _, rep = self.queue.get()
            if rep is None:
                return
            t_start = time.time()
            entry = {k: rep[k] for k in ("flow_id", "src", "dst", "src_port",
                                         "dst_port", "t")}
            entry["t_start"] = t_start
            self._handle(rep, entry)
            self.log.append(entry)
            if self.rate_cap:
                time.sleep(max(0.0, 1.0 / self.rate_cap
                               - (time.time() - t_start)))

    def _handle(self, rep, entry):
        if rep["reason"] == 4:
            # Flood alert: block the named source towards the victim in
            # both directions; once more sources than FLOOD_SOURCES_MAX
            # have been named for one victim, block the victim's port
            # for every source.
            entry.update(kind="flood", level=rep["alert_level"],
                         score=None, block=True, handles=[])
            self._add(entry, rep["src"], rep["dst"], None)
            self._add(entry, rep["dst"], rep["src"], None)
            self.flood_sources[rep["dst"]].add(rep["src"])
            victim = (rep["dst"], rep["dst_port"])
            if len(self.flood_sources[rep["dst"]]) > FLOOD_SOURCES_MAX \
                    and victim not in self.victim_entries:
                self.victim_entries.add(victim)
                entry["kind"] = "victim"
                self._add(entry, None, rep["dst"], rep["dst_port"])
            entry["t_installed"] = time.time()
            return
        score = self.adj.score(rep)
        entry.update(kind="flow", score=score,
                     block=bool(self.adj.block(rep)), handles=[])
        if entry["block"]:
            # Both directions, so the whole flow stops at the switch.
            self._add(entry, rep["src"], rep["dst"], rep["dst_port"])
            self._add(entry, rep["dst"], rep["src"], rep["src_port"])
            entry["t_installed"] = time.time()

    def submit(self, rep):
        if rep["reason"] not in (1, 4):
            return
        self.offered += 1
        if rep["reason"] == 1 and self.maxsize \
                and self.queue.qsize() >= self.maxsize:
            self.lost += 1
            return
        self.seq += 1
        self.queue.put((0 if rep["reason"] == 4 else 1, self.seq, rep))

    def finish(self):
        self.seq += 1
        self.queue.put((2, self.seq, None))
        self.thread.join()
        for e in self.log:
            hits = 0
            for h in e["handles"]:
                m = re.search(r"(\d+)\s+packets", self._cmd(
                    f"counter_read SplitMambaIngress.mitigation_hits {h}"))
                hits += int(m.group(1)) if m else 0
            e["hits"] = hits
        stats = {"maxsize": self.maxsize, "rate_cap": self.rate_cap,
                 "offered": self.offered, "accepted": self.offered - self.lost,
                 "lost": self.lost}
        return self.log, stats


def run_trace_replay(args):
    """Container side: replay the rebuilt trace through BMv2."""
    setup_veths()
    proc = start_switch(args.json, "/dev/null", quiet=True)
    try:
        n_entries = install_state(args.entries)
        ctl = LiveController(args.controller_model, args.controller_queue,
                             args.controller_rate_cap) if args.closed_loop else None
        inspect_count = start_frame_counter(PORTS[INSPECT_PORT][1])
        reports, stats = replay_trace(
            args.trace, on_report=ctl.submit if ctl else None)
        controller_log, qstats = ctl.finish() if ctl else ([], None)
        counters = {k: read_counter("pipeline_events", i)
                    for i, k in enumerate(PIPELINE_EVENTS)}
        inspected = inspect_count() if inspect_count else None
        result = {"environment": {
            "switch": sh("simple_switch --version").stdout.strip(),
            "classifier_entries": n_entries, "logging": "off",
            "closed_loop": bool(ctl), "entries": args.entries,
            "meter_rates": METER_RATES},
            "replay": stats, "pipeline_events": counters,
            "frames_at_inspection_port": inspected,
            "reports": reports, "controller": controller_log,
            "controller_queue": qstats}
        Path(args.trace_out).write_text(json.dumps(result))
        logger.info("trace replay: %d reports, counters %s, lateness %s",
                    len(reports), counters, stats["send_lateness_ms"])
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def closed_loop_outcomes(args, index, flows, verdict, fields, ben, times=None,
                         directions=None):
    """Per-flow result of the live switch plus controller run."""
    from controller.report_adjudicator import REPORT_FEATURES, ReportAdjudicator
    adj = ReportAdjudicator(args.controller_model)
    cl = json.loads(Path(args.closed_loop_out).read_text())
    n = len(flows)
    sw = [0] * n
    for r in cl["reports"]:
        j = index.get(r["flow_id"])
        if j is not None and r["reason"] in (1, 2) and sw[j] == 0:
            sw[j] = r["reason"]
    ctl = [None] * n
    for e in cl["controller"]:
        j = index.get(e["flow_id"])
        if j is not None and ctl[j] is None:
            ctl[j] = e
    stop = [sw[j] == 2 or (ctl[j] is not None and ctl[j]["block"])
            for j in range(n)]
    # Expected: emulator switch verdict, controller scored on the registers
    # the emulator computed from recorded timing.
    expect = [verdict[j] == 2 or (verdict[j] == 1 and fields[j] is not None and
                                  adj.block(dict(zip(REPORT_FEATURES, fields[j]))))
              for j in range(n)]
    blocked = [e for e in ctl if e is not None and e["block"]]
    lat = sorted(1e3 * (e["t_installed"] - e["t"]) for e in blocked)
    post = [(max(flows[index[e["flow_id"]]]["packets"] - DECISION_PACKETS, 0),
             e["hits"]) for e in blocked]
    # Later packets sent before the rule existed, from the trace timing and
    # the measured report-to-install delay of each blocked flow; and packets
    # of OTHER flows that a (source, destination, destination port) rule
    # would also match once installed.
    before, collateral = None, None
    if times is not None and directions is not None:
        before = 0
        rules = {}
        for e in blocked:
            tt = times[index[e["flow_id"]]]
            wait_us = 1e6 * (e["t_installed"] - e["t"])
            if len(tt) > DECISION_PACKETS:
                t10 = tt[DECISION_PACKETS - 1]
                before += sum(t - t10 < wait_us for t in tt[DECISION_PACKETS:])
            if len(tt) >= DECISION_PACKETS:
                act = tt[DECISION_PACKETS - 1] + wait_us
                for key in ((e["src"], e["dst"], e["dst_port"]),
                            (e["dst"], e["src"], e["src_port"])):
                    rules[key] = min(rules.get(key, float("inf")), act)
        own = {index[e["flow_id"]] for e in blocked}
        hit_pk, hit_fl = {0: 0, 1: 0}, {0: set(), 1: set()}
        for j, f in enumerate(flows):
            if j in own:
                continue
            s, d, sp, dp, _ = f["key"]
            for t, fwd in zip(times[j], directions[j]):
                a = rules.get((s, d, dp) if fwd else (d, s, sp))
                if a is not None and t > a:
                    c = int(f["label"] != 0)
                    hit_pk[c] += 1
                    hit_fl[c].add(j)
        collateral = {"benign_packets": hit_pk[0], "benign_flows": len(hit_fl[0]),
                      "attack_packets": hit_pk[1], "attack_flows": len(hit_fl[1])}
    att = [not b for b in ben]

    def rate(mask, sel):
        k = [m for m, s in zip(mask, sel) if s]
        return sum(k) / len(k) if k else None
    res = {
        "flows": n, "benign_block_budget": json.loads(
            Path(args.controller_model).read_text())["benign_block_budget"],
        "switch_drop_flows": sum(v == 2 for v in sw),
        "deferred_flows": sum(v == 1 for v in sw),
        "controller_blocked_flows": len(blocked),
        "attack_stopped": rate(stop, att),
        "attack_stopped_at_switch": rate([v == 2 for v in sw], att),
        "benign_stopped": rate(stop, ben),
        "agreement_with_expected": sum(a == b for a, b in zip(stop, expect)) / n,
        "disagreements_on_flows_deferred_in_both": sum(
            a != b and sw[j] == 1 and verdict[j] == 1
            for j, (a, b) in enumerate(zip(stop, expect))),
        "disagreements": sum(a != b for a, b in zip(stop, expect)),
        "expected_attack_stopped": rate(expect, att),
        "expected_benign_stopped": rate(expect, ben),
        "rule_install_ms": {"p50": lat[len(lat) // 2], "p95": lat[int(.95 * (len(lat) - 1))],
                            "max": lat[-1]} if lat else None,
        "post_decision_packets_of_blocked_flows": sum(p for p, _ in post),
        "of_which_dropped_by_installed_rules": sum(h for _, h in post),
        "of_which_sent_before_install": before,
        "rule_matches_on_other_flows": collateral,
        "mitigation_entries": sum(len(e["handles"]) for e in blocked),
        "pipeline_events": cl["pipeline_events"], "replay": cl["replay"]}
    logger.info("closed loop: %s", {k: v for k, v in res.items()
                                    if k not in ("replay", "pipeline_events")})
    return res


def _action_rates(act, truth, n):
    """Fast-path coverage and error, overall and per action."""
    fast = [a > 0 for a in act]
    nf = sum(fast)
    per = {}
    for a in sorted({a for a in act if a > 0}):
        sel = [x == a for x in act]
        k = sum(sel)
        per[str(a)] = {"coverage": k / n,
                       "error": sum(s and t != a for s, t in zip(sel, truth)) / k}
    return {"coverage": nf / n, "fast_flows": nf,
            "selective_error": (sum(f and t != a for f, a, t in
                                    zip(fast, act, truth)) / nf) if nf else None,
            "per_action": per}


def _trace_events(trace, manifest):
    """
    Emulator events of a rebuilt held-out trace (UDP frames of build_trace),
    with the flow keys, the tag index, the manifest flows and each flow's
    packet times and directions.
    """
    man = json.loads(Path(manifest).read_text())
    flows = man["flows"]
    keys = [tuple(f["key"]) for f in flows]
    index = {_flow_hash(*k) % 0xFFFFFFFF: j for j, k in enumerate(keys)}
    events = []
    per_flow_k = [0] * len(flows)
    times = [[] for _ in flows]
    dirs = [[] for _ in flows]
    for ts_us, fr in read_pcap(trace):
        src = int.from_bytes(fr[26:30], "big")
        dst = int.from_bytes(fr[30:34], "big")
        sp = int.from_bytes(fr[34:36], "big")
        dp = int.from_bytes(fr[36:38], "big")
        j = index[_flow_hash(src, dst, sp, dp, 17) % 0xFFFFFFFF]
        L = int.from_bytes(fr[16:18], "big")
        events.append((ts_us, j, per_flow_k[j], L, int(src == keys[j][0])))
        times[j].append(ts_us)
        dirs[j].append(int(src == keys[j][0]))
        per_flow_k[j] += 1
    return man, flows, keys, index, events, times, dirs


def reconcile(args):
    """Offline verdicts against the emulator and against BMv2, plus a load sweep."""
    np, T, _ = _host_imports()
    from controller.report_adjudicator import REPORT_FEATURES
    amap = T.ACTION_MAPS[args.action_map]
    out = {}
    edges, admitted, radix = policy_tables(args.policy_results, args.entries)

    # 1. The same trace through the emulator and through BMv2.
    if Path(args.trace).exists() and Path(args.trace_out).exists():
        man, flows, keys, index, events, times, dirs = _trace_events(
            args.trace, args.manifest)
        verdict, k_at, fields, _ = emulate(events, keys, edges, admitted,
                                           len(flows), radix)
        bm = json.loads(Path(args.trace_out).read_text())
        bmv = [0] * len(flows)
        bmf = [None] * len(flows)
        for r in bm["reports"]:
            j = index.get(r["flow_id"])
            if j is not None and r["reason"] in (1, 2) and bmv[j] == 0:
                bmv[j] = 1 if r["reason"] == 1 else 1 + r["dp_action"]
                bmf[j] = tuple(r[f] for f in REPORT_FEATURES)
        n = len(flows)
        truth = [int(amap[f["label"]]) for f in flows]
        ben = [f["label"] == 0 for f in flows]
        off_act = [f.get("offline_action", int(f["offline_drop"]))
                   for f in flows]
        emu_act = [v - 1 if v >= 2 else 0 for v in verdict]
        bm_act = [v - 1 if v >= 2 else 0 for v in bmv]
        pairs = [(a, b) for a, b in zip(fields, bmf) if a and b]
        per_register = {}
        for i, name in enumerate(REPORT_FEATURES):
            d = sorted(abs(a[i] - b[i]) for a, b in pairs)
            per_register[name] = {
                "identical": sum(x == 0 for x in d) / len(d),
                "abs_diff_p50": d[len(d) // 2],
                "abs_diff_p99": d[int(.99 * (len(d) - 1))],
                "abs_diff_max": d[-1]} if d else None
        differences = [
            {"flow": j, "label": flows[j]["label"], "emulator": verdict[j],
             "bmv2": bmv[j],
             "cell_emulator": _cell(fields[j][:5], edges, radix) if fields[j] else None,
             "cell_bmv2": _cell(bmf[j][:5], edges, radix) if bmf[j] else None}
            for j in range(len(flows)) if verdict[j] != bmv[j]]
        out["bmv2_validation"] = {
            "flows": n, "packets": len(events), "rate_flows_per_s":
            man["rate_flows_per_s"], "duration_s": man["duration_s"],
            "action_map": args.action_map, "key_levels": radix,
            "offline": _action_rates(off_act, truth, n),
            "emulator": _action_rates(emu_act, truth, n),
            "bmv2": _action_rates(bm_act, truth, n),
            "verdict_agreement_bmv2_emulator": sum(
                a == b for a, b in zip(verdict, bmv)) / n,
            "action_agreement_bmv2_offline": sum(
                a == b for a, b in zip(bm_act, off_act)) / n,
            "decided_bmv2": sum(v > 0 for v in bmv) / n,
            "decided_emulator": sum(v > 0 for v in verdict) / n,
            "policy_registers_identical": sum(a[:5] == b[:5] for a, b in pairs)
            / len(pairs) if pairs else None,
            "report_registers_identical": sum(a == b for a, b in pairs)
            / len(pairs) if pairs else None,
            "report_cells_identical": sum(
                _cell(a[:5], edges, radix) == _cell(b[:5], edges, radix)
                for a, b in pairs) / len(pairs) if pairs else None,
            "per_register": per_register,
            "verdict_differences": differences,
            "replay": bm["replay"], "pipeline_events": bm["pipeline_events"],
            "frames_at_inspection_port": bm.get("frames_at_inspection_port"),
            "environment": bm["environment"]}
        logger.info("BMv2 vs emulator: %s", {k: v for k, v in
                    out["bmv2_validation"].items() if k not in
                    ("replay", "environment", "pipeline_events")})
        if args.action_map == "2-action" and Path(args.closed_loop_out).exists():
            out["closed_loop"] = closed_loop_outcomes(
                args, index, flows, verdict, fields, ben, times, dirs)

    # 2. Load sweep on the full held-out partition, jumbo frames included,
    # for the two-choice table and for the earlier one-way index.
    sweep = []
    for seed in args.sweep_seeds:
        res_json = (args.policy_results if seed == args.seed else
                    f"results/dataplane_policy_seed{seed}.json")
        ent = (args.entries if seed == args.seed else
               f"{args.sweep_entries_dir}/gen_s{seed}/classifier_entries.txt")
        e_s, a_s, r_s = policy_tables(res_json, ent)
        flows, y, raw10 = held_out_flows(args.corpus, seed, args.trace_packets)
        off = np.array([_cell(r.astype(int).tolist(), e_s, r_s) in a_s
                        for r in raw10])
        ben = y == 0
        for rate in args.sweep_rates:
            events, keys = schedule(flows, rate, seed)
            for ways in (2, 1):
                v, k_at, _, acted = emulate(events, keys, e_s, a_s, len(flows),
                                            r_s, ways)
                v = np.array(v)
                drop = v == 2
                sweep.append({
                    "seed": seed, "rate_flows_per_s": rate, "ways": ways,
                    "flows": len(flows),
                    "offline_coverage": float(off.mean()),
                    "offline_selective_drop_error": float(ben[off].mean()),
                    "online_coverage": float(drop.mean()),
                    "online_selective_drop_error": float(ben[drop].mean())
                    if drop.any() else None,
                    "verdict_agreement": float((drop == off).mean()),
                    "decided": float((v > 0).mean()),
                    "never_decided": float((v == 0).mean()),
                    "decided_after_eviction": float(
                        ((v > 0) & (np.array(k_at) != DECISION_PACKETS - 1)).mean()),
                    "attack_packets_dropped": int(np.array(acted)[~ben].sum()),
                    "benign_packets_dropped": int(np.array(acted)[ben].sum())})
                logger.info("sweep seed %d rate %d ways %d: %s", seed, rate, ways,
                            {k: round(x, 4) if isinstance(x, float) else x
                             for k, x in sweep[-1].items()})
    out["load_sweep"] = sweep
    Path(args.reconcile_out).write_text(json.dumps(out, indent=1))
    logger.info("written %s", args.reconcile_out)


def build_meter_check(args):
    """
    Functional test of the per-flow rate-limit meter. Every flow the tiered
    table rate-limits on the timed trace keeps its first DECISION_PACKETS
    packets exactly as recorded, so it reaches the same cell and action, and
    then sends ``--meter-packets`` copies of its decision packet at
    ``--meter-rate`` packets per second, faster than the configured peak.
    """
    import struct
    man = json.loads(Path(args.manifest).read_text())
    flows = man["flows"]
    keys = [tuple(f["key"]) for f in flows]
    index = {_flow_hash(*k) % 0xFFFFFFFF: j for j, k in enumerate(keys)}
    chosen = {j for j, f in enumerate(flows) if f.get("offline_action") == 2}
    first = {j: [] for j in chosen}
    for ts_us, fr in read_pcap(args.trace):
        src = int.from_bytes(fr[26:30], "big")
        dst = int.from_bytes(fr[30:34], "big")
        sp = int.from_bytes(fr[34:36], "big")
        dp = int.from_bytes(fr[36:38], "big")
        j = index[_flow_hash(src, dst, sp, dp, 17) % 0xFFFFFFFF]
        if j in chosen and len(first[j]) < DECISION_PACKETS:
            first[j].append((ts_us, fr))
    gap = int(round(1e6 / args.meter_rate))
    events = []
    # Flows start one second apart; each keeps its own recorded timing.
    for rank, (j, pk) in enumerate(sorted(first.items())):
        shift = TS_BASE_US + rank * 1_000_000 - pk[0][0]
        pk = [(t + shift, fr) for t, fr in pk]
        events += pk
        t_dec, fr_dec = pk[-1]
        events += [(t_dec + gap * (i + 1), fr_dec)
                   for i in range(args.meter_packets)]
    events.sort(key=lambda e: e[0])
    with open(args.trace_out, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for t, fr in events:
            fh.write(struct.pack("<IIII", t // 1_000_000, t % 1_000_000,
                                 len(fr), len(fr)))
            fh.write(fr)
    meta = {"source_trace": args.trace, "flows": len(chosen),
            "packets_per_flow_after_decision": args.meter_packets,
            "rate_pps": args.meter_rate, "meter_rates": METER_RATES,
            "packets": len(events)}
    Path(args.trace_out).with_suffix(".json").write_text(json.dumps(meta))
    logger.info("meter check trace: %s", meta)


def _corpus_key(src, dst, sport, dport, proto):
    """The extractor's MD5 flow key (pcap_to_entropy_fast._flow_key)."""
    import hashlib
    a, b = (_dotted(src), sport), (_dotted(dst), dport)
    if a > b:
        a, b = b, a
    return hashlib.md5(f"{a[0]}:{a[1]}-{b[0]}:{b[1]}-{proto}".encode()).hexdigest()


def _capture_packets(path):
    """(seconds, raw IPv4 bytes) of every IPv4 packet in a pcap or pcapng."""
    import dpkt
    out = []
    with open(path, "rb") as f:
        try:
            rd = dpkt.pcapng.Reader(f)
        except ValueError:
            f.seek(0)
            rd = dpkt.pcap.Reader(f)
        for ts, buf in rd:
            off, etype = 14, buf[12:14]
            if etype == b"\x81\x00":
                off, etype = 18, buf[16:18]
            if etype != b"\x08\x00" or len(buf) < off + 20 or buf[off] >> 4 != 4:
                continue
            out.append((float(ts), bytes(buf[off:])))
    return out


def _pcap_events(path):
    """
    Emulator events and flow keys of one capture, flows indexed in order
    of first appearance. Ports are read at the fixed 20-byte IPv4 offset,
    as the P4 parser does.
    """
    events, keys, index, per_k = [], [], {}, []
    t0 = None
    for ts, ip in _capture_packets(path):
        proto = ip[9]
        src = int.from_bytes(ip[12:16], "big")
        dst = int.from_bytes(ip[16:20], "big")
        sp = dp = 0
        if proto in (6, 17):
            ip = ip.ljust(24, b"\x00")
            sp = int.from_bytes(ip[20:22], "big")
            dp = int.from_bytes(ip[22:24], "big")
        canon = _flow_hash(src, dst, sp, dp, proto) % 0xFFFFFFFF
        j = index.get(canon)
        if j is None:
            j = index[canon] = len(keys)
            keys.append((src, dst, sp, dp, proto))
            per_k.append(0)
        if t0 is None:
            t0 = ts
        d = int((src, sp) == (keys[j][0], keys[j][2]))
        events.append((TS_BASE_US + int(round((ts - t0) * 1e6)), j, per_k[j],
                       int.from_bytes(ip[2:4], "big"), d))
        per_k[j] += 1
    return events, keys


def flood_check(args):
    """
    Run the per-victim new-flow counter over whole captures and record,
    per capture, the peak window count of any victim slot, the victims
    behind the highest counts and the flood alerts the switch would
    report. Used to set the alert level on benign traffic and to show
    which attack captures cross it.
    """
    edges, admitted, radix = policy_tables(args.policy_results, args.entries)
    out = {"window_us": NEW_FLOW_WINDOW_US, "level_min": 1 << FLOOD_LEVEL_MIN,
           "victim_slots": VICTIM_SLOTS, "flow_slots": FLOW_SLOTS,
           "captures": {}}
    for p in args.pcap:
        name = Path(p).stem
        events, keys = _pcap_events(p)
        alerts, claims = [], []
        emulate(events, keys, edges, admitted, len(keys), radix,
                alerts=alerts, claims=claims)
        # Window peaks per victim address and per victim slot (the slot
        # merges addresses that collide in the 4096-entry table).
        cnt, w0, peak = {}, {}, {}
        for t, i, k, d in claims:
            dst = keys[i][1] if d else keys[i][0]
            for key in (("addr", dst), ("slot", _victim_slot(dst))):
                if key not in w0 or t - w0[key] > NEW_FLOW_WINDOW_US:
                    cnt[key], w0[key] = 0, t
                cnt[key] += 1
                peak[key] = max(peak.get(key, 0), cnt[key])
        addr_peaks = sorted(((v, a) for (kind, a), v in peak.items()
                             if kind == "addr"), reverse=True)
        slot_peak = max((v for (kind, _), v in peak.items() if kind == "slot"),
                        default=0)
        span = (events[-1][0] - events[0][0]) / 1e6 if events else 0
        rec = {"packets": len(events), "flows": len(keys),
               "slot_claims": len(claims), "span_s": span,
               "victims": sum(1 for kind, _ in peak if kind == "addr"),
               "peak_new_flows_per_window": addr_peaks[0][0] if addr_peaks else 0,
               "peak_new_flows_per_window_slot": slot_peak,
               "top_victims": [{"addr": _dotted(a), "peak": v}
                               for v, a in addr_peaks[:5]],
               "alerts": len(alerts),
               "alert_levels": sorted({lvl for _, _, _, lvl, _, _, _ in alerts}),
               "first_alert_s": (alerts[0][0] - events[0][0]) / 1e6
               if alerts else None,
               "alert_victims": sorted({_dotted(dst)
                                        for *_, dst, _ in alerts}),
               "alert_sources": len({src for *_, src, _, _ in alerts})}
        out["captures"][name] = rec
        logger.info("%s: %d packets, %d flows, %d claims, peak %d new flows "
                    "per window (slot %d), %d alerts", name, len(events),
                    len(keys), len(claims), rec["peak_new_flows_per_window"],
                    slot_peak, len(alerts))
    Path(args.capture_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.capture_out).write_text(json.dumps(out, indent=1))


def build_capture_trace(args):
    """
    Merge the real capture slices on their recorded timestamps into one pcap.

    Every slice starts at the same instant, so the captures interleave with
    their own recorded packet timing. Every IPv4 packet is kept: payload
    beyond the link MTU is cut, but the IPv4 total length stays as captured,
    so the switch registers see true packet lengths. As in the P4 parser,
    ports are read at a fixed 20-byte IPv4 offset.
    """
    import hashlib
    np, T, P = _host_imports()
    eth = b"\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x01\x08\x00"
    events, digests, per_capture = [], {}, {}
    for p in args.pcap:
        name = Path(p).stem
        digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if digest in digests:
            logger.info("skip %s: identical to %s", name, digests[digest])
            continue
        digests[digest] = name
        pk = _capture_packets(p)
        if args.head_packets:
            pk = pk[:args.head_packets]
        per_capture[name] = len(pk)
        t0 = pk[0][0]
        for ts, ip in pk:
            proto = ip[9]
            src = int.from_bytes(ip[12:16], "big")
            dst = int.from_bytes(ip[16:20], "big")
            need = 20 + (20 if proto == 6 else 8 if proto == 17 else 0)
            ip = ip.ljust(need, b"\x00")
            sp = dp = 0
            if proto in (6, 17):
                sp = int.from_bytes(ip[20:22], "big")
                dp = int.from_bytes(ip[22:24], "big")
            total = int.from_bytes(ip[2:4], "big")
            events.append((TS_BASE_US + int(round((ts - t0) * 1e6)), name,
                           (src, dst, sp, dp, proto), total,
                           eth + ip[:MAX_FRAME - 14]))
    events.sort(key=lambda e: e[0])
    # Optionally space packets that follow each other more closely than a
    # software switch on virtual interfaces can absorb; order is kept and
    # the emulator scores the same paced trace.
    if args.min_gap_us:
        paced, t_prev = [], None
        for e in events:
            t = e[0] if t_prev is None else max(e[0], t_prev + args.min_gap_us)
            paced.append((t,) + e[1:])
            t_prev = t
        events = paced
    flows = {}
    for t, name, key, total, _ in events:
        canon = _flow_hash(*key) % 0xFFFFFFFF
        if canon not in flows:
            flows[canon] = {"key": list(key), "captures": set(), "packets": 0}
        flows[canon]["captures"].add(name)
        flows[canon]["packets"] += 1
    # Which flows did the deployed table's compiler see? Keys are the
    # extractor's MD5 of the canonical 5-tuple.
    _, X_ent, y, fid = P.load_corpus(args.corpus)
    keep = P.stratified_subsample(y, 12000, args.seed)
    tr, va, _ = P.split_indices(y[keep], fid[keep], X_ent[keep],
                                "flow-isolated", args.seed)
    fitted = set(fid[keep][np.concatenate([tr, va])].tolist())
    corpus = set(fid.tolist())
    man = []
    for canon, f in flows.items():
        caps = sorted(f["captures"])
        ck = _corpus_key(*f["key"])
        man.append({"flow_id": canon, "key": f["key"], "captures": caps,
                    "label": SLICE_LABEL[caps[0]] if len(caps) == 1 else None,
                    "packets": f["packets"], "in_corpus": ck in corpus,
                    "seen_by_compiler": ck in fitted})
    write = [(t, fr) for t, _, _, _, fr in events]
    import struct
    with open(args.trace, "wb") as fh:
        fh.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for t, fr in write:
            fh.write(struct.pack("<IIII", t // 1_000_000, t % 1_000_000,
                                 len(fr), len(fr)))
            fh.write(fr)
    Path(args.manifest).write_text(json.dumps({
        "captures": per_capture, "skipped_duplicates": [
            Path(p).stem for p in args.pcap if Path(p).stem not in per_capture],
        "packets": len(events), "duration_s":
        (events[-1][0] - events[0][0]) / 1e6, "seed": args.seed,
        "min_gap_us": args.min_gap_us,
        "flows": man}))
    logger.info("capture trace: %d packets, %d flows, %.1f s, captures %s",
                len(events), len(man), (events[-1][0] - events[0][0]) / 1e6,
                per_capture)


FLOOD_SOURCES_MAX = 16          # reported sources per victim before a victim-side entry


def _install_latency_us(path):
    """
    Median and 95th-percentile rule install latency measured on the BMv2
    closed loop (results/online_reconciliation.json), in microseconds.
    """
    try:
        ri = json.loads(Path(path).read_text())["closed_loop"]["rule_install_ms"]
        return {"p50": int(round(ri["p50"] * 1e3)),
                "p95": int(round(ri["p95"] * 1e3))}
    except (OSError, KeyError, ValueError):
        logger.warning("no measured install latency in %s; using 2 and 9 ms",
                       path)
        return {"p50": 2000, "p95": 9000}


class ServiceQueue:
    """
    One controller worker behind a bounded FIFO, in trace time.

    A report occupies the worker for the measured scoring time, or for the
    measured report-to-install time when it leads to entries. ``rate_cap``
    bounds the worker to that many reports per second, and a deferred
    report that finds ``depth`` reports waiting is lost. A priority report
    (a flood alert) is served ahead of the waiting reports, after the one
    in service, and is never lost; it delays the backlog by its own
    service, which the due times already handed out do not reflect (at
    most one alert service per alert). ``admit`` returns the time the
    report's entries take effect, or None when the report is lost.
    """

    def __init__(self, score_us, install_us, depth=None, rate_cap=None):
        from collections import deque
        self.score_us, self.install_us = score_us, install_us
        self.depth, self.rate_cap = depth, rate_cap
        self.busy_until = 0
        self.jobs = deque()            # (start, end) in start order
        self.prio = deque()            # (start, end) of alerts admitted
        self.offered = self.lost = 0
        self.arrivals = []
        self.waits = []
        self.busy_us = 0
        self.peak_queue = 0

    def admit(self, t, installs, priority=False):
        self.offered += 1
        self.arrivals.append(t)
        while self.jobs and self.jobs[0][1] <= t:
            self.jobs.popleft()
        in_service = bool(self.jobs) and self.jobs[0][0] <= t
        waiting = len(self.jobs) - int(in_service)
        self.peak_queue = max(self.peak_queue, waiting)
        service = self.install_us if installs else self.score_us
        if self.rate_cap:
            service = max(service, 1e6 / self.rate_cap)
        if priority:
            while self.prio and self.prio[0][1] <= t:
                self.prio.popleft()
            ahead = [s for s, e in self.prio if s > t]
            start = max(self.jobs[0][1] if in_service else t,
                        self.prio[-1][1] if self.prio else t)
            end = start + service
            self.jobs.insert(int(in_service) + len(ahead), (start, end))
            self.prio.append((start, end))
            self.busy_until = self.busy_until + service if waiting else end
        else:
            if self.depth is not None and waiting >= self.depth:
                self.lost += 1
                return None
            start = max(t, self.busy_until)
            end = start + service
            self.busy_until = end
            self.jobs.append((start, end))
        self.waits.append(start - t)
        self.busy_us += service
        return end

    def summary(self, span_us):
        w = sorted(self.waits)
        # Rates over the window in which reports arrived, not the trace span,
        # which long idle flows stretch far past the last report.
        a = self.arrivals
        win = (a[-1] - a[0]) if len(a) > 1 else span_us
        bins = defaultdict(int)
        for t in a:
            bins[t // 1_000_000] += 1
        per_s = sorted(bins.values())
        return {"depth": self.depth, "rate_cap": self.rate_cap,
                "score_us": self.score_us, "install_us": self.install_us,
                "offered": self.offered, "lost": self.lost,
                "lost_share": self.lost / self.offered if self.offered else None,
                "report_window_s": win / 1e6,
                "offered_per_s": self.offered / (win / 1e6) if win else None,
                "peak_per_s": per_s[-1] if per_s else 0,
                "p95_per_s": per_s[int(.95 * (len(per_s) - 1))] if per_s else 0,
                "busy_share": self.busy_us / win if win else None,
                "peak_queue": self.peak_queue,
                "wait_ms": {"p50": w[len(w) // 2] / 1e3,
                            "p95": w[int(.95 * (len(w) - 1))] / 1e3,
                            "max": w[-1] / 1e3} if w else None}


class EntryModel:
    """
    The controller's rule installation, driven from inside emulate().

    Entries take effect ``lat_us`` after the packet that caused the report,
    or at the completion time a ``queue`` (ServiceQueue) gives them. A
    deferred flow the adjudicator blocks gets the two flow entries the
    live controller installs. A flood alert names one source towards one
    victim: the controller blocks that source-victim pair in both
    directions and, once more than ``s_max`` distinct sources have been
    reported for one victim, blocks the victim's port for every source.
    Packets an entry drops touch no flow state and are not reported, as in
    the switch. ``mech`` records per packet what stopped it: 0 nothing,
    1 the switch verdict, 2 a flow entry, 3 a flood entry. With ``alerts``
    False the switch runs no new-flow counter.
    """

    def __init__(self, adj, lat_us, n_events, s_max=FLOOD_SOURCES_MAX,
                 queue=None, alerts=True):
        from controller.report_adjudicator import REPORT_FEATURES
        self.features = REPORT_FEATURES
        self.adj, self.lat, self.s_max = adj, lat_us, s_max
        self.queue, self.alerts_enabled = queue, alerts
        self.pair, self.victim, self.flow = {}, {}, {}
        self.sources = defaultdict(set)
        self.mech = [0] * n_events
        self.alerts = []
        self.blocked_flows = set()
        self.deferred = self.lost_deferred = self.lost_alerts = 0

    def hit(self, t, src, dst, dp):
        inf = 1 << 62
        if self.flow.get((src, dst, dp), inf) <= t:
            return 2
        if self.pair.get((src, dst), inf) <= t \
                or self.victim.get((dst, dp), inf) <= t:
            return 3
        return 0

    def _due(self, t, installs, priority=False):
        if self.queue is None:
            return t + self.lat
        return self.queue.admit(t, installs, priority)

    def on_alert(self, t, i, level, src, dst, dp):
        self.alerts.append((t, i, level, src, dst, dp))
        due = self._due(t, True, True)
        if due is None:
            self.lost_alerts += 1
            return
        for key in ((src, dst), (dst, src)):
            self.pair[key] = min(self.pair.get(key, due), due)
        self.sources[dst].add(src)
        if len(self.sources[dst]) > self.s_max:
            self.victim[(dst, dp)] = min(self.victim.get((dst, dp), due), due)

    def on_defer(self, t, i, src, dst, sp, dp, fields):
        self.deferred += 1
        block = self.adj.block(dict(zip(self.features, fields)))
        due = self._due(t, block)
        if due is None:
            self.lost_deferred += 1
            return
        if block:
            self.blocked_flows.add(i)
            for key in ((src, dst, dp), (dst, src, sp)):
                self.flow[key] = min(self.flow.get(key, due), due)

    def summary(self, t0):
        return {"latency_ms": self.lat / 1e3, "flood_alerts": len(self.alerts),
                "first_alert_s": (self.alerts[0][0] - t0) / 1e6
                if self.alerts else None,
                "alert_victims": sorted({_dotted(a[4]) for a in self.alerts}),
                "alert_sources": sorted({_dotted(a[3]) for a in self.alerts}),
                "alerts": [{"t_s": (a[0] - t0) / 1e6, "level": a[2],
                            "src": _dotted(a[3]), "dst": _dotted(a[4]),
                            "dst_port": a[5]} for a in self.alerts[:64]],
                "pair_entries": len(self.pair),
                "victim_entries": len(self.victim),
                "flow_entries": len(self.flow),
                "deferred_reports": self.deferred,
                "lost_deferred_reports": self.lost_deferred,
                "lost_alerts": self.lost_alerts}


def _time_scorer(adj, fields_list):
    """
    Microseconds the controller spends on one report: the 70-byte report is
    packed as the switch emits it, parsed, and scored by the deployed
    adjudicator, timed per report on this host. Median and 95th percentile
    over the given register tuples, after one warm-up pass.
    """
    import struct
    bufs = []
    for f in fields_list:
        by, le, ie, mn, mx, fp, fb, imn, imx, du, ld, fl = f
        m32, m16 = 0xFFFFFFFF, 0xFFFF
        bufs.append(struct.pack(
            REPORT_FORMAT, 0, 0, 0, 0, 0, 6, 255, 0, 1, DECISION_PACKETS,
            by & m32, 0, 0, ie & m32, le & m32, mn & m16, mx & m16, fp & m16,
            fb & m32, imn & m32, imx & m32, du & m32, ld & m32, fl & m16, 0, 0))
    times = []
    for rnd in range(2):
        times = []
        for b in bufs:
            t0 = time.perf_counter_ns()
            adj.block(parse_report(b))
            times.append((time.perf_counter_ns() - t0) / 1e3)
    times.sort()
    return {"reports": len(times), "p50": times[len(times) // 2],
            "p95": times[int(.95 * (len(times) - 1))], "max": times[-1]}


def _spoofed_flood(rng, victim, dport, rate, start_us, duration_s, n_keys):
    """
    Single-packet flows from distinct spoofed sources to one victim port:
    Poisson arrivals at ``rate`` flows per second for ``duration_s``.
    Returns events (flow indices from ``n_keys`` on) and their keys.
    """
    n = int(rate * duration_s)
    gaps = rng.exponential(1e6 / rate, n)
    t = start_us
    events, keys = [], []
    for k in range(n):
        t += gaps[k]
        src = ((int(rng.randint(1, 224)) << 24) | int(rng.randint(0, 1 << 24)))
        if src >> 24 in (10, 172, 192):
            src = (100 << 24) | (src & 0xFFFFFF)
        keys.append((src, victim, int(rng.randint(1024, 65536)), dport, 6))
        events.append((int(t), n_keys + k, 0, 44, 1))
    return events, keys


def saturation(args):
    """
    The controller under load, in the emulator with the loop closed.

    Held-out flows of the deployed seed arrive on a Poisson schedule and
    every report goes through a single controller worker behind a bounded
    queue whose service times are measured: the deployed scorer on the
    real serialised reports, and the BMv2 closed loop's report-to-install
    time for a report that leads to entries. Three studies: the worker
    capped at decreasing report rates on one trace, the measured worker
    at increasing flow arrival rates, and a spoofed-source flood of
    single-packet flows over the same background with the new-flow alert
    on and off. Per case: reports offered and lost, queueing delay, the
    attack and benign flows stopped, the packets of deferred flows that
    were forwarded, and for the flood the share of its packets stopped
    and the time to the victim entry.
    """
    np, T, _ = _host_imports()
    from controller.report_adjudicator import ReportAdjudicator
    edges, admitted, radix = policy_tables(args.policy_results, args.entries)
    adj = ReportAdjudicator(args.controller_model)
    flows, y, _ = held_out_flows(args.corpus, args.seed, args.trace_packets)
    labels = [int(v) for v in y]
    n = len(flows)
    lat = _install_latency_us(args.latency_results)
    base_events, base_keys = schedule(flows, args.sat_rate, args.seed)
    _, _, fields, _ = emulate(base_events, base_keys, edges, admitted, n, radix)
    scorer = _time_scorer(adj, [f for f in fields if f is not None])
    score_us, install_us = scorer["p50"], lat["p50"]
    logger.info("scorer %.0f us per report (p95 %.0f), install %d us",
                score_us, scorer["p95"], install_us)

    def run(events, keys, labs, depth, cap, alerts=True, flood_from=None,
            s_max=FLOOD_SOURCES_MAX):
        q = ServiceQueue(score_us, install_us, depth, cap)
        ctl = EntryModel(adj, install_us, len(events), s_max, queue=q,
                         alerts=alerts)
        verdict, k_at, fields, acted = emulate(events, keys, edges, admitted,
                                               len(keys), radix, controller=ctl)
        m = len(labs) if flood_from is None else flood_from
        att = [j for j in range(m) if labs[j] != 0]
        ben = [j for j in range(m) if labs[j] == 0]
        stopped = [verdict[j] >= 2 or j in ctl.blocked_flows for j in range(m)]
        span = events[-1][0] - events[0][0]
        pk = {"attack_forwarded": 0, "benign_dropped": 0,
              "deferred_forwarded_after_decision": 0,
              "flood": 0, "flood_pair": 0, "flood_victim": 0,
              "flood_forwarded": 0}
        for e_i, (t, j, k, L, d) in enumerate(events):
            mech = ctl.mech[e_i]
            if j >= m:
                pk["flood"] += 1
                if mech == 3:
                    if ctl.pair.get((keys[j][0], keys[j][1]), 1 << 62) <= t:
                        pk["flood_pair"] += 1
                    else:
                        pk["flood_victim"] += 1
                elif mech == 0:
                    pk["flood_forwarded"] += 1
                continue
            if mech == 0:
                if labs[j] != 0:
                    pk["attack_forwarded"] += 1
                if verdict[j] == 1 and k > k_at[j]:
                    pk["deferred_forwarded_after_decision"] += 1
            elif labs[j] == 0:
                pk["benign_dropped"] += 1
        res = {"flows": m, "span_s": span / 1e6,
               "decided": sum(v > 0 for v in verdict[:m]) / m,
               "deferred": sum(v == 1 for v in verdict[:m]),
               "controller_blocked": len(ctl.blocked_flows),
               "attack_stopped": sum(stopped[j] for j in att) / len(att),
               "attack_stopped_at_switch": sum(
                   verdict[j] >= 2 for j in att) / len(att),
               "attack_deferred_unmitigated": sum(
                   verdict[j] == 1 and j not in ctl.blocked_flows
                   for j in att) / len(att),
               "benign_blocked": sum(stopped[j] for j in ben) / len(ben),
               "packets": pk, "queue": q.summary(span),
               "controller": ctl.summary(events[0][0])}
        if flood_from is not None:
            t0 = min(e[0] for e in events if e[1] >= m)
            vic = [v for (d_, p_), v in ctl.victim.items()]
            res["flood"] = {
                "flows": len(keys) - m, "packets": pk["flood"],
                "stopped_share": (pk["flood_pair"] + pk["flood_victim"])
                / pk["flood"] if pk["flood"] else None,
                "stopped_by_pair_entries": pk["flood_pair"],
                "stopped_by_victim_entry": pk["flood_victim"],
                "forwarded": pk["flood_forwarded"],
                "alerts": len(ctl.alerts),
                "first_alert_s": (ctl.alerts[0][0] - t0) / 1e6 if ctl.alerts else None,
                "victim_entry_s": (min(vic) - t0) / 1e6 if vic else None,
                "pair_entries": len(ctl.pair)}
        return res

    out = {"seed": args.seed, "flows": n, "base_rate_flows_per_s": args.sat_rate,
           "scorer_us": scorer, "install_us": lat,
           "service_model": "score_us per report; install_us (BMv2 median "
                            "report-to-install) per report that installs "
                            "entries; rate cap as a minimum service time",
           "queue_depth": args.sat_depth, "rate_caps": {}, "flow_rates": {},
           "flow_rates_unbounded": {}, "depths": {}, "flood": {}}
    for cap in args.sat_caps:
        out["rate_caps"][str(cap)] = run(base_events, base_keys, labels,
                                        args.sat_depth, cap or None)
        logger.info("cap %s: %s", cap, json.dumps(
            {k: out["rate_caps"][str(cap)][k] for k in
             ("attack_stopped", "benign_blocked", "attack_deferred_unmitigated")}
            | {"lost": out["rate_caps"][str(cap)]["queue"]["lost_share"]}))
    for depth in args.sat_depths:
        out["depths"][str(depth)] = run(base_events, base_keys, labels,
                                       depth or None, None)
        logger.info("depth %s: lost %s peak %s", depth,
                    out["depths"][str(depth)]["queue"]["lost_share"],
                    out["depths"][str(depth)]["queue"]["peak_queue"])
    for rate in args.sat_flow_rates:
        ev, ks = schedule(flows, rate, args.seed)
        out["flow_rates"][str(rate)] = run(ev, ks, labels, args.sat_depth, None)
        out["flow_rates_unbounded"][str(rate)] = run(ev, ks, labels, None, None)
        logger.info("rate %s: %s", rate, json.dumps(
            {k: out["flow_rates"][str(rate)][k] for k in
             ("decided", "attack_stopped", "benign_blocked")}
            | {k: out["flow_rates"][str(rate)]["queue"][k] for k in
               ("peak_per_s", "lost_share", "busy_share")}
            | {"unbounded_wait95_ms": out["flow_rates_unbounded"][str(rate)]
               ["queue"]["wait_ms"]["p95"]}))
    rng = np.random.RandomState(args.seed + 7)
    victim, dport = 0xC0A80A0A, 80
    t_start = base_events[0][0] + int(args.sat_flood_start * 1e6)
    for frate in args.sat_flood_rates:
        fev, fkeys = _spoofed_flood(rng, victim, dport, frate, t_start,
                                    args.sat_flood_duration, n)
        events = sorted(base_events + fev)
        keys = base_keys + fkeys
        labs = labels + [1] * len(fkeys)
        for tag, alerts, cap, s_max in (
                ("alert", True, None, FLOOD_SOURCES_MAX),
                ("no_alert", False, None, FLOOD_SOURCES_MAX),
                ("alert_capped", True, args.sat_flood_cap, FLOOD_SOURCES_MAX),
                ("alert_sources8", True, None, 8),
                ("alert_sources4", True, None, 4)):
            r = run(events, keys, labs, args.sat_depth, cap, alerts, n, s_max)
            r["sources_max"] = s_max
            out["flood"][f"{frate:g}_{tag}"] = r
            logger.info("flood %g/s %s: %s", frate, tag, json.dumps(r["flood"]))
    Path(args.capture_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.capture_out).write_text(json.dumps(out, indent=1))


def saturation_live(args):
    """
    Closed-loop BMv2 runs of one rebuilt trace with the controller capped
    at different report rates, each against the emulator's expectation
    under the same cap and queue bound (ServiceQueue with the measured
    service times).
    """
    import copy
    from controller.report_adjudicator import ReportAdjudicator
    edges, admitted, radix = policy_tables(args.policy_results, args.entries)
    adj = ReportAdjudicator(args.controller_model)
    man, flows, keys, index, events, times, dirs = _trace_events(
        args.trace, args.manifest)
    verdict, k_at, fields, _ = emulate(events, keys, edges, admitted,
                                       len(flows), radix)
    ben = [f["label"] == 0 for f in flows]
    att = [j for j in range(len(flows)) if not ben[j]]
    lat = _install_latency_us(args.latency_results)
    scorer = _time_scorer(adj, [f for f in fields if f is not None])
    out = {"trace": {k: man[k] for k in ("seed", "rate_flows_per_s", "packets",
                                         "duration_s")},
           "flows": len(flows), "scorer_us": scorer, "install_us": lat,
           "runs": {}}
    for path in args.closed_loop_runs:
        a = copy.copy(args)
        a.closed_loop_out = path
        res = closed_loop_outcomes(a, index, flows, verdict, fields, ben,
                                   times, dirs)
        raw = json.loads(Path(path).read_text())
        qs = raw.get("controller_queue") or {}
        res["controller_queue"] = qs
        blocked = {index[e["flow_id"]] for e in raw["controller"]
                   if e["block"] and e["flow_id"] in index}
        deferred = [index[r["flow_id"]] for r in raw["reports"]
                    if r["reason"] == 1 and r["flow_id"] in index]
        unblocked = [j for j in set(deferred) if j not in blocked]
        res["deferred_unblocked_flows"] = len(unblocked)
        res["deferred_unblocked_packets_after_decision"] = sum(
            max(flows[j]["packets"] - DECISION_PACKETS, 0) for j in unblocked)
        waits = sorted((e["t_start"] - e["t"]) * 1e3 for e in raw["controller"]
                       if "t_start" in e)
        res["queue_wait_ms"] = {"p50": waits[len(waits) // 2],
                                "p95": waits[int(.95 * (len(waits) - 1))],
                                "max": waits[-1]} if waits else None
        # The emulator under the same cap and bound.
        q = ServiceQueue(scorer["p50"], lat["p50"],
                         qs.get("maxsize") or None, qs.get("rate_cap"))
        ctl = EntryModel(adj, lat["p50"], len(events), queue=q)
        v2, _, _, _ = emulate(events, keys, edges, admitted, len(flows), radix,
                              controller=ctl)
        stop = [v2[j] >= 2 or j in ctl.blocked_flows for j in range(len(flows))]
        res["emulator_under_cap"] = {
            "attack_stopped": sum(stop[j] for j in att) / len(att),
            "benign_stopped": sum(stop[j] for j in range(len(flows))
                                  if ben[j]) / sum(ben),
            "controller_blocked_flows": len(ctl.blocked_flows),
            "reports_lost_share": q.summary(events[-1][0] - events[0][0])["lost_share"],
            "wait_ms": q.summary(events[-1][0] - events[0][0])["wait_ms"]}
        out["runs"][Path(path).stem] = res
    Path(args.capture_out).write_text(json.dumps(out, indent=1))
    logger.info("saturation live: %s", {k: (v["attack_stopped"], v["benign_stopped"],
                                            v["controller_queue"])
                                        for k, v in out["runs"].items()})


def capture_reconcile(args):
    """Per-capture outcome of the recorded-timing replay: emulator and BMv2."""
    np, T, _ = _host_imports()
    from controller.report_adjudicator import REPORT_FEATURES, ReportAdjudicator
    edges, admitted, radix = policy_tables(args.policy_results, args.entries)
    man = json.loads(Path(args.manifest).read_text())
    flows = man["flows"]
    index = {f["flow_id"]: j for j, f in enumerate(flows)}
    keys = [tuple(f["key"]) for f in flows]
    events, per_k = [], [0] * len(flows)
    for ts_us, fr in read_pcap(args.trace):
        ip = fr[14:]
        proto = ip[9]
        src = int.from_bytes(ip[12:16], "big")
        dst = int.from_bytes(ip[16:20], "big")
        sp = dp = 0
        if proto in (6, 17):
            sp = int.from_bytes(ip[20:22], "big")
            dp = int.from_bytes(ip[22:24], "big")
        j = index[_flow_hash(src, dst, sp, dp, proto) % 0xFFFFFFFF]
        d = int((src, sp) == (keys[j][0], keys[j][2]))
        events.append((ts_us, j, per_k[j], int.from_bytes(ip[2:4], "big"), d))
        per_k[j] += 1
    alerts = []
    verdict, k_at, fields, acted = emulate(events, keys, edges, admitted,
                                           len(flows), radix, alerts=alerts)
    # Per flow: bytes, bytes from the decision packet on, and the time span
    # of its packets, for the operational summary of each capture.
    f_bytes = [0] * len(flows)
    acted_bytes = [0] * len(flows)
    t_first = [None] * len(flows)
    t_last = [0] * len(flows)
    for t, j, k, L, _ in events:
        f_bytes[j] += L
        if verdict[j] >= 2 and k >= k_at[j]:
            acted_bytes[j] += L
        if t_first[j] is None:
            t_first[j] = t
        t_last[j] = t
    adj = ReportAdjudicator(args.controller_model)
    # The controller's entries replayed over the trace at the measured
    # median and 95th-percentile install latency of the BMv2 closed loop.
    lat = _install_latency_us(args.latency_results)
    models = {}
    for tag, lat_us in lat.items():
        ctl = EntryModel(adj, lat_us, len(events))
        emulate(events, keys, edges, admitted, len(flows), radix,
                controller=ctl)
        m = {"ctl": ctl, "latency_us": lat_us,
             "pk": [[0] * len(flows) for _ in range(4)],
             "by": [[0] * len(flows) for _ in range(4)], "fwd": [],
             "alert_flows": defaultdict(list)}
        for n, (t, j, k, L, _) in enumerate(events):
            m["pk"][ctl.mech[n]][j] += 1
            m["by"][ctl.mech[n]][j] += L
            if ctl.mech[n] == 0:
                m["fwd"].append((t, j))
        for t, i, lvl, src, dst, dp in ctl.alerts:
            m["alert_flows"][i].append((t, lvl, src, dst, dp))
        models[tag] = m
    runs = {}
    for tag, path in (("open_loop", args.trace_out),
                      ("closed_loop", args.closed_loop_out)):
        if not Path(path).exists():
            continue
        bm = json.loads(Path(path).read_text())
        bmv = [0] * len(flows)
        regs = [None] * len(flows)
        for r in bm["reports"]:
            j = index.get(r["flow_id"])
            if j is not None and r["reason"] in (1, 2) and bmv[j] == 0:
                bmv[j] = 1 if r["reason"] == 1 else 1 + r["dp_action"]
                regs[j] = tuple(r[f] for f in REPORT_FEATURES)
        blocked = set()
        for e in bm.get("controller", []):
            j = index.get(e["flow_id"])
            if j is not None and e["block"]:
                blocked.add(j)
        runs[tag] = {"verdict": bmv, "blocked": blocked, "raw": bm,
                     "regs": regs}
    # A flow whose eight timing-independent registers match the emulator's
    # at the decision point reached it with the same ten packets.
    static = [i for i, n in enumerate(REPORT_FEATURES) if n not in
              ("mean_iat_us", "iat_min_us", "iat_max_us", "duration_us")]

    def summarise(sel):
        idx = [j for j in range(len(flows)) if sel(flows[j])]
        dec = [j for j in idx if flows[j]["packets"] >= DECISION_PACKETS]
        row = {"flows": len(idx), "flows_reaching_decision": len(dec),
               "packets": sum(flows[j]["packets"] for j in idx),
               "packets_in_decidable_flows": sum(flows[j]["packets"] for j in dec),
               "emulator_fast": sum(verdict[j] >= 2 for j in idx),
               "emulator_deferred": sum(verdict[j] == 1 for j in idx),
               "emulator_fast_packets": sum(acted[j] for j in idx),
               "emulator_controller_blocks": sum(
                   verdict[j] == 1 and fields[j] is not None and
                   adj.block(dict(zip(REPORT_FEATURES, fields[j]))) for j in idx)}
        # Operational view: where the packets and bytes sit by flow length,
        # what the fast path removes after the decision, and the report rate
        # the controller sees over the span of these flows.
        row["bytes"] = sum(f_bytes[j] for j in idx)
        row["length_bins"] = {}
        for name, lo, hi in LENGTH_BINS:
            b = [j for j in idx if flows[j]["packets"] >= lo
                 and (hi is None or flows[j]["packets"] <= hi)]
            row["length_bins"][name] = {
                "flows": len(b), "packets": sum(flows[j]["packets"] for j in b),
                "bytes": sum(f_bytes[j] for j in b)}
        for lab, key in ((0, "benign"), (1, "attack")):
            grp = [j for j in idx if (flows[j]["label"] != 0) == bool(lab)]
            fast = [j for j in grp if verdict[j] >= 2]
            row[f"{key}_flows"] = len(grp)
            row[f"{key}_packets"] = sum(flows[j]["packets"] for j in grp)
            row[f"{key}_bytes"] = sum(f_bytes[j] for j in grp)
            row[f"emulator_fast_{key}_flows"] = len(fast)
            row[f"emulator_fast_{key}_packets"] = sum(acted[j] for j in fast)
            row[f"emulator_fast_{key}_bytes"] = sum(acted_bytes[j] for j in fast)
        starts = [t_first[j] for j in idx if t_first[j] is not None]
        span = (max(t_last[j] for j in idx) - min(starts)) / 1e6 if starts else 0
        reports = sum(verdict[j] > 0 for j in idx)
        row["span_s"] = span
        row["emulator_reports"] = reports
        row["emulator_reports_per_s"] = reports / span if span else None
        # Switch verdicts plus controller entries (flow blocks and flood
        # alerts) replayed at the measured install latency.
        t0 = min(starts) if starts else 0
        in_sel = set(idx)
        row["entries"] = {}
        for tag, m in models.items():
            al = [a for j in idx for a in m["alert_flows"].get(j, [])]
            gate = min(a[0] for a in al) + m["latency_us"] if al else None
            pk = {name: sum(m["pk"][c][j] for j in idx)
                  for c, name in ((1, "switch"), (2, "flow_entries"),
                                  (3, "flood_entries"))}
            by = {name: sum(m["by"][c][j] for j in idx)
                  for c, name in ((1, "switch"), (2, "flow_entries"),
                                  (3, "flood_entries"))}
            pk["total"] = sum(pk.values())
            by["total"] = sum(by.values())
            att = [j for j in idx if flows[j]["label"] != 0]
            ben = [j for j in idx if flows[j]["label"] == 0]
            att_pk = sum(flows[j]["packets"] for j in att)
            row["entries"][tag] = {
                "latency_ms": m["latency_us"] / 1e3,
                "flood_alerts": len(al),
                "flood_alert_levels": sorted({a[1] for a in al}),
                "first_alert_s": (min(a[0] for a in al) - t0) / 1e6 if al else None,
                "alert_victims": len({a[3] for a in al}),
                "alert_sources": len({a[2] for a in al}),
                "dropped_packets": pk, "dropped_bytes": by,
                "dropped_share": pk["total"] / row["packets"]
                if row["packets"] else None,
                "attack_dropped_packets": sum(
                    m["pk"][c][j] for c in (1, 2, 3) for j in att),
                "attack_dropped_share": sum(
                    m["pk"][c][j] for c in (1, 2, 3) for j in att) / att_pk
                if att_pk else None,
                "benign_dropped_packets": sum(
                    m["pk"][c][j] for c in (1, 2, 3) for j in ben),
                "benign_flood_entry_packets": sum(m["pk"][3][j] for j in ben),
                "passed_after_first_entry": sum(
                    1 for t, j in m["fwd"] if t >= gate and j in in_sel)
                if gate is not None else None}
        for tag, run in runs.items():
            v = run["verdict"]
            same = [j for j in dec if run["regs"][j] is not None
                    and fields[j] is not None
                    and all(run["regs"][j][i] == fields[j][i] for i in static)]
            row[tag] = {"fast": sum(v[j] >= 2 for j in idx),
                        "deferred": sum(v[j] == 1 for j in idx),
                        "decidable_flows_decided": sum(v[j] > 0 for j in dec),
                        "decidable_flows_same_ten_packets": len(same),
                        "agreement_same_ten_packets": sum(
                            v[j] == verdict[j] for j in same) / len(same)
                        if same else None,
                        "agreement_with_emulator": sum(
                            v[j] == verdict[j] for j in idx) / len(idx) if idx else None,
                        "agreement_on_decidable_flows": sum(
                            v[j] == verdict[j] for j in dec) / len(dec) if dec else None,
                        "controller_blocked": sum(j in run["blocked"] for j in idx)}
        return row

    clean = [f for f in flows if f["label"] is not None]
    out = {"trace": {k: man[k] for k in ("captures", "skipped_duplicates",
                                         "packets", "duration_s")},
           "min_gap_us": man.get("min_gap_us", 0),
           "oversized_packets": sum(e[3] > 1500 for e in events),
           "ambiguous_flows": len(flows) - len(clean),
           "key_levels": radix, "per_capture": {}, "all": {},
           "flood_alert": {"window_us": NEW_FLOW_WINDOW_US,
                           "level_min": 1 << FLOOD_LEVEL_MIN,
                           "sources_max": FLOOD_SOURCES_MAX,
                           "alerts_emitted_by_switch_alone": len(alerts)},
           "entries_model": {tag: m["ctl"].summary(events[0][0])
                             for tag, m in models.items()}}
    for cap in man["captures"]:
        out["per_capture"][cap] = summarise(
            lambda f, c=cap: f["label"] is not None and f["captures"] == [c])
    for name, sel in (("all", lambda f: f["label"] is not None),
                      ("not_seen_by_compiler", lambda f: f["label"] is not None
                       and not f["seen_by_compiler"]),
                      ("not_in_corpus", lambda f: f["label"] is not None
                       and not f["in_corpus"])):
        out["all"][name] = summarise(sel)
        # Selective error on the decided flows of this subset.
        idx = [j for j in range(len(flows)) if sel(flows[j])]
        fast = [j for j in idx if verdict[j] >= 2]
        out["all"][name]["emulator_fast_benign"] = sum(
            flows[j]["label"] == 0 for j in fast)
        # Remote endpoints of the benign flows the table drops, by address
        # outside the testbed subnet (the testbed side varies by device).
        out["all"][name]["emulator_fast_benign_remote"] = sorted({
            _dotted(k) for j in fast if flows[j]["label"] == 0
            for k in flows[j]["key"][:2]
            if not _dotted(k).startswith("192.168.")})
        # What the loop would do on the emulator's registers, for the flows
        # the emulator decides: the expectation the live run is compared to.
        e_dec = [j for j in idx if verdict[j] > 0]
        e_stop = {j for j in e_dec if verdict[j] >= 2 or (
            fields[j] is not None
            and adj.block(dict(zip(REPORT_FEATURES, fields[j]))))}
        for lab, key in ((0, "benign"), (1, "attack")):
            grp = [j for j in e_dec if (flows[j]["label"] != 0) == bool(lab)]
            out["all"][name][f"emulator_decided_{key}_flows"] = len(grp)
            out["all"][name][f"emulator_decided_{key}_stopped"] = (
                sum(j in e_stop for j in grp) / len(grp) if grp else None)
        for tag, run in runs.items():
            fb = [j for j in idx if run["verdict"][j] >= 2]
            stop = [j for j in idx if run["verdict"]
                    [j] >= 2 or j in run["blocked"]]
            dec = [j for j in idx if run["verdict"][j] > 0]
            att = [j for j in dec if flows[j]["label"] != 0]
            ben = [j for j in dec if flows[j]["label"] == 0]
            out["all"][name][tag].update({
                "fast_benign": sum(flows[j]["label"] == 0 for j in fb),
                "decided_attack_flows": len(att),
                "decided_benign_flows": len(ben),
                "decided_attack_stopped": sum(j in stop for j in att) / len(att)
                if att else None,
                "decided_benign_stopped": sum(j in stop for j in ben) / len(ben)
                if ben else None})
    for tag, run in runs.items():
        log = run["raw"].get("controller", [])
        kinds = sorted({e.get("kind", "flow") for e in log})
        out[tag] = {"pipeline_events": run["raw"]["pipeline_events"],
                    "pipeline_packets": sum(run["raw"]["pipeline_events"].values()),
                    "replay": run["raw"]["replay"],
                    "reports": len(run["raw"]["reports"]),
                    "diversity_alerts": sum(r["reason"] == 3
                                            for r in run["raw"]["reports"]),
                    "diversity_alert_victims": [
                        _dotted(r["dst"]) for r in run["raw"]["reports"]
                        if r["reason"] == 3],
                    "flood_alerts": sum(r["reason"] == 4
                                        for r in run["raw"]["reports"]),
                    "flood_alert_sources": sorted({
                        _dotted(r["src"]) for r in run["raw"]["reports"]
                        if r["reason"] == 4}),
                    "entries_by_kind": {
                        k: sum(len(e["handles"]) for e in log
                               if e.get("kind", "flow") == k) for k in kinds},
                    "mitigation_hits_by_kind": {
                        k: sum(e.get("hits", 0) for e in log
                               if e.get("kind", "flow") == k) for k in kinds},
                    "install_ms": sorted(
                        round((e["t_installed"] - e["t"]) * 1e3, 2)
                        for e in log if "t_installed" in e)}
    Path(args.capture_out).write_text(json.dumps(out, indent=1))
    logger.info("capture replay: %s", json.dumps(out["all"]["all"]))


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
    ap.add_argument("--build-trace", action="store_true",
                    help="Host: rebuild held-out flows as a timed pcap")
    ap.add_argument("--replay-trace", action="store_true",
                    help="Container: replay the timed pcap through BMv2")
    ap.add_argument("--reconcile", action="store_true",
                    help="Host: compare offline, emulator and BMv2 verdicts")
    ap.add_argument("--corpus", default="data/sequences/entropy_flows_v2.h5")
    ap.add_argument("--seed", type=int, default=30,
                    help="Seed the deployed table was compiled from")
    ap.add_argument("--policy-results",
                    default="results/dataplane_classifier.json")
    ap.add_argument("--trace", default="results/trace/heldout_s30.pcap")
    ap.add_argument("--manifest", default="results/trace/heldout_s30.json")
    ap.add_argument("--trace-out", default="results/bmv2_trace_replay.json")
    ap.add_argument("--closed-loop", action="store_true",
                    help="Adjudicate deferred flows live and install drop rules")
    ap.add_argument("--controller-queue", type=int, default=0,
                    help="Closed loop: bound on the report queue (0 = none)")
    ap.add_argument("--controller-rate-cap", type=float, default=None,
                    help="Closed loop: reports per second the controller accepts")
    ap.add_argument("--controller-model",
                    default="p4/generated/controller_model.json")
    ap.add_argument("--latency-results",
                    default="results/online_reconciliation.json",
                    help="Closed-loop run whose measured rule install "
                         "latency the entry model replays")
    ap.add_argument("--closed-loop-out",
                    default="results/bmv2_closed_loop.json")
    ap.add_argument("--interleave-seed", type=int, default=None,
                    help="Collection mode: random interleaving with this seed")
    ap.add_argument("--trace-flows", type=int, default=3000)
    ap.add_argument("--trace-rate", type=float, default=20.0,
                    help="Flow arrivals per second in the rebuilt trace")
    ap.add_argument("--trace-packets", type=int, default=100)
    ap.add_argument("--trace-cut-s", type=float, default=0,
                    help="Build trace: drop packets after this many seconds")
    ap.add_argument("--sweep-seeds", type=int, nargs="+", default=[30, 31, 32])
    ap.add_argument("--sweep-rates", type=float, nargs="+",
                    default=[10, 100, 1000, 5000])
    ap.add_argument("--sweep-entries-dir", default="results/trace")
    ap.add_argument("--reconcile-out",
                    default="results/online_reconciliation.json")
    ap.add_argument("--action-map", default="2-action",
                    help="Action map of the installed entries, for scoring")
    ap.add_argument("--capture-trace", action="store_true",
                    help="Host: merge the capture slices on recorded time")
    ap.add_argument("--capture-reconcile", action="store_true",
                    help="Host: per-capture outcome of the capture replay")
    ap.add_argument("--flood-check", action="store_true",
                    help="Host: new-flow counter peaks and flood alerts of "
                         "whole captures")
    ap.add_argument("--saturation", action="store_true",
                    help="Host: controller under load in the closed-loop "
                         "emulator (rate caps, flow rates, spoofed flood)")
    ap.add_argument("--sat-rate", type=float, default=100.0,
                    help="Saturation: flow arrivals per second of the base trace")
    ap.add_argument("--sat-depth", type=int, default=16,
                    help="Saturation: report queue depth")
    ap.add_argument("--sat-depths", type=int, nargs="+",
                    default=[1, 4, 16, 64, 256, 0],
                    help="Saturation: queue depths for the measured worker "
                         "(0 = unbounded)")
    ap.add_argument("--sat-caps", type=float, nargs="+",
                    default=[0, 1000, 300, 100, 50, 30, 20, 10, 5, 2],
                    help="Saturation: controller report-rate caps (0 = none)")
    ap.add_argument("--sat-flow-rates", type=float, nargs="+",
                    default=[20, 50, 100, 200, 500, 1000, 2000, 5000, 10000,
                             20000],
                    help="Saturation: flow arrival rates for the measured worker")
    ap.add_argument("--sat-flood-rates", type=float, nargs="+",
                    default=[2000, 20000],
                    help="Saturation: spoofed-source flood rates (flows per second)")
    ap.add_argument("--sat-flood-start", type=float, default=20.0,
                    help="Saturation: seconds into the base trace the flood starts")
    ap.add_argument("--sat-flood-duration", type=float, default=10.0)
    ap.add_argument("--sat-flood-cap", type=float, default=2.0,
                    help="Saturation: report-rate cap of the capped flood case")
    ap.add_argument("--saturation-live", action="store_true",
                    help="Host: compare capped closed-loop BMv2 runs of one trace")
    ap.add_argument("--closed-loop-runs", nargs="+", default=[],
                    help="Saturation live: closed-loop result files to compare")
    ap.add_argument("--capture-out", default="results/capture_replay.json")
    ap.add_argument("--min-gap-us", type=int, default=0,
                    help="Capture trace: space packets at least this far apart")
    ap.add_argument("--head-packets", type=int, default=0,
                    help="Capture trace: keep only the first N packets of "
                         "each capture")
    ap.add_argument("--rate-check", type=float, nargs="+", default=None,
                    help="Host: write the first --limit packets of --trace at "
                         "each of these constant rates (packets per second)")
    ap.add_argument("--meter-check", action="store_true",
                    help="Host: functional test trace for the rate-limit meter")
    ap.add_argument("--meter-packets", type=int, default=200,
                    help="Packets each rate-limited flow sends after its decision")
    ap.add_argument("--meter-rate", type=float, default=1000.0,
                    help="Packets per second of those later packets")
    args = ap.parse_args()

    if args.meter_check:
        build_meter_check(args)
        return
    if args.rate_check:
        import struct
        pk = read_pcap(args.trace)[:args.limit]
        out_dir = Path(args.trace_out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rate in args.rate_check:
            gap = int(round(1e6 / rate))
            with open(out_dir / f"rate{rate:g}.pcap", "wb") as fh:
                fh.write(struct.pack(
                    "<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
                for i, (_, fr) in enumerate(pk):
                    t = TS_BASE_US + i * gap
                    fh.write(struct.pack("<IIII", t // 1_000_000, t % 1_000_000,
                                         len(fr), len(fr)))
                    fh.write(fr)
        logger.info("rate check traces: %d packets at %s packets/s in %s",
                    len(pk), args.rate_check, out_dir)
        return

    if args.capture_trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        build_capture_trace(args)
        return
    if args.capture_reconcile:
        capture_reconcile(args)
        return
    if args.flood_check:
        flood_check(args)
        return
    if args.saturation:
        saturation(args)
        return
    if args.saturation_live:
        saturation_live(args)
        return

    if args.build_trace:
        Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
        build_trace(args)
        return
    if args.reconcile:
        reconcile(args)
        return
    if args.replay_trace:
        run_trace_replay(args)
        return

    setup_veths()
    log_path = "/tmp/bmv2.log"
    proc = start_switch(args.json, log_path)

    try:
        n_entries = install_state(
            os.devnull if args.no_classifier else args.entries)

        if args.collect:
            labels = {Path(p).stem: SLICE_LABEL[Path(p).stem]
                      for p in args.pcap}
            result = run_collection(args.pcap, labels, args.limit, args.rate,
                                    args.interleave_seed)
            result["interleave_seed"] = args.interleave_seed
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
                 f"{dotted}&&&0xFFFFFFFF 0.0.0.0&&&0.0.0.0 0&&&0 => 10"])
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
