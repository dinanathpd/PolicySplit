#!/usr/bin/env python3
"""
Tofino 2 model test bed for the TNA program (p4/tna/policysplit.p4).

The compiled binary runs on Intel's reference model (tofino-model) under
bf_switchd; this script is its control plane over BF-RT gRPC and the driver
of packet jobs. Root is needed only for the model itself and for raw packet
I/O, which topology/tofino_model.sh and the ``agent`` command provide.

Control commands need the SDE's BF-RT client, which is built for Python
3.10: run them with a Python 3.10 interpreter and
PYTHONPATH=$SDE/lib/python3.10/site-packages:$SDE/lib/python3.10/site-packages/tofino
(the bfshell that ships with the SDE cannot be used here, because its
bfrt_python plugin needs a libpython the switch daemon does not find).
  setup            ports, mirror session to the controller port, default route
  buckets          quantisation cut points of p4/generated/bucketise.p4inc as
                   ternary prefix entries (inter-arrival cuts in 256 ns units)
  classifier FILE  admission entries of a BMv2 entries file, with the verdict
                   the controller derives (confidence guard 179)
  verdict B1..B5 CLS CONF ACT   one admission entry on a bucket key
  mitigate SRC[/M] DST[/M] DPORT[/M] [PRIO]   one mitigation entry
  meter CIR CBS PIR PBS          the per-flow meter, packets per second
  reset            clear every register (all flow and sketch state)
  sweep            free the slots idle beyond the 60 s flow timeout
  clear            remove mitigation and admission entries
  dump             outcome, mitigation and class counters, port statistics
Job commands (any Python 3):
  run JOB.json     hand a replay job to the root agent, decode the reports
  agent            the root agent: watches WORK/inbox, captures, replays
  flow12           write the 12-packet test flow and its expected report
  flood            single-packet flows from many sources for the flood alert
                   (model started with --time-disable)
  compare MODEL.json BMV2.json   same reports from model and BMv2?
  selftest         the functional tests above, from a fresh state, as one
                   results file (needs the SDE client and the agent)
"""
import argparse
import json
import os
import re
import socket
import struct
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "tmp" / "tofino"
GRPC = "localhost:50052"
CONTROLLER_PORT, INSPECT_PORT = 2, 3
FRONT = [8, 9, 10, 11]
FORWARD_PORT = 9
MIRROR_SESSION = 100
CONF_GUARD = 179
UNIT_NS = 256                   # TNA time unit (ingress MAC timestamp >> 8)
REPORT_FORMAT = "!IIIHHBBBBIIHHIIHHHIIIIIHBB"
REPORT_BYTES = 70
REPORT_PORT = 9999
REPORT_FIELDS = (
    "flow_id", "src", "dst", "src_port", "dst_port", "protocol", "dp_class",
    "dp_confidence", "reason", "pkt_count", "byte_count", "src_diversity",
    "dstport_fanout", "mean_iat_us", "mean_len", "min_len", "max_len",
    "fwd_pkts", "fwd_bytes", "iat_min_us", "iat_max_us", "duration_us",
    "len_dev", "first_len", "dp_action", "alert_level")
TIME_FIELDS = ("mean_iat_us", "iat_min_us", "iat_max_us", "duration_us")
# bucketise.p4inc feature -> (TNA bucket table, key field)
BUCKET_TABLES = {
    "byte_count": ("pipe.Ingress.bucket_bytes", "meta.bytes", "Ingress.set_byte_bucket"),
    "mean_len": ("pipe.Ingress.bucket_len", "meta.len_acc", "Ingress.set_len_bucket"),
    "mean_iat_us": ("pipe.Ingress.bucket_iat", "meta.iat_acc", "Ingress.set_iat_bucket"),
    "min_len": ("pipe.Ingress.bucket_minlen", "meta.min_len", "Ingress.set_minlen_bucket"),
    "max_len": ("pipe.Ingress.bucket_maxlen", "meta.max_len", "Ingress.set_maxlen_bucket"),
}
CLASSIFIER_KEYS = ("meta.byte_bucket", "meta.len_bucket", "meta.iat_bucket",
                   "meta.minlen_bucket", "meta.maxlen_bucket")


# ------------------------------------------------------------------ pcap I/O
def read_pcap(path):
    out = []
    with open(path, "rb") as f:
        magic = f.read(24)
        little = magic[:4] in (b"\xd4\xc3\xb2\xa1", b"\x4d\x3c\xb2\xa1")
        fmt = "<IIII" if little else ">IIII"
        while True:
            h = f.read(16)
            if len(h) < 16:
                break
            sec, usec, incl, _ = struct.unpack(fmt, h)
            out.append((sec * 1_000_000 + usec, f.read(incl)))
    return out


def write_pcap(path, frames):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
        for t_us, fr in frames:
            f.write(struct.pack("<IIII", t_us // 1_000_000, t_us % 1_000_000,
                                len(fr), len(fr)))
            f.write(fr)


def parse_report(payload):
    rec = dict(zip(REPORT_FIELDS, struct.unpack(REPORT_FORMAT, payload[:REPORT_BYTES])))
    return rec


def reports_in(path, unit_ns=UNIT_NS):
    """Reports in a controller-port capture; time fields in microseconds."""
    out = []
    for t_us, fr in read_pcap(path):
        if len(fr) < 42 + REPORT_BYTES or fr[12:14] != b"\x08\x00" or fr[23] != 17:
            continue
        u = 14 + (fr[14] & 0x0F) * 4
        if int.from_bytes(fr[u + 2:u + 4], "big") != REPORT_PORT:
            continue
        rec = parse_report(fr[u + 8:u + 8 + REPORT_BYTES])
        rec["raw_time_units"] = {k: rec[k] for k in TIME_FIELDS}
        for k in TIME_FIELDS:
            rec[k] = rec[k] * unit_ns / 1000.0
        rec["t"] = t_us / 1e6
        out.append(rec)
    return out


def dotted(a):
    return ".".join(str((a >> s) & 0xFF) for s in (24, 16, 8, 0))


def ip_of(s):
    return int.from_bytes(socket.inet_aton(s), "big")


# ------------------------------------------------------------ test flow
def _ipv4_checksum(hdr):
    s = sum(int.from_bytes(hdr[j:j + 2], "big") for j in range(0, 20, 2))
    while s >> 16:
        s = (s & 0xFFFF) + (s >> 16)
    return (~s) & 0xFFFF


def udp_frame(src, dst, sport, dport, length):
    length = max(length, 28)
    ip = bytearray(struct.pack("!BBHHHBBHII", 0x45, 0, length, 0, 0, 64, 17,
                               0, src, dst))
    ip[10:12] = _ipv4_checksum(ip).to_bytes(2, "big")
    udp = struct.pack("!HHHH", sport, dport, length - 20, 0)
    return (b"\x00\x00\x00\x00\x00\x02\x00\x00\x00\x00\x00\x01\x08\x00"
            + bytes(ip) + udp + bytes(length - 28))


FLOW12_LENS = (120, 80, 300, 64, 200, 150, 90, 400, 110, 130, 70, 95)
FLOW12_DIRS = (1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1)


def make_flow12(gap_ms, out, src="10.0.0.1", dst="10.0.0.2", sport=1234, dport=443):
    """
    Write a 12-packet UDP flow, one packet per gap, and return the report
    the emulator of scripts/replay_attack.py expects at its tenth packet.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import replay_attack as R
    src, dst = ip_of(src), ip_of(dst)
    keys = [(src, dst, sport, dport, 17)]
    gap = int(round(gap_ms * 1000))
    events = [(R.TS_BASE_US + i * gap, 0, i, L, d)
              for i, (L, d) in enumerate(zip(FLOW12_LENS, FLOW12_DIRS))]
    frames = [(t, udp_frame(src, dst, sport, dport, L) if d else
               udp_frame(dst, src, dport, sport, L)) for t, _, _, L, d in events]
    write_pcap(out, frames)
    verdict, k_at, fields, _ = R.emulate(events, keys, [[]] * 5, {}, 1, 10)
    f = fields[0]
    return {"pkt_count": R.DECISION_PACKETS, "byte_count": f[0], "mean_len": f[1],
            "mean_iat_us": f[2], "min_len": f[3], "max_len": f[4],
            "fwd_pkts": f[5], "fwd_bytes": f[6], "iat_min_us": f[7],
            "iat_max_us": f[8], "duration_us": f[9], "len_dev": f[10],
            "first_len": f[11], "reason": 1, "dp_class": 255,
            "flow_id": R._flow_hash(*keys[0]) % 0xFFFFFFFF,
            "src_diversity": 1, "dstport_fanout": 1}


def flow12(args):
    expect = make_flow12(args.gap_ms, args.out, args.src, args.dst, args.sport,
                         args.dport)
    Path(args.expect).write_text(json.dumps(expect, indent=1))
    print(f"{args.out}: 12 packets, {args.gap_ms} ms apart; expected report "
          f"in {args.expect}")


# ------------------------------------------------------------- the agent
class Capture(threading.Thread):
    """Writes every frame that arrives on one interface to a pcap file."""
    PACKET_OUTGOING = 4

    def __init__(self, iface, out):
        super().__init__(daemon=True)
        self.iface, self.out, self.count = iface, out, 0
        self.stop = threading.Event()
        # Protocol 0 until the bind: an ETH_P_ALL socket receives from every
        # interface between socket() and bind().
        self.sock = socket.socket(socket.AF_PACKET, socket.SOCK_RAW, 0)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 64 << 20)
        self.sock.bind((iface, 3))
        self.sock.settimeout(0.2)

    def run(self):
        with open(self.out, "wb") as f:
            f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
            while not self.stop.is_set():
                try:
                    fr, addr = self.sock.recvfrom(65535)
                except socket.timeout:
                    continue
                if addr[2] == self.PACKET_OUTGOING or addr[0] != self.iface:
                    continue
                t = time.time()
                sec = int(t)
                f.write(struct.pack("<IIII", sec, int((t - sec) * 1e6), len(fr), len(fr)))
                f.write(fr)
                self.count += 1
        self.sock.close()


def replay(pcap, iface, rate_scale, max_packets):
    tx = socket.socket(socket.AF_PACKET, socket.SOCK_RAW)
    tx.bind((iface, 0))
    packets = read_pcap(pcap)
    if max_packets:
        packets = packets[:max_packets]
    base = packets[0][0]
    t0 = time.perf_counter() + 0.5
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
    tx.close()
    late.sort()
    return {"packets_sent": len(packets),
            "send_lateness_ms": {"p50": round(1e3 * late[len(late) // 2], 3),
                                 "max": round(1e3 * late[-1], 3)}}


def agent_run(job_path, inbox, outbox, owner):
    job = json.loads(job_path.read_text())
    name = job_path.stem
    caps = [Capture(iface, outbox / f"{name}_{iface}.pcap")
            for iface in job.get("capture", [])]
    for c in caps:
        c.start()
    time.sleep(1.0)
    t_start = time.time()
    stats = replay(job["pcap"], job["iface"], job.get("rate_scale", 1.0),
                   job.get("max_packets", 0))
    time.sleep(job.get("settle_s", 3))
    for c in caps:
        c.stop.set()
    for c in caps:
        c.join(timeout=5)
    stats.update(t_start=t_start, t_end=time.time(),
                 captured={c.iface: c.count for c in caps})
    (outbox / f"{name}.done").write_text(json.dumps(stats))
    done = job_path.with_suffix(".ran")
    job_path.rename(done)
    for p in list(outbox.iterdir()) + [done]:
        os.chown(p, *owner)


def agent(args):
    """Root: serve replay jobs dropped into WORK/inbox."""
    import pwd
    inbox, outbox = args.work / "inbox", args.work / "outbox"
    inbox.mkdir(parents=True, exist_ok=True)
    outbox.mkdir(parents=True, exist_ok=True)
    uid = int(os.environ.get("SUDO_UID", 0)) or pwd.getpwnam(args.owner).pw_uid
    gid = int(os.environ.get("SUDO_GID", 0)) or uid
    print("agent uid", os.getuid(), "serving", inbox, flush=True)
    while True:
        for job in sorted(inbox.glob("*.json")):
            try:
                agent_run(job, inbox, outbox, (uid, gid))
                print("ran", job.name, flush=True)
            except Exception as e:      # keep serving later jobs
                print("failed", job.name, repr(e), flush=True)
                traceback.print_exc()
                job.rename(job.with_suffix(".failed"))
        time.sleep(0.5)


def submit(work, name, job, timeout=3600):
    """Hand a job to the agent, wait, and decode the captures."""
    inbox, outbox = work / "inbox", work / "outbox"
    for old in outbox.glob(f"{name}*"):
        old.unlink()
    (inbox / f"{name}.json").write_text(json.dumps(job))
    done = outbox / f"{name}.done"
    failed = inbox / f"{name}.failed"
    t0 = time.time()
    while not done.exists():
        if failed.exists():
            sys.exit(f"agent failed {name}; see {work}/agent.out")
        if time.time() - t0 > timeout:
            sys.exit(f"agent did not finish {name} in {timeout}s")
        time.sleep(0.5)
    stats = json.loads(done.read_text())
    out = {"job": job, "replay": stats, "captures": {}}
    ctl = job.get("controller_iface", "veth251")
    for iface in job.get("capture", []):
        pcap = outbox / f"{name}_{iface}.pcap"
        frames = read_pcap(pcap)
        cap = {"frames": len(frames)}
        if iface == ctl:
            cap["reports"] = reports_in(pcap)
        out["captures"][iface] = cap
    out["reports"] = out["captures"].get(ctl, {}).get("reports", [])
    print(f"{name}: sent {stats['packets_sent']}, captured "
          f"{stats['captured']}, {len(out['reports'])} report(s)")
    return out


def run_job(args):
    """Submit a job, wait for the agent, decode the controller capture."""
    job = json.loads(Path(args.job).read_text())
    name = args.name or Path(args.job).stem
    out = submit(args.work, name, job, args.timeout)
    if args.expect:
        exp = json.loads(Path(args.expect).read_text())
        out["check"] = check_report(out["reports"], exp, args.time_tolerance)
        print("check:", out["check"]["verdict"], out["check"].get("mismatch", ""))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1))
    return out


def check_report(reps, exp, time_tol):
    """Exact on counts and lengths; time fields within a relative tolerance."""
    if not reps:
        return {"verdict": "NO REPORT"}
    r = reps[0]
    bad = {}
    for k, v in exp.items():
        got = r.get(k)
        if k in TIME_FIELDS:
            if time_tol is None:
                continue
            if abs(got - v) > time_tol * max(v, 1):
                bad[k] = [v, got]
        elif got != v:
            bad[k] = [v, got]
    return {"verdict": "MATCH" if not bad else "MISMATCH", "mismatch": bad,
            "compared": sorted(k for k in exp if k not in TIME_FIELDS or time_tol is not None)}


# ------------------------------------------------------------- gRPC control
def connect(client_id=0):
    import bfrt_grpc.client as gc
    c = gc.ClientInterface(GRPC, client_id=client_id, device_id=0)
    c.bind_pipeline_config("policysplit")
    info = c.bfrt_info_get("policysplit")
    return gc, c, info, gc.Target(device_id=0, pipe_id=0xFFFF)


def setup(gc, info, tgt, args):
    port = info.table_get("$PORT")
    for dp in FRONT + [CONTROLLER_PORT, INSPECT_PORT]:
        try:
            port.entry_add(tgt, [port.make_key([gc.KeyTuple("$DEV_PORT", dp)])],
                           [port.make_data([gc.DataTuple("$SPEED", str_val="BF_SPEED_10G"),
                                            gc.DataTuple("$FEC", str_val="BF_FEC_TYP_NONE"),
                                            gc.DataTuple("$PORT_ENABLE", bool_val=True)])])
            print("port", dp, "added")
        except gc.BfruntimeRpcException as e:
            print("port", dp, str(e).splitlines()[0][:100])
    mir = info.table_get("$mirror.cfg")
    try:
        mir.entry_add(tgt, [mir.make_key([gc.KeyTuple("$sid", MIRROR_SESSION)])],
                      [mir.make_data([gc.DataTuple("$direction", str_val="INGRESS"),
                                      gc.DataTuple("$ucast_egress_port", CONTROLLER_PORT),
                                      gc.DataTuple("$ucast_egress_port_valid", bool_val=True),
                                      gc.DataTuple("$session_enable", bool_val=True),
                                      gc.DataTuple("$max_pkt_len", 16384)], "$normal")])
        print("mirror session", MIRROR_SESSION, "to port", CONTROLLER_PORT)
    except gc.BfruntimeRpcException as e:
        print("mirror", str(e).splitlines()[0][:120])
    lpm = info.table_get("pipe.Ingress.ipv4_lpm")
    try:
        lpm.entry_add(tgt, [lpm.make_key([gc.KeyTuple("hdr.ipv4.dstAddr", 0, prefix_len=0)])],
                      [lpm.make_data([gc.DataTuple("port", FORWARD_PORT)], "Ingress.forward")])
        print("default route to port", FORWARD_PORT)
    except gc.BfruntimeRpcException as e:
        print("lpm", str(e).splitlines()[0][:120])


def cut_points(path):
    """Feature -> sorted [(cut, bucket)] from the generated bucketise action."""
    cuts = {}
    for m in re.finditer(r"meta\.(\w+) >= (\d+)\) \{ meta\.\w+_bucket = (\d+)",
                         Path(path).read_text()):
        cuts.setdefault(m.group(1), []).append((int(m.group(2)), int(m.group(3))))
    return {k: sorted(v) for k, v in cuts.items()}


def range_prefixes(lo, hi, width=32):
    """Disjoint (value, mask) prefixes covering [lo, hi]."""
    out = []
    while lo <= hi:
        step = 1
        while (lo & ((step << 1) - 1)) == 0 and lo + (step << 1) - 1 <= hi \
                and (step << 1) <= (1 << width):
            step <<= 1
        out.append((lo, ((1 << width) - 1) & ~(step - 1)))
        lo += step
    return out


def buckets(gc, info, tgt, args):
    """
    Install the cut points; the inter-arrival cuts in TNA time units, times
    the replay's slow-down so that a trace replayed at 1/scale of its rate
    falls into the same buckets.
    """
    n_total = 0
    for feat, cuts in cut_points(args.bucketise).items():
        table, field, action = BUCKET_TABLES[feat]
        t = info.table_get(table)
        t.entry_del(tgt, [])
        if feat == "mean_iat_us":
            cuts = [(int(round(c * args.time_scale * 1000 / UNIT_NS)), b) for c, b in cuts]
        keys, data = [], []
        for i, (lo, b) in enumerate(cuts):
            hi = cuts[i + 1][0] - 1 if i + 1 < len(cuts) else (1 << 32) - 1
            for value, mask in range_prefixes(lo, hi):
                keys.append(t.make_key([gc.KeyTuple(field, value, mask),
                                        gc.KeyTuple("$MATCH_PRIORITY", 0)]))
                data.append(t.make_data([gc.DataTuple("b", b)], action))
        t.entry_add(tgt, keys, data)
        n_total += len(keys)
        print(f"{table}: {len(cuts)} cuts, {len(keys)} prefix entries")
    print("bucket entries:", n_total)


def tna_verdict(cls, conf, act):
    """What the controller installs: (dec, rsn) from class, confidence, action."""
    if cls != 255 and conf >= CONF_GUARD and 1 <= act <= 3:
        return act + 1, 2
    return 1, 1


def add_verdict(gc, info, tgt, key, cls, conf, act, prio=1):
    t = info.table_get("pipe.Ingress.flow_classifier")
    dec, rsn = tna_verdict(cls, conf, act)
    t.entry_add(tgt, [t.make_key([gc.KeyTuple(f, v, 0xFF) for f, v in zip(CLASSIFIER_KEYS, key)]
                                 + [gc.KeyTuple("$MATCH_PRIORITY", prio)])],
                [t.make_data([gc.DataTuple("cls", cls), gc.DataTuple("conf", conf),
                              gc.DataTuple("act", act), gc.DataTuple("dec", dec),
                              gc.DataTuple("rsn", rsn)], "Ingress.set_verdict")])
    return dec, rsn


def classifier(gc, info, tgt, args):
    """Install a BMv2 entries file (table_add ... => cls conf act [prio])."""
    n = 0
    for ln in Path(args.file).read_text().splitlines():
        if not ln.startswith("table_add"):
            continue
        head, tail = ln.split("=>")
        key = [int(tok.split("&&&")[0]) for tok in head.split()[3:8]]
        vals = [int(v) for v in tail.split()]
        prio = vals[3] if len(vals) > 3 else 1
        add_verdict(gc, info, tgt, key, vals[0], vals[1], vals[2], prio)
        n += 1
    print("admission entries:", n)


def verdict(gc, info, tgt, args):
    dec, rsn = add_verdict(gc, info, tgt, args.key, args.cls, args.conf, args.act)
    print(f"admission entry {args.key} -> class {args.cls} conf {args.conf} "
          f"action {args.act}: decision {dec}, reason {rsn}")


def _masked(spec, width):
    if "/" in spec:
        v, m = spec.split("/")
        return v, int(m)
    return spec, width


def mitigate(gc, info, tgt, args):
    """One mitigation entry; a wildcard is '*' or a /0 mask."""
    t = info.table_get("pipe.Ingress.mitigation")

    def addr(spec):
        if spec == "*":
            return 0, 0
        v, bits = _masked(spec, 32)
        return ip_of(v), ((1 << 32) - 1) & ~((1 << (32 - bits)) - 1)

    def port(spec):
        if spec == "*":
            return 0, 0
        return int(spec), 0xFFFF
    s, sm = addr(args.src)
    d, dm = addr(args.dst)
    p, pm = port(args.dport)
    t.entry_add(tgt, [t.make_key([gc.KeyTuple("hdr.ipv4.srcAddr", s, sm),
                                  gc.KeyTuple("hdr.ipv4.dstAddr", d, dm),
                                  gc.KeyTuple("meta.dst_port", p, pm),
                                  gc.KeyTuple("$MATCH_PRIORITY", args.prio)])],
                [t.make_data([], "Ingress.mitigate")])
    print(f"mitigation entry {args.src} -> {args.dst}:{args.dport} priority {args.prio}")


def meter(gc, info, tgt, args):
    t = info.table_get("pipe.Ingress.flow_meter")
    n = t.info.size_get()
    keys = [t.make_key([gc.KeyTuple("$METER_INDEX", i)]) for i in range(n)]
    data = [t.make_data([gc.DataTuple("$METER_SPEC_CIR_PPS", args.cir),
                         gc.DataTuple("$METER_SPEC_PIR_PPS", args.pir),
                         gc.DataTuple("$METER_SPEC_CBS_PKTS", args.cbs),
                         gc.DataTuple("$METER_SPEC_PBS_PKTS", args.pbs)])] * n
    t.entry_mod(tgt, keys, data)
    print(f"meter: {n} entries, committed {args.cir} pps burst {args.cbs}, "
          f"peak {args.pir} pps burst {args.pbs}")


def reset(gc, info, tgt, args):
    n = 0
    for name in sorted(info.table_name_list_get()):
        t = info.table_get(name)
        if t.info.type_get() == "Register":
            t.entry_del(tgt, [])
            n += 1
    for name in ("pipe.Ingress.class_verdicts",):
        t = info.table_get(name)
        t.entry_del(tgt, [])
    print("registers cleared:", n)


FLOW_TIMEOUT_UNITS = 60_000_000_000 // UNIT_NS      # 60 s idle timeout
BANK_SIZE = 8192


def sweep(gc, info, tgt, args):
    """
    The idle sweep of the TNA port: clear the tag of every slot whose last
    arrival is older than the flow timeout, which BMv2 does in band. The
    control plane has no clock of the data plane, so the newest timestamp
    in the array stands for now.
    """
    ts = info.table_get("pipe.Ingress.last_ts")
    last = {}
    for data, key in ts.entry_get(tgt, [], {"from_hw": True}):
        v = _val(data.to_dict()["Ingress.last_ts.f1"])
        if v:
            last[key.to_dict()["$REGISTER_INDEX"]["value"]] = v
    if not last:
        print("sweep: table empty")
        return
    now = max(last.values())
    idle = [i for i, v in last.items() if (now - v) & 0xFFFFFFFF > FLOW_TIMEOUT_UNITS]
    for bank, name in ((0, "pipe.Ingress.tag_a"), (1, "pipe.Ingress.tag_b")):
        t = info.table_get(name)
        slots = [i - bank * BANK_SIZE for i in idle if i // BANK_SIZE == bank]
        if slots:
            t.entry_mod(tgt, [t.make_key([gc.KeyTuple("$REGISTER_INDEX", s)]) for s in slots],
                        [t.make_data([gc.DataTuple(f"Ingress.{name.split('.')[-1]}.f1", 0)])] * len(slots))
        print(f"{name}: {len(slots)} idle slots freed")
    print(f"sweep: {len(last)} occupied slots, {len(idle)} idle beyond "
          f"{FLOW_TIMEOUT_UNITS * UNIT_NS / 1e9:.0f} s")


def clear(gc, info, tgt, args):
    for name in ("pipe.Ingress.mitigation", "pipe.Ingress.flow_classifier"):
        info.table_get(name).entry_del(tgt, [])
    print("mitigation and admission entries removed")


def _val(v):
    return v[0] if isinstance(v, list) else v


def dump(gc, info, tgt, args):
    out = {}
    for name in ("pipe.Ingress.outcome", "pipe.Ingress.mitigation",
                 "pipe.Ingress.flow_classifier"):
        t = info.table_get(name)
        rows = []
        for data, key in t.entry_get(tgt, [], {"from_hw": True}):
            k, d = key.to_dict(), data.to_dict()
            row = {"key": {kk: (vv.get("value"), vv.get("mask")) if "mask" in vv else vv.get("value")
                           for kk, vv in k.items()},
                   "action": d.get("action_name")}
            if "$COUNTER_SPEC_PKTS" in d:
                row["pkts"] = _val(d["$COUNTER_SPEC_PKTS"])
            for f in ("cls", "conf", "act", "dec", "rsn"):
                if f in d:
                    row[f] = d[f]
            rows.append(row)
        out[name] = rows
    cv = info.table_get("pipe.Ingress.class_verdicts")
    out["class_verdicts"] = [
        _val(data.to_dict()["$COUNTER_SPEC_PKTS"])
        for data, key in cv.entry_get(tgt, [], {"from_hw": True})]
    stat = info.table_get("$PORT_STAT")
    ports = {}
    for dp in FRONT + [CONTROLLER_PORT, INSPECT_PORT]:
        for data, key in stat.entry_get(tgt, [stat.make_key([gc.KeyTuple("$DEV_PORT", dp)])],
                                        {"from_hw": True}):
            d = data.to_dict()
            ports[dp] = {"rx": d.get("$FramesReceivedOK"), "tx": d.get("$FramesTransmittedOK")}
    out["ports"] = ports
    if args.registers:
        regs = {}
        for name in sorted(info.table_name_list_get()):
            t = info.table_get(name)
            if t.info.type_get() != "Register":
                continue
            nz = []
            for data, key in t.entry_get(tgt, [], {"from_hw": True}):
                d = data.to_dict()
                vals = {kk: _val(vv) for kk, vv in d.items()
                        if kk not in ("action_name", "is_default_entry")}
                if any(vals.values()):
                    nz.append((key.to_dict()["$REGISTER_INDEX"]["value"], vals))
            regs[name] = {"nonzero": len(nz), "first": nz[:args.registers]}
        out["registers"] = regs
    text = json.dumps(out, indent=1, default=str)
    if args.out:
        Path(args.out).write_text(text)
    print(text)


# --------------------------------------------------------------- self-test
TEST_IFACES = {"iface": "veth1", "forward": "veth3", "inspect": "veth9",
               "controller": "veth251"}


def _job(pcap, settle=3, rate_scale=1.0):
    return {"pcap": str(pcap), "iface": TEST_IFACES["iface"],
            "capture": [TEST_IFACES["forward"], TEST_IFACES["controller"],
                        TEST_IFACES["inspect"]],
            "settle_s": settle, "rate_scale": rate_scale}


def _counts(out):
    return {"forwarded": out["captures"][TEST_IFACES["forward"]]["frames"],
            "inspected": out["captures"][TEST_IFACES["inspect"]]["frames"],
            "reports": len(out["reports"])}


def selftest(gc, info, tgt, args):
    """
    The functional tests of the compiled program on the model, from a fresh
    state, written as one results file: the 12-packet flow at two spacings
    against the emulator, the three verdicts installed as admission entries,
    a mitigation entry, the diversity alert, and the model's packet rate.
    """
    work = args.work
    cuts = cut_points(args.bucketise)
    res = {"environment": {
        "program": "p4/tna/policysplit.p4", "target": "tofino2",
        "model": "tofino-model (Open P4 Studio SDE), chip type Tofino 2",
        "control_plane": "BF-RT gRPC", "time_unit_ns": UNIT_NS,
        "clock_granularity_ms": 1,
        "note": "functional model: packets are stamped with the host clock "
                "at 1 ms granularity, no line-rate claim"}}
    args.time_scale = 1.0
    buckets(gc, info, tgt, args)
    clear(gc, info, tgt, args)

    # The 12-packet flow at two spacings, each on a clean state. The model
    # stamps packets from a 1 ms timer, so each 50 ms gap carries up to
    # about 2 ms of jitter.
    res["flow12"] = {}
    for gap_ms, sport, tol in ((1.0, 1234, None), (50.0, 1235, 0.05)):
        pcap = work / f"selftest_flow12_{gap_ms:g}ms.pcap"
        exp = make_flow12(gap_ms, pcap, sport=sport)
        reset(gc, info, tgt, args)
        out = submit(work, f"selftest_flow12_{gap_ms:g}ms", _job(pcap))
        got = out["reports"][0] if out["reports"] else None
        res["flow12"][f"{gap_ms:g}ms"] = {
            "expected": exp, "report": got, **_counts(out),
            "check": check_report(out["reports"], exp, tol),
            "time_rel_error": {k: round(abs(got[k] - exp[k]) / exp[k], 4)
                               for k in TIME_FIELDS} if got else None}
        print(" ", gap_ms, "ms:", res["flow12"][f"{gap_ms:g}ms"]["check"]["verdict"])

    # The three verdicts, as admission entries on the bucket key of the
    # 1 ms flow, which the verdict flows repeat under other ports.
    key = list(bucket_key(res["flow12"]["1ms"]["expected"], cuts))
    res["verdicts"] = {"bucket_key": key}
    plan = (("drop", (1, 255, 1), None, 1301), ("redirect", (3, 255, 3), None, 1302),
            ("rate_limit", (2, 255, 2), (1, 1, 1, 1), 1303))
    for name, (cls, conf, act), met, sport in plan:
        clear(gc, info, tgt, args)
        if met:
            args.cir, args.cbs, args.pir, args.pbs = met
            meter(gc, info, tgt, args)
        dec, rsn = add_verdict(gc, info, tgt, key, cls, conf, act)
        pcap = work / f"selftest_{name}.pcap"
        make_flow12(1.0, pcap, sport=sport)
        reset(gc, info, tgt, args)
        out = submit(work, f"selftest_{name}", _job(pcap))
        rep = out["reports"][0] if out["reports"] else None
        res["verdicts"][name] = {
            "entry": {"cls": cls, "conf": conf, "act": act, "dec": dec, "rsn": rsn},
            "meter_pps_burst": met, **_counts(out),
            "report": {k: rep[k] for k in ("reason", "dp_class", "dp_confidence",
                                           "dp_action", "pkt_count")} if rep else None}
        print(" ", name, _counts(out), res["verdicts"][name]["report"])
    clear(gc, info, tgt, args)

    # A mitigation entry on the forward direction: dropped and counted,
    # the reverse direction passes, no flow state is touched.
    args.src, args.dst, args.dport, args.prio = "10.0.0.1", "10.0.0.2", "443", 1
    mitigate(gc, info, tgt, args)
    pcap = work / "selftest_mitigation.pcap"
    make_flow12(1.0, pcap, sport=1400)
    reset(gc, info, tgt, args)
    out = submit(work, "selftest_mitigation", _job(pcap))
    t = info.table_get("pipe.Ingress.mitigation")
    hits = [_val(d.to_dict().get("$COUNTER_SPEC_PKTS"))
            for d, k in t.entry_get(tgt, [], {"from_hw": True})]
    res["mitigation"] = {"entry": "10.0.0.1 -> 10.0.0.2 port 443",
                         "forward_packets": sum(FLOW12_DIRS),
                         "reverse_packets": len(FLOW12_DIRS) - sum(FLOW12_DIRS),
                         "entry_hits": hits[0] if hits else None, **_counts(out)}
    print("  mitigation", res["mitigation"])
    clear(gc, info, tgt, args)

    # Single-packet flows from many sources: the diversity alert, and the
    # model's packet rate from the forwarded frames' spacing.
    n_flows, n_sources = 300, 64
    pcap = work / "selftest_burst.pcap"
    write_pcap(pcap, [(1_000_000 + i * 500,
                       udp_frame(ip_of("10.1.0.1") + (i % n_sources),
                                 ip_of("10.9.9.9"), 20000 + i, 80, 100))
                      for i in range(n_flows)])
    reset(gc, info, tgt, args)
    out = submit(work, "selftest_burst", _job(pcap, settle=4))
    fr = read_pcap(work / "outbox" / f"selftest_burst_{TEST_IFACES['forward']}.pcap")
    span = (fr[-1][0] - fr[0][0]) / 1e6 if len(fr) > 1 else None
    res["burst"] = {"flows": n_flows, "sources": n_sources, **_counts(out),
                    "alerts": [{k: r[k] for k in ("reason", "alert_level", "src",
                                                  "dst_port", "src_diversity")}
                               for r in out["reports"]],
                    "forwarded_span_s": round(span, 3) if span else None,
                    "model_packets_per_s": round(len(fr) / span) if span else None}
    print("  burst", {k: v for k, v in res["burst"].items() if k != "alerts"},
          res["burst"]["alerts"])
    res["environment"]["model_packets_per_s"] = res["burst"]["model_packets_per_s"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


# ---------------------------------------------------------------- compare
def flood(args):
    """
    Single-packet flows from distinct sources to one destination, for the
    flood alert. The model processes too few packets per second to reach
    256 new flows in one second of its clock, so this is meant for a model
    started with --time-disable, whose clock stands still.
    """
    work = args.work
    pcap = work / "flood_burst.pcap"
    write_pcap(pcap, [(1_000_000 + i * 500,
                       udp_frame(ip_of("10.2.0.0") + i, ip_of("10.9.9.9"),
                                 30000 + i, 80, 100))
                      for i in range(args.flows)])
    out = submit(work, "flood_burst", _job(pcap, settle=5))
    alerts = [{k: r[k] for k in ("reason", "alert_level", "src", "src_diversity",
                                 "pkt_count", "duration_us")}
              for r in out["reports"]]
    res = {"flows": args.flows, "sources": args.flows, "destination": "10.9.9.9:80",
           "model_clock": args.model_clock, **_counts(out), "alerts": alerts,
           "flood_alerts": [a for a in alerts if a["reason"] == 4],
           "diversity_alerts": [a for a in alerts if a["reason"] == 3]}
    print(json.dumps({k: v for k, v in res.items() if k != "alerts"}, indent=1))
    Path(args.out).write_text(json.dumps(res, indent=1))


def bucket_key(rec, cuts):
    """The admission key of a report, from its fields and the cut points."""
    key = []
    for feat in ("byte_count", "mean_len", "mean_iat_us", "min_len", "max_len"):
        v = rec[feat]
        key.append(sum(v >= c for c, _ in cuts.get(feat, [])))
    return tuple(key)


def flows_with_decision(trace, max_packets):
    """
    Flow ids whose tenth packet lies within the first max_packets, the
    packets of each flow from its tenth on, and the number of frames.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    import replay_attack as R
    count, decided, from_tenth = {}, set(), {}
    frames = read_pcap(trace)
    if max_packets:
        frames = frames[:max_packets]
    for _, fr in frames:
        if len(fr) < 34 or fr[12:14] != b"\x08\x00":
            continue
        ihl = (fr[14] & 0x0F) * 4
        proto = fr[23]
        src = int.from_bytes(fr[26:30], "big")
        dst = int.from_bytes(fr[30:34], "big")
        sp = dp = 0
        if proto in (6, 17) and len(fr) >= 14 + ihl + 4:
            sp = int.from_bytes(fr[14 + ihl:16 + ihl], "big")
            dp = int.from_bytes(fr[16 + ihl:18 + ihl], "big")
        fid = R._flow_hash(src, dst, sp, dp, proto) % 0xFFFFFFFF
        count[fid] = count.get(fid, 0) + 1
        if count[fid] == R.DECISION_PACKETS:
            decided.add(fid)
        if count[fid] >= R.DECISION_PACKETS:
            from_tenth[fid] = from_tenth.get(fid, 0) + 1
    return decided, from_tenth, len(frames)


def compare(args):
    """
    Model reports against BMv2 reports of the same trace (a --replay-trace
    result of scripts/replay_attack.py). Decision reports are matched by
    flow id; the verdict, the counts and the lengths must agree exactly,
    the time fields within a tolerance after undoing the replay's slow-down.
    A verdict disagreement is attributed to the bucket that differs. The
    model handles one packet at a time, in about --model-packet-ms, so a
    recorded gap shorter than that (in trace time, divided by the slow-down)
    is stretched to it; the time fields are therefore also summarised over
    the flows whose BMv2 value lies above that floor. The frames the model
    forwarded are checked against the packets its own drop verdicts should
    have stopped (every packet of a dropped flow from its tenth on).
    """
    model = json.loads(Path(args.model).read_text())
    bm = json.loads(Path(args.bmv2).read_text())
    m_reps = model["captures"][args.controller_iface]["reports"]
    b_reps = bm["reports"]
    for r in m_reps:
        for k in TIME_FIELDS:
            r[k] = r[k] / args.time_scale
    cuts = cut_points(args.bucketise)
    floor_us = args.model_packet_ms * 1000 / args.time_scale
    # Reports recorded before the alert level existed lack that field.
    exact = [k for k in REPORT_FIELDS if k not in TIME_FIELDS
             and all(k in r for r in b_reps[:1])]
    verdict_fields = ("reason", "dp_class", "dp_confidence", "dp_action")

    def decisions(reps):
        out = {}
        for r in reps:
            if r["reason"] in (1, 2) and r["flow_id"] not in out:
                out[r["flow_id"]] = r
        return out
    m_dec, b_dec = decisions(m_reps), decisions(b_reps)
    scope, from_tenth, n_frames = None, {}, None
    if args.trace:
        scope, from_tenth, n_frames = flows_with_decision(args.trace, args.max_packets)
        b_dec = {k: v for k, v in b_dec.items() if k in scope}
    both = sorted(set(m_dec) & set(b_dec))
    field_diff, verdict_diff = {}, []
    time_diff = {k: [] for k in TIME_FIELDS}
    time_above = {k: [] for k in TIME_FIELDS}
    for fid in both:
        m, b = m_dec[fid], b_dec[fid]
        for f in exact:
            if m[f] != b[f]:
                field_diff[f] = field_diff.get(f, 0) + 1
        if any(m[f] != b[f] for f in verdict_fields):
            km, kb = bucket_key(m, cuts), bucket_key(b, cuts)
            verdict_diff.append({
                "flow_id": fid, "model": [m[f] for f in verdict_fields],
                "bmv2": [b[f] for f in verdict_fields],
                "buckets_model": km, "buckets_bmv2": kb,
                "differing_buckets": [n for n, (x, y) in enumerate(zip(km, kb)) if x != y]})
        for k in TIME_FIELDS:
            if b[k]:
                e = abs(m[k] - b[k]) / b[k]
                time_diff[k].append(e)
                if b[k] >= floor_us:
                    time_above[k].append(e)

    def pct(xs, q):
        xs = sorted(xs)
        return round(100 * xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None
    same_verdict = len(both) - len(verdict_diff)
    same_all = sum(all(m_dec[f][k] == b_dec[f][k] for k in exact) for f in both)
    # Enforcement: the model's own drop verdicts name the packets it should
    # not have forwarded; redirected flows leave on the inspection port.
    forwarded = model["captures"].get(TEST_IFACES["forward"], {}).get("frames")
    inspected = model["captures"].get(TEST_IFACES["inspect"], {}).get("frames")
    expect_drop = sum(from_tenth.get(f, 0) for f, r in m_dec.items()
                      if r["reason"] == 2 and r["dp_action"] == 1)
    expect_redirect = sum(from_tenth.get(f, 0) for f, r in m_dec.items()
                          if r["reason"] == 2 and r["dp_action"] == 3)
    res = {
        "time_scale": args.time_scale, "max_packets": args.max_packets,
        "frames_sent": n_frames,
        "flows_decided_in_trace": len(scope) if scope is not None else None,
        "model_reports": len(m_reps), "bmv2_reports_in_scope": len(b_dec),
        "model_decisions": len(m_dec),
        "model_verdicts": {"defer": sum(r["reason"] == 1 for r in m_dec.values()),
                           "drop": sum(r["reason"] == 2 and r["dp_action"] == 1
                                       for r in m_dec.values()),
                           "rate_limit": sum(r["reason"] == 2 and r["dp_action"] == 2
                                             for r in m_dec.values()),
                           "redirect": sum(r["reason"] == 2 and r["dp_action"] == 3
                                           for r in m_dec.values())},
        "model_alerts": sum(r["reason"] in (3, 4) for r in m_reps),
        "bmv2_alerts": sum(r["reason"] in (3, 4) for r in b_reps),
        "decisions_in_both": len(both),
        "only_model": sorted(set(m_dec) - set(b_dec)),
        "only_bmv2": sorted(set(b_dec) - set(m_dec)),
        "same_verdict": same_verdict,
        "same_all_exact_fields": same_all,
        "field_disagreements": field_diff,
        "verdict_disagreements": verdict_diff,
        "time_field_rel_error_pct": {k: {"p50": pct(v, .5), "p95": pct(v, .95), "max": pct(v, 1.0)}
                                     for k, v in time_diff.items()},
        "model_packet_ms": args.model_packet_ms,
        "time_floor_trace_us": floor_us,
        "time_fields_above_floor": {
            k: {"flows": len(v), "within_3pct": sum(e <= 0.03 for e in v),
                "p50": pct(v, .5), "p95": pct(v, .95)}
            for k, v in time_above.items()},
        "enforcement": {"forwarded": forwarded, "inspected": inspected,
                        "expected_dropped": expect_drop,
                        "expected_redirected": expect_redirect,
                        "expected_forwarded": (n_frames - expect_drop - expect_redirect)
                        if n_frames else None},
        "exact_fields": exact, "time_fields": list(TIME_FIELDS)}
    print(json.dumps({k: v for k, v in res.items()
                      if k not in ("only_model", "only_bmv2", "verdict_disagreements")}, indent=1))
    print("only_model", len(res["only_model"]), "only_bmv2", len(res["only_bmv2"]),
          "verdict_disagreements", len(verdict_diff))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=1))


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, default=WORK,
                    help="runtime directory (inbox, outbox, logs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("setup", "reset", "sweep", "clear"):
        sub.add_parser(name)
    p = sub.add_parser("buckets")
    p.add_argument("--bucketise", default=str(ROOT / "p4/generated/bucketise.p4inc"))
    p.add_argument("--time-scale", type=float, default=1.0,
                   help="slow-down of the replay (1/rate_scale of the job)")
    p = sub.add_parser("classifier")
    p.add_argument("file", nargs="?", default=str(ROOT / "p4/generated/classifier_entries.txt"))
    p = sub.add_parser("verdict")
    p.add_argument("key", type=int, nargs=5, help="byte len iat minlen maxlen buckets")
    p.add_argument("cls", type=int)
    p.add_argument("conf", type=int)
    p.add_argument("act", type=int, help="1 drop, 2 rate limit, 3 redirect")
    p = sub.add_parser("mitigate")
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("dport")
    p.add_argument("prio", type=int, nargs="?", default=1)
    p = sub.add_parser("meter")
    p.add_argument("cir", type=int)
    p.add_argument("cbs", type=int)
    p.add_argument("pir", type=int)
    p.add_argument("pbs", type=int)
    p = sub.add_parser("dump")
    p.add_argument("--registers", type=int, default=0,
                   help="also list non-zero register entries (first N each)")
    p.add_argument("--out")
    p = sub.add_parser("selftest")
    p.add_argument("--bucketise", default=str(ROOT / "p4/generated/bucketise.p4inc"))
    p.add_argument("--out", default=str(ROOT / "results/tofino_model.json"))
    p = sub.add_parser("run")
    p.add_argument("job")
    p.add_argument("--name")
    p.add_argument("--expect")
    p.add_argument("--time-tolerance", type=float, default=None,
                   help="relative tolerance on time fields (default: not compared)")
    p.add_argument("--timeout", type=float, default=3600)
    p.add_argument("--out")
    p = sub.add_parser("agent")
    p.add_argument("--owner", default="neuronetix")
    p = sub.add_parser("flow12")
    p.add_argument("--out", default=str(WORK / "flow12.pcap"))
    p.add_argument("--expect", default=str(WORK / "flow12_expected.json"))
    p.add_argument("--gap-ms", type=float, default=1.0)
    p.add_argument("--src", default="10.0.0.1")
    p.add_argument("--dst", default="10.0.0.2")
    p.add_argument("--sport", type=int, default=1234)
    p.add_argument("--dport", type=int, default=443)
    p = sub.add_parser("flood")
    p.add_argument("--flows", type=int, default=600)
    p.add_argument("--model-clock", default="stopped (--time-disable)",
                   help="how the model was started, recorded in the result")
    p.add_argument("--out", default=str(ROOT / "results/tofino_model_flood.json"))
    p = sub.add_parser("compare")
    p.add_argument("model")
    p.add_argument("bmv2")
    p.add_argument("--controller-iface", default="veth251")
    p.add_argument("--time-scale", type=float, default=1.0)
    p.add_argument("--trace", help="the replayed pcap, to scope the BMv2 reports")
    p.add_argument("--max-packets", type=int, default=0)
    p.add_argument("--model-packet-ms", type=float, default=7.0,
                   help="time the model spends on one packet (floor of its gaps)")
    p.add_argument("--bucketise", default=str(ROOT / "p4/generated/bucketise.p4inc"))
    p.add_argument("--out")
    args = ap.parse_args()

    if args.cmd == "agent":
        agent(args)
    elif args.cmd == "run":
        run_job(args)
    elif args.cmd == "flow12":
        flow12(args)
    elif args.cmd == "flood":
        flood(args)
    elif args.cmd == "compare":
        compare(args)
    else:
        gc, c, info, tgt = connect()
        try:
            {"setup": setup, "buckets": buckets, "classifier": classifier,
             "verdict": verdict, "mitigate": mitigate, "meter": meter,
             "reset": reset, "sweep": sweep, "clear": clear, "dump": dump,
             "selftest": selftest}[args.cmd](gc, info, tgt, args)
        finally:
            c.tear_down_stream()


if __name__ == "__main__":
    main()
