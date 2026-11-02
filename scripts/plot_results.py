"""
plot_results.py - Aggregate the controlled runs into tables and figures.

Reads every result file produced by the experiment grid, the baseline
suite, the data-plane classifier and the BMv2 test bed, aggregates
repeated seeds, and writes

  * results/controlled/summary.json  - one machine-readable record,
  * reports/tables/*.tex             - LaTeX tables and quoted numbers the
                                       manuscript includes verbatim,
  * reports/figures/*                - vector figures, and PGFPlots data in
                                       reports/figures/data.

The manuscript never restates a number; it includes the generated table,
so the text and the experiments cannot drift apart.

Usage:
    PYTHONPATH=. python scripts/plot_results.py
"""

import argparse
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing.feature_extractor import ID_TO_GROUP_LABEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "pdf.fonttype": 42,
    # Times, as in the manuscript text and its TikZ figures.
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes",
                   "Liberation Serif", "FreeSerif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.spines.top": False, "axes.spines.right": False,
})

# Colour-blind safe, and distinguishable when printed in greyscale.
PALETTE = ["#3C6E9E", "#8FB8DA", "#C6C6C6", "#96484B", "#D9A441"]

CLASSES = [ID_TO_GROUP_LABEL.get(i, str(i)) for i in range(8)]

# Placeholder for a cell that was not run. LaTeX renders a bare "--"
# as an en dash.
NA = r"n/a"

ENTROPY_LABEL = {
    "none": "no context",
    "real": "measured context",
    "class-marginal": "oracle class-conditioned context",
    "shuffle": "shuffled context",
}
SPLIT_LABEL = {
    "flow-isolated": "flow-isolated",
    "context-disjoint": "context-signature-disjoint",
    "random": "row-level",
}
HATCHES = ("", "///", "\\\\", "xx")
RUN_RE = re.compile(r"bimamba_(?P<ent>[a-z-]+)_(?P<aug>aug|noaug)_"
                    r"(?P<split>[a-z-]+)_s(?P<seed>\d+)_results\.json$")

CAPTURE_LABEL = {
    "benign": "Benign", "ddos_syn": "DDoS SYN", "dos_tcp": "DoS TCP",
    "recon": "Recon", "mirai": "Mirai", "spoofing_arp": "Spoofing ARP",
    "spoofing_dns": "Spoofing DNS", "bruteforce": "BruteForce",
}


def load_grid(results_dir: Path):
    groups = defaultdict(list)
    for path in sorted(results_dir.glob("bimamba_*_results.json")):
        m = RUN_RE.search(path.name)
        if not m:
            continue
        rec = json.loads(path.read_text())
        key = (m["ent"], m["aug"] == "aug", m["split"])
        groups[key].append(rec)
    return groups


def agg(records, path):
    vals = [r["test"][path] for r in records]
    return float(np.mean(vals)), float(np.std(vals)), len(vals)


def _per_class_map(record):
    """Per-class metrics keyed by class name, for either result layout."""
    pc = record["test"]["per_class"]
    if isinstance(pc, dict):
        return pc
    return {CLASSES[int(e["class"])]: e for e in pc if int(e["class"]) < len(CLASSES)}


def per_class_f1(records):
    out = {}
    maps = [_per_class_map(r) for r in records]
    for cls in CLASSES:
        vals = [m[cls]["f1"] for m in maps if cls in m]
        if vals:
            out[cls] = (float(np.mean(vals)), float(np.std(vals)))
    return out


def fmt(mean, std, pct=True):
    scale = 100.0 if pct else 1.0
    return f"{mean * scale:.1f} $\\pm$ {std * scale:.1f}"


# ── Tables ────────────────────────────────────────────────────────────

def table_ablation(groups, out: Path):
    """Two-factor ablation: context channel crossed with augmentation."""
    rows = []
    for ent in ("none", "real"):
        for aug in (False, True):
            recs = groups.get((ent, aug, "flow-isolated"))
            if not recs:
                continue
            a = agg(recs, "accuracy")
            f = agg(recs, "macro_f1")
            pc = per_class_f1(recs)
            rows.append((ENTROPY_LABEL[ent], "yes" if aug else "no",
                         a, f, pc.get("DDoS"), pc.get("DoS")))

    lines = [
        r"\begin{tabular}{llcccc}", r"\toprule",
        r"Context channel & Augmentation & Accuracy (\%) & Macro F1 (\%) "
        r"& DDoS F1 (\%) & DoS F1 (\%) \\", r"\midrule"]
    for ctx, aug, a, f, ddos, dos in rows:
        lines.append(
            f"{ctx} & {aug} & {fmt(a[0], a[1])} & {fmt(f[0], f[1])} & "
            f"{fmt(*ddos) if ddos else NA} & {fmt(*dos) if dos else NA}"
            r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")

    # Main effects, which is the quantity the ablation exists to estimate.
    effects = {}
    for aug in (False, True):
        a = groups.get(("none", aug, "flow-isolated"))
        b = groups.get(("real", aug, "flow-isolated"))
        if a and b:
            effects[f"context_effect_aug={aug}"] = round(
                100 * (agg(b, "macro_f1")[0] - agg(a, "macro_f1")[0]), 2)
    for ent in ("none", "real"):
        a = groups.get((ent, False, "flow-isolated"))
        b = groups.get((ent, True, "flow-isolated"))
        if a and b:
            effects[f"augmentation_effect_context={ent}"] = round(
                100 * (agg(b, "macro_f1")[0] - agg(a, "macro_f1")[0]), 2)
    return effects


REFERENCE_BASELINE = 'LightGBM'


def _baseline_cell(baselines, estimator, context, split):
    """Aggregated baseline entry, or None when that cell was not run."""
    key = context if split == 'flow-isolated' else f'{context}@{split}'
    return baselines.get('models', {}).get(estimator, {}).get(key)


def table_context_controls(groups, baselines, out: Path):
    """Measured context against controls that break its per-flow meaning."""
    order = ('none', 'shuffle', 'class-marginal', 'real')
    lines = [
        r'\begin{tabular}{lcccc}', r'\toprule',
        r'& \multicolumn{2}{c}{' + REFERENCE_BASELINE + r'} '
        r'& \multicolumn{2}{c}{BiMamba} \\',
        r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
        r'Context channel presented & Accuracy (\%) & Macro F1 (\%) '
        r'& Accuracy (\%) & Macro F1 (\%) \\', r'\midrule']
    values, base_values = {}, {}
    for ent in order:
        cells = []
        b = _baseline_cell(baselines, REFERENCE_BASELINE, ent, 'flow-isolated')
        if b:
            base_values[ent] = b['macro_f1_mean']
            cells += [fmt(b['accuracy_mean'], b['accuracy_std']),
                      fmt(b['macro_f1_mean'], b['macro_f1_std'])]
        else:
            cells += [NA, NA]

        recs = groups.get((ent, True, 'flow-isolated'))
        if recs:
            a, f = agg(recs, 'accuracy'), agg(recs, 'macro_f1')
            values[ent] = f[0]
            cells += [fmt(*a[:2]), fmt(*f[:2])]
        else:
            cells += [NA, NA]

        if cells.count(NA) == 4:
            continue
        lines.append(f'{ENTROPY_LABEL[ent]} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    out.write_text('\n'.join(lines) + '\n')

    diag = {}
    for tag, vals in (('bimamba', values), (REFERENCE_BASELINE, base_values)):
        d = {}
        if 'real' in vals and 'none' in vals:
            d['total_context_gain_pp'] = round(
                100 * (vals['real'] - vals['none']), 2)
        if 'real' in vals and 'class-marginal' in vals:
            d['gain_beyond_class_marginal_pp'] = round(
                100 * (vals['real'] - vals['class-marginal']), 2)
        if 'class-marginal' in vals and 'none' in vals:
            d['gain_explained_by_class_marginal_pp'] = round(
                100 * (vals['class-marginal'] - vals['none']), 2)
        if 'shuffle' in vals and 'none' in vals:
            d['gain_from_shuffled_context_pp'] = round(
                100 * (vals['shuffle'] - vals['none']), 2)
        if d:
            diag[tag] = d
    return diag


def table_protocol(groups, baselines, out: Path):
    """Effect of the partitioning rule at fixed model and context."""
    lines = [
        r'\begin{tabular}{llccc}', r'\toprule',
        r'Partitioning & Context & ' + REFERENCE_BASELINE +
        r' F1 (\%) & BiMamba F1 (\%) & Groups \\', r'\midrule']
    deltas, base_deltas = {}, {}
    for split in ('random', 'flow-isolated', 'context-disjoint'):
        for ent in ('none', 'real'):
            b = _baseline_cell(baselines, REFERENCE_BASELINE, ent, split)
            recs = groups.get((ent, True, split))
            if not b and not recs:
                continue

            if b:
                base_deltas[f'{split}|{ent}'] = b['macro_f1_mean']
                b_cell = fmt(b['macro_f1_mean'], b['macro_f1_std'])
            else:
                b_cell = NA

            ng = None
            if recs:
                f = agg(recs, 'macro_f1')
                deltas[f'{split}|{ent}'] = f[0]
                m_cell = fmt(*f[:2])
                ng = recs[0]['protocol']['split_report'].get('n_groups')
            else:
                m_cell = NA

            split_label = ("signature-disjoint" if split == "context-disjoint"
                           else SPLIT_LABEL[split])
            lines.append(
                f'{split_label} & {ENTROPY_LABEL[ent]} & '
                f'{b_cell} & {m_cell} & {ng if ng else NA}' + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    out.write_text('\n'.join(lines) + '\n')

    summary = {}
    for tag, vals in (('bimamba', deltas), (REFERENCE_BASELINE, base_deltas)):
        for ent in ('none', 'real'):
            base = vals.get(f'flow-isolated|{ent}')
            for split in ('random', 'context-disjoint'):
                other = vals.get(f'{split}|{ent}')
                if base is not None and other is not None:
                    summary[f'{tag}_{split}_minus_flow_isolated_{ent}_pp'] = \
                        round(100 * (other - base), 2)
    return summary


def table_baselines(baselines, groups, out: Path, bench_path: Path = None):
    """Estimators compared on one corpus, one partition, one feature set,
    with the CPU cost per flow measured with the context channel."""
    def cost(ms):
        return f"{ms:.3g}" if ms else NA
    lines = [
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"& \multicolumn{2}{c}{No context} & \multicolumn{3}{c}{Measured "
        r"context} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-6}",
        r"Estimator & Accuracy (\%) & Macro F1 (\%) & Accuracy (\%) "
        r"& Macro F1 (\%) & CPU ms per flow \\", r"\midrule"]
    for name, entry in baselines["models"].items():
        cells = []
        for ctx in ("none", "real"):
            e = entry.get(ctx)
            cells += ([fmt(e["accuracy_mean"], e["accuracy_std"]),
                       fmt(e["macro_f1_mean"], e["macro_f1_std"])]
                      if e else [NA, NA])
        cells.append(cost((entry.get("real") or {}).get("inference_ms_per_flow")))
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    no_ctx = groups.get(("none", True, "flow-isolated"))
    ctx = groups.get(("real", True, "flow-isolated"))
    if no_ctx and ctx:
        ms = None
        if bench_path is not None and bench_path.exists():
            bench = {b["name"]: b for b in json.loads(bench_path.read_text())}
            ms = (bench.get("BiMamba+Entropy (proposed)") or {}).get(
                "latency", {}).get("per_sample_in_batch_ms")
        cells = [fmt(*agg(no_ctx, "accuracy")[:2]),
                 fmt(*agg(no_ctx, "macro_f1")[:2]),
                 fmt(*agg(ctx, "accuracy")[:2]),
                 fmt(*agg(ctx, "macro_f1")[:2]), cost(ms)]
        lines.append(r"BiMamba (this work) & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def p4_resources(source: Path) -> dict:
    """Register and table requirements read from the P4 source declarations.

    Taken from the source rather than a compiled artefact so the reported
    footprint cannot silently describe an older build.
    """
    text = source.read_text()
    consts = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"#define\s+(\w+)\s+(\d+)", text)}
    regs = [(int(w), consts.get(n, n)) for w, n in
            re.findall(r"register<bit<(\d+)>>\((\w+)\)", text)]
    bits = sum(w * n for w, n in regs if isinstance(n, int))
    tables = {m.group(1): int(m.group(2)) for m in re.finditer(
        r"table\s+(\w+)\s*\{.*?size\s*=\s*(\d+);", text, re.S)}
    return {"n_registers": len(regs), "register_bits": bits, "tables": tables}


def table_dataplane(dp, bmv2, p4src: Path, out: Path):
    """What the specified pipeline requires and what the policy resolved."""
    res = p4_resources(p4src)
    tables = res["tables"]
    q = dp["quantisation"]
    d = dp["deployment_policy"]
    rows = [
        ("Register arrays", f"{res['n_registers']}"),
        ("Register state", f"{res['register_bits'] / 8 / 1024:.1f} KiB"),
        ("Flow table", "16{,}384 slots, two-choice"),
        ("Policy table capacity",
         f"{tables.get('flow_classifier', 0)} entries"),
        ("Mitigation table capacity",
         f"{tables.get('mitigation', 0)} entries"),
        ("Escalation report", "70 bytes, 12 flow registers"),
        ("Levels per key field", f"{q.get('levels', NA)}"),
        ("Key space (occupied by training flows)",
         f"{_grouped(q['key_space'])} ({_grouped(q['cells_occupied'])})"),
        ("Deployed entries, two-action table", f"{d['entries_installed']}"),
    ]
    if dp.get("tiered_policy"):
        t = dp["tiered_policy"]
        rows.append(("Deployed entries, calibrated four-action table",
                     f"{t['entries']} ({t['per_action']['1']['entries']} drop, "
                     f"{t['per_action']['2']['entries']} rate limit, "
                     f"{t['per_action']['3']['entries']} redirect)"))
    fn = _read("p4/generated/tiered_functional/classifier_entries.json")
    if fn:
        a = fn["action_distribution"]
        rows.append(("Four-action table of the switch function tests$^{a}$",
                     f"{fn['entries_installed']} ({a.get('1', 0)} drop, "
                     f"{a.get('2', 0)} rate limit, {a.get('3', 0)} redirect)"))
    rows += [
        ("Selective error target, calibration",
         f"{100 * d['false_drop_budget']:.0f}\\%, "
         + {"dependence": "dependence bound",
            "block": "block bootstrap bound",
            "binomial": "exact binomial bound"}.get(
             d.get("calibration", "binomial"), d.get("calibration", ""))),
        ("Held-out coverage of the deployed table",
         f"{100 * d['coverage']:.1f}\\%"),
        ("Held-out selective error", f"{100 * d['false_drop_rate']:.2f}\\%"),
    ]
    lines = [r"\begin{tabular}{ll}", r"\toprule",
             r"Quantity & Value \\", r"\midrule"]
    lines += [f"{k} & {v} \\\\" for k, v in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


# Stateful accesses a packet makes to each register array, read from the
# control flow of feature_extraction.p4: the two-choice probe reads the tag
# and timestamp of both candidate slots, every other per-flow array is read
# and written once at the chosen slot, and the sketch and decision arrays
# are read once and written when their value changes.
REGISTER_ACCESS = {
    "flow_tag": "2 reads, 1 write", "flow_last_ts": "2 reads, 1 write",
    "flow_decision": "1 read, 1 write",
    "victim_new_flows": "1 read, 1 write (new flow)",
    "victim_window_ts": "1 read, 1 write (new flow)",
}
REGISTER_ROLE = {
    "flow_pkts": "packets", "flow_bytes": "bytes",
    "flow_last_ts": "last arrival, probe and timeout",
    "flow_iat_ewma": "mean inter-arrival time (EWMA)",
    "flow_len_ewma": "mean length (EWMA)", "flow_min_len": "minimum length",
    "flow_max_len": "maximum length", "flow_tag": "flow tag, probe",
    "flow_decision": "verdict state", "flow_init_orient": "first direction",
    "flow_first_ts": "first arrival", "flow_first_len": "first length",
    "flow_fwd_pkts": "forward packets", "flow_fwd_bytes": "forward bytes",
    "flow_iat_min": "minimum inter-arrival", "flow_iat_max": "maximum inter-arrival",
    "flow_len_dev": "length deviation (EWMA)",
    "victim_src_bitmap": "source bitmap per victim",
    "victim_src_count": "source occupancy per victim",
    "source_port_bitmap": "port bitmap per source",
    "source_port_count": "port occupancy per source",
    "victim_reported": "alert sent",
    "victim_new_flows": "new flows per victim, window",
    "victim_window_ts": "window start per victim",
}
# Hash computations on a packet's path: three CRC32 for the two slots and
# the tag (one branch of compute_slots), two CRC16 for the sketch slots, and
# one CRC32 each for the bitmap bit of the two sketches.
HASHES_PER_PACKET = {"CRC32": 5, "CRC16": 2}
# Longest chain of dependent steps in the ingress control flow; on an
# RMT-style pipeline each step needs its own stage, so its length is a
# lower bound on the stage count. The forwarding and mitigation lookups
# depend only on headers and can share stages with the chain.
DEPENDENCY_CHAIN = (
    "hash the canonical key into two slots and the tag",
    "probe the tag and timestamp of both candidate slots",
    "select the slot (match, free, idle or less recently used)",
    "packet and byte counters, timestamp, tag and verdict reset",
    "first-packet fields, direction counts, length extrema, sketch bitmaps",
    "running means and deviation, sketch occupancies",
    "quantise the five policy registers (range match)",
    "admission table (ternary match)",
    "record the verdict and clone the report",
    "meter, drop or redirect",
)


def _p4_field_widths(p4dir: Path) -> dict:
    """Width of every header and metadata field as ``hdr.x.y`` or ``meta.y``."""
    text = (p4dir / "headers.p4").read_text()
    blocks = {m.group(1): m.group(2) for m in
              re.finditer(r"(?:header|struct)\s+(\w+)\s*\{(.*?)\}", text, re.S)}
    fields = {}
    for typ, body in blocks.items():
        for w, name in re.findall(r"bit<(\d+)>\s+(\w+);", body):
            fields[(typ, name)] = int(w)
    widths = {}
    for typ, inst in re.findall(r"(\w+_t)\s+(\w+);", blocks["headers_t"]):
        for (t, name), w in fields.items():
            if t == typ:
                widths[f"hdr.{inst}.{name}"] = w
    for (t, name), w in fields.items():
        if t == "metadata_t":
            widths[f"meta.{name}"] = w
    return widths


def table_resources(p4dir: Path, out: Path, macros: dict):
    """
    Resource accounting of the data plane read from the P4 source: every
    register array with width, depth, size and accesses per packet, the
    match tables with key width and match kinds, the meter and hashes, and
    the dependency chain that bounds the stage count from below.
    """
    text = (p4dir / "feature_extraction.p4").read_text()
    consts = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"#define\s+(\w+)\s+(\d+)", text)}
    regs = [(name, int(w), consts[n]) for w, n, name in
            re.findall(r"register<bit<(\d+)>>\((\w+)\)\s+(\w+);", text)]
    widths = _p4_field_widths(p4dir)
    tables = []
    for name, body in re.findall(r"table\s+(\w+)\s*\{(.*?)\n    \}", text, re.S):
        keys = re.findall(r"([\w.]+)\s*:\s*(\w+);", body)
        size = int(re.search(r"size\s*=\s*(\d+);", body).group(1))
        tables.append((name, size, keys, sum(widths[k] for k, _ in keys)))
    meter = re.search(r"meter\((\w+),\s*MeterType\.(\w+)\)", text)
    lines = [r"\begin{tabular}{llrrrl}", r"\toprule",
             r"Structure & Role & Width & Entries & KiB & Per packet \\",
             r"\midrule"]
    total_bits, group_bits = 0, {"flow": 0, "sketch": 0}
    ops = 0
    for name, w, n in regs:
        kib = w * n / 8 / 1024
        total_bits += w * n
        group = "flow" if name.startswith("flow_") else "sketch"
        group_bits[group] += w * n
        access = REGISTER_ACCESS.get(name, "1 read, 1 write")
        ops += int(access[0]) + int(access.split(", ")[1][0])
        tex = name.replace("_", r"\_")
        lines.append(f"\\texttt{{{tex}}} & {REGISTER_ROLE[name]} & {w} & "
                     f"{_grouped(n)} & {kib:.1f} & {access} \\\\")
    lines.append(r"\midrule")
    mname, mtype = meter.group(1), meter.group(2)
    lines.append(f"\\texttt{{flow\\_meter}} & two-rate meter, {mtype} & "
                 f"& {_grouped(consts[mname])} & & 1 execute \\\\")
    lines.append(r"\midrule")
    key_desc = {"flow_classifier": "5 quantised registers",
                "mitigation": "source, destination, port",
                "ipv4_lpm": "destination prefix",
                "flood_levels": "new-flow count, 8 constant entries"}
    for name, size, keys, kbits in tables:
        kinds = sorted({k for _, k in keys}, key=lambda s: s != "exact")
        tex = name.replace("_", r"\_")
        lines.append(f"\\texttt{{{tex}}} & {key_desc[name]} "
                     f"({', '.join(kinds)}) & {kbits} & {_grouped(size)} & & "
                     f"1 lookup \\\\")
    lines.append(r"\midrule")
    hashes = ", ".join(f"{v} {k}" for k, v in HASHES_PER_PACKET.items())
    lines.append(f"Hashes & slots, tag, sketch bits & & & & {hashes} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    tcam = {name: kbits for name, _, keys, kbits in tables
            if any(k == "ternary" for _, k in keys)}
    macros.update({
        "ResRegisterArrays": str(len(regs)),
        "ResFlowArrays": str(sum(n.startswith("flow_") for n, _, _ in regs)),
        "ResSketchArrays": str(sum(not n.startswith("flow_") for n, _, _ in regs)),
        "ResFlowKiB": f"{group_bits['flow'] / 8 / 1024:.0f}",
        "ResSketchKiB": f"{group_bits['sketch'] / 8 / 1024:.0f}",
        "ResTotalKiB": f"{total_bits / 8 / 1024:.0f}",
        "ResFlowBitsPerSlot": str(sum(w for n, w, _ in regs if n.startswith("flow_"))),
        "ResStatefulOps": str(ops + 1),
        "ResHashesPerPacket": str(sum(HASHES_PER_PACKET.values())),
        "ResChainSteps": str(len(DEPENDENCY_CHAIN)),
        "ResClassifierKeyBits": str(tcam.get("flow_classifier")),
        "ResMitigationKeyBits": str(tcam.get("mitigation")),
        "ResTcamEntries": _grouped(sum(size for name, size, _, _ in tables
                                       if name in tcam)),
        "ResMeterEntries": _grouped(consts[mname]),
    })
    chain = "\n".join(f"    \\item {s}" for s in DEPENDENCY_CHAIN)
    out.with_name("resources_chain.tex").write_text(
        "\\begin{enumerate}\n" + chain + "\n\\end{enumerate}\n")


# Structures of the TNA port (p4/tna/policysplit.p4) grouped as the compiled
# resource table presents them; the names are the compiler's.
TNA_GROUPS = (
    ("Forwarding table", "forwarding", ("ipv4_lpm",)),
    ("Mitigation table and its counter", "mitigation",
     ("mitigation", "mitigation_hits")),
    ("Flow tag banks A and B", "tags", ("tag_a", "tag_b")),
    ("Per-flow registers", "flow",
     ("last_ts", "pkts", "bytes", "min_len", "max_len", "init_orient",
      "first_len", "first_ts", "fwd_pkts", "fwd_bytes", "iat_min", "iat_max",
      "iat_mean", "len_mean", "dev_mean", "decision", "verdict_class",
      "verdict_confidence", "verdict_action")),
    ("Sketches and alert flag", "sketch",
     ("victim_src_bits", "source_port_bits", "victim_src_count",
      "source_port_count", "victim_reported")),
    ("New-flow counter and alert levels", "flood",
     ("victim_new_flows", "flood_levels")),
    ("Per-flow meter", "meter", ("flow_meter",)),
    ("Quantisation and phase tables", "buckets",
     ("bucket_bytes", "bucket_len", "bucket_iat", "bucket_minlen",
      "bucket_maxlen", "decision_phase")),
    ("Admission table", "admission", ("flow_classifier",)),
    ("Outcome table and counters", "outcome",
     ("outcome", "outcome_hits", "class_verdicts")),
    ("Egress report rewrite", "egress", ("cast_as_udp",)),
)


def tna_resources_facts(compiled: Path, out: Path, macros: dict):
    """
    What the Tofino compiler reports for the TNA port: the stage count and
    the SRAM, TCAM and ALU use per structure group (table), and the totals
    against the device's capacity (macros). Reads the logs that
    p4/tna/build.sh keeps under compiled/<target>/.
    """
    import gzip
    metrics = json.loads((compiled / "metrics.json").read_text())
    with gzip.open(compiled / "resources.json.gz", "rt") as fh:
        res = json.load(fh)["resources"]
    stages = res["mau"]["mau_stages"]
    st0 = stages[0]
    cap = {
        "sram": st0["rams"]["nRows"] * st0["rams"]["nColumns"],
        "tcam": st0["tcams"]["nRows"] * st0["tcams"]["nColumns"],
        "salu": st0["meter_alus"]["nAlus"],
        "stats": st0["statistic_alus"]["nAlus"],
        "gateways": st0["gateways"]["nRows"] * st0["gateways"]["nUnits"],
        "hash": st0["hash_bits"]["nBits"] * st0["hash_bits"]["nFunctions"],
        "vliw": st0["vliw"]["size"],
    }
    # resources.json lists the stages the program uses; the device's stage
    # count is the number of rows of the per-stage table in the MAU log.
    n_stages = 1 + max(int(m) for m in re.findall(
        r"^\|\s+(\d+)\s+\|", (compiled / "mau.resources.log").read_text(), re.M))
    use = {}

    def _add(name, kind, stage):
        base = name.split(".")[-1].split("$")[0]
        if base.startswith("tbl_"):
            base = base[4:]
        e = use.setdefault(base, {"sram": 0, "tcam": 0, "salu": 0, "stats": 0,
                                  "stages": set()})
        e[kind] += 1
        e["stages"].add(stage)

    totals = {"gateways": 0, "hash": 0, "vliw": 0}
    for st in stages:
        n = st["stage_number"]
        for t in st["rams"]["srams"]:
            for u in t["usages"]:
                _add(u["used_by"], "sram", n)
        for t in st["tcams"]["tcams"]:
            for u in t["usages"]:
                _add(u["used_by"], "tcam", n)
        for m in st["meter_alus"]["meters"]:
            for u in m["usages"]:
                _add(u["used_by"], "salu", n)
        for m in st["statistic_alus"]["stats"]:
            for u in m["usages"]:
                _add(u["used_by"], "stats", n)
        totals["gateways"] += len(st["gateways"]["gateways"])
        totals["hash"] += len(st["hash_bits"]["bits"])
        totals["vliw"] += len(st["vliw"]["instructions"])
    seen = set()
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Structures & Stages & SRAM & TCAM & ALUs \\", r"\midrule"]
    sums = {"sram": 0, "tcam": 0, "salu": 0, "stats": 0}
    for label, tag, names in TNA_GROUPS:
        g = {"sram": 0, "tcam": 0, "salu": 0, "stats": 0, "stages": set()}
        for nm in names:
            e = use.get(nm)
            if e is None:
                continue
            seen.add(nm)
            for k in ("sram", "tcam", "salu", "stats"):
                g[k] += e[k]
            g["stages"] |= e["stages"]
        for k in sums:
            sums[k] += g[k]
        lo, hi = min(g["stages"]), max(g["stages"])
        span = str(lo) if lo == hi else f"{lo} to {hi}"
        alus = []
        if g["salu"]:
            alus.append(f"{g['salu']} stateful")
        if g["stats"]:
            alus.append(f"{g['stats']} statistics")
        lines.append(f"{label} & {span} & {g['sram'] or ''} & {g['tcam'] or ''}"
                     f" & {', '.join(alus)} \\\\")
        macros[f"Tna{tag.capitalize()}Sram"] = str(g["sram"])
        macros[f"Tna{tag.capitalize()}Tcam"] = str(g["tcam"])
    missing = set(use) - seen
    assert not missing, f"TNA structures without a group: {sorted(missing)}"
    mau = metrics["mau"]
    assert sums["sram"] == mau["srams"] and sums["tcam"] == mau["tcams"], \
        (sums, mau)
    lines.append(r"\midrule")
    lines.append(f"All & {metrics_stages(metrics, compiled)} of {n_stages} & "
                 f"{sums['sram']} of {cap['sram'] * n_stages} & "
                 f"{sums['tcam']} of {cap['tcam'] * n_stages} & "
                 f"{sums['salu']} of {cap['salu'] * n_stages} stateful, "
                 f"{sums['stats']} of {cap['stats'] * n_stages} statistics \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    phv = {p["bit_width"]: p["containers_occupied"] for p in metrics["phv"]["normal"]}
    used_stages = metrics_stages(metrics, compiled)
    macros.update({
        "TnaCompiler": metrics["compiler_version"],
        "TnaTarget": {"tofino2": "Tofino 2", "tofino": "Tofino"}.get(
            metrics["target"], metrics["target"]),
        "TnaStages": str(used_stages), "TnaStagesTotal": str(n_stages),
        "TnaEgressStages": str(egress_stages(compiled)),
        "TnaSram": str(sums["sram"]), "TnaSramTotal": _grouped(cap["sram"] * n_stages),
        "TnaSramPct": _pc(sums["sram"] / (cap["sram"] * n_stages), 1) + r"\%",
        "TnaTcam": str(sums["tcam"]), "TnaTcamTotal": str(cap["tcam"] * n_stages),
        "TnaTcamPct": _pc(sums["tcam"] / (cap["tcam"] * n_stages), 1) + r"\%",
        "TnaSalu": str(sums["salu"]), "TnaSaluTotal": str(cap["salu"] * n_stages),
        "TnaStatsAlu": str(sums["stats"]),
        "TnaGateways": str(totals["gateways"]),
        "TnaGatewaysTotal": str(cap["gateways"] * n_stages),
        "TnaHashBits": str(totals["hash"]),
        "TnaHashBitsTotal": _grouped(cap["hash"] * n_stages),
        "TnaVliw": str(totals["vliw"]), "TnaVliwTotal": str(cap["vliw"] * n_stages),
        "TnaPhvEight": str(phv.get(8, 0)), "TnaPhvSixteen": str(phv.get(16, 0)),
        "TnaPhvThirtyTwo": str(phv.get(32, 0)),
        "TnaParserRows": str(metrics["parser"]["ingress"]["tcam_rows"]),
        "TnaLatency": str(next(x["cycles"] for x in mau["latency"]
                               if x["gress"] == "ingress")),
    })
    # Share of packets a flow's first ten packets make up on the recorded
    # captures: these are the packets the port resubmits (the decision
    # packet is the tenth, as in the capture facts above).
    for tag, path in (("Captures", "results/capture/slices_merged.json"),
                      ("Benign", "results/capture/benign_unseen.json")):
        man = _read(path)
        if man:
            fl = man["flows"]
            tot = sum(f["packets"] for f in fl)
            first = sum(min(f["packets"], 10) for f in fl)
            macros[f"TnaResubmitShare{tag}"] = _pc(first / tot, 1) + r"\%"
            macros[f"TnaPasses{tag}"] = f"{1 + first / tot:.2f}"
            # Port rate the ingress pipeline sustains when every resubmitted
            # packet takes a second pass, as a share of its packet rate.
            macros[f"TnaPortShare{tag}"] = _pc(1 / (1 + first / tot), 0) + r"\%"


def tofino_model_facts(selftest, trace, macros: dict):
    """
    Macros from the runs of the compiled TNA program on the Tofino 2
    reference model (scripts/tofino_model.py selftest and compare).
    """
    if selftest:
        env = selftest["environment"]
        macros["TnaModelPps"] = str(env["model_packets_per_s"])
        slow = selftest["flow12"]["50ms"]
        macros["TnaModelSlowGapMs"] = "50"
        macros["TnaModelSlowTimeErr"] = _pc(
            max(slow["time_rel_error"].values()), 1) + r"\%"
        macros["TnaModelSlowMatch"] = slow["check"]["verdict"].lower()
        macros["TnaModelFastMatch"] = selftest["flow12"]["1ms"]["check"]["verdict"].lower()
        v = selftest["verdicts"]
        macros.update({
            "TnaModelDropForwarded": str(v["drop"]["forwarded"]),
            "TnaModelRedirectForwarded": str(v["redirect"]["forwarded"]),
            "TnaModelRedirectInspected": str(v["redirect"]["inspected"]),
            "TnaModelRateForwarded": str(v["rate_limit"]["forwarded"]),
            "TnaModelMitigationHits": str(selftest["mitigation"]["entry_hits"]),
            "TnaModelMitigationForwarded": str(selftest["mitigation"]["forwarded"]),
            "TnaModelMitigationReports": str(selftest["mitigation"]["reports"]),
            "TnaModelBurstFlows": str(selftest["burst"]["flows"]),
            "TnaModelBurstSources": str(selftest["burst"]["sources"]),
            "TnaModelBurstAlerts": str(len(selftest["burst"]["alerts"])),
        })
    if trace:
        same_v = trace["same_verdict"]
        both = trace["decisions_in_both"]
        above = trace["time_fields_above_floor"]
        steady = ("mean_iat_us", "iat_max_us", "duration_us")
        enf = trace["enforcement"]
        macros.update({
            "TnaModelTracePackets": _grouped(trace["max_packets"]),
            "TnaModelTraceScale": f"{trace['time_scale']:g}",
            "TnaModelTraceInScope": _grouped(trace["flows_decided_in_trace"]),
            "TnaModelTraceDecisions": _grouped(trace["model_decisions"]),
            "TnaModelTraceDrops": _grouped(trace["model_verdicts"]["drop"]),
            "TnaModelTraceBoth": _grouped(both),
            "TnaModelTraceSameVerdict": _grouped(same_v),
            "TnaModelTraceSameVerdictPct": _pc(same_v / both, 1) + r"\%" if both else "",
            "TnaModelTraceSameFields": _grouped(trace["same_all_exact_fields"]),
            "TnaModelTraceOnlyModel": str(len(trace["only_model"])),
            "TnaModelTraceOnlyBmv": str(len(trace["only_bmv2"])),
            "TnaModelTraceVerdictDiff": str(len(trace["verdict_disagreements"])),
            "TnaModelTraceAlerts": str(trace["model_alerts"]),
            # The three time fields a flow's longer gaps set: median error
            # over all flows, and the share within 3% where the recorded
            # value exceeds the model's per-packet time.
            "TnaModelTraceTimeMedian": _pc(max(
                trace["time_field_rel_error_pct"][k]["p50"] for k in steady) / 100, 2) + r"\%",
            "TnaModelTraceTimeWithin": _pc(min(
                above[k]["within_3pct"] / above[k]["flows"] for k in steady), 0) + r"\%",
            "TnaModelTraceFloorMs": f"{trace['time_floor_trace_us'] / 1000:g}",
            "TnaModelPacketMs": f"{trace['model_packet_ms']:g}",
            "TnaModelTraceMinGapBelow": _grouped(both - above["iat_min_us"]["flows"]),
            "TnaModelTraceMinGapBelowPct": _pc(
                (both - above["iat_min_us"]["flows"]) / both, 0) + r"\%" if both else "",
            "TnaModelTraceMinGapWithin": _pc(
                above["iat_min_us"]["within_3pct"] / above["iat_min_us"]["flows"], 0) + r"\%",
            "TnaModelTraceForwarded": _grouped(enf["forwarded"]),
            "TnaModelTraceExpectedDropped": _grouped(enf["expected_dropped"]),
            "TnaModelTraceExpectedForwarded": _grouped(enf["expected_forwarded"]),
        })
        fd = trace["field_disagreements"]
        macros["TnaModelTraceFieldDiff"] = ", ".join(
            f"{k.replace('_', ' ')} {v}" for k, v in sorted(fd.items())) or "none"
        vd = trace["verdict_disagreements"]
        names = ("byte", "length", "inter-arrival", "minimum length", "maximum length")
        macros["TnaModelTraceVerdictDiffBucket"] = ", ".join(
            " and ".join(names[i] for i in d["differing_buckets"]) for d in vd) or "none"


def metrics_stages(metrics: dict, compiled: Path) -> int:
    """Stages the final table allocation used, from the table summary."""
    text = (compiled / "table_summary.log").read_text()
    return int(re.findall(r"Number of stages in table allocation: (\d+)", text)[-1])


def egress_stages(compiled: Path) -> int:
    text = (compiled / "table_summary.log").read_text()
    return int(re.findall(
        r"Number of stages for egress table allocation: (\d+)", text)[-1])


def aggregate_policy(paths):
    """Aggregate the flat and policy-aligned admission runs by budget."""
    runs = [json.loads(p.read_text()) for p in paths]
    if not runs:
        return {}
    budgets = sorted({r["false_drop_budget"]
                      for d in runs
                      for r in d["policy_comparison"]["points"]})
    out = {}
    for budget in budgets:
        rows = [next(r for r in d["policy_comparison"]["points"]
                     if abs(r["false_drop_budget"] - budget) < 1e-12)
                for d in runs]
        entry = {"n_seeds": len(rows)}
        for method in ("label_confidence", "action_confidence"):
            # A budget the calibration grid cannot meet admits nothing, and
            # then the benign share among admitted drops is undefined.
            cov = np.array([r[method]["coverage"] or 0.0 for r in rows])
            err = np.array([r[method]["false_drop_rate"] for r in rows
                            if r[method]["false_drop_rate"] is not None])
            entry[method] = {
                "n_calibrated": int(len(err)),
                "coverage_mean": float(cov.mean()),
                "coverage_std": float(cov.std()),
                "false_drop_mean": float(err.mean()) if err.size else None,
                "false_drop_std": float(err.std()) if err.size else None,
            }
        entry["coverage_gain_pp"] = 100 * (
            entry["action_confidence"]["coverage_mean"] -
            entry["label_confidence"]["coverage_mean"])
        out[budget] = entry
    return out


def table_policy(policy, out: Path):
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"Budget & \multicolumn{2}{c}{Label confidence} & "
             r"\multicolumn{2}{c}{Policy-aligned} & Gain \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"& Coverage & Benign share & Coverage & Benign share & (points) \\",
             r"\midrule"]
    for budget, e in sorted(policy.items()):
        flat, aligned = e["label_confidence"], e["action_confidence"]

        def cells(d):
            mark = ("" if d["n_calibrated"] == e["n_seeds"]
                    else f"$^{{{d['n_calibrated']}/{e['n_seeds']}}}$")
            share = (fmt(d["false_drop_mean"], d["false_drop_std"])
                     if d["false_drop_mean"] is not None else NA)
            return (f"{fmt(d['coverage_mean'], d['coverage_std'])}{mark}",
                    share)

        flat_cov, flat_err = cells(flat)
        aligned_cov, aligned_err = cells(aligned)
        lines.append(
            f"{100 * budget:.1f}\\% & {flat_cov} & {flat_err} & "
            f"{aligned_cov} & {aligned_err} & "
            f"{e['coverage_gain_pp']:+.1f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def figure_policy(policy, out: Path):
    budgets = np.array(sorted(policy))
    flat = np.array([100 * policy[b]["label_confidence"]["coverage_mean"]
                     for b in budgets])
    flat_std = np.array([100 * policy[b]["label_confidence"]["coverage_std"]
                         for b in budgets])
    aligned = np.array([100 * policy[b]["action_confidence"]["coverage_mean"]
                        for b in budgets])
    aligned_std = np.array([
        100 * policy[b]["action_confidence"]["coverage_std"] for b in budgets])

    def share(b, key):
        v = policy[b]["action_confidence"][key]
        return np.nan if v is None else 100 * v

    errors = np.array([share(b, "false_drop_mean") for b in budgets])
    errors_std = np.array([share(b, "false_drop_std") for b in budgets])
    targets = 100 * budgets

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7),
                                 gridspec_kw={"wspace": 0.34})
    a1.errorbar(targets, flat, yerr=flat_std, fmt="o-", capsize=2,
                color=PALETTE[2], label="label confidence")
    a1.errorbar(targets, aligned, yerr=aligned_std, fmt="s-", capsize=2,
                color=PALETTE[0], label="policy-aligned confidence")
    a1.set_xscale("log")
    a1.set_xlabel("Selective drop-error budget (%)")
    a1.set_ylabel("Candidate drop coverage (%)")
    a1.legend(frameon=False, fontsize=8)
    a1.grid(alpha=0.3, which="both")

    a2.errorbar(targets, errors, yerr=errors_std, fmt="s-", capsize=2,
                color=PALETTE[3], label="benign share among drops")
    lim = [targets.min() * 0.8, targets.max() * 1.2]
    a2.plot(lim, lim, "k--", lw=0.9, label="the budget")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("Selective drop-error budget (%)")
    a2.set_ylabel("Benign share among held-out drops (%)")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.grid(alpha=0.3, which="both")
    fig.savefig(out)
    plt.close(fig)


def table_bmv2_trials(bmv2, out: Path):
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Capture & Packets & Forwarded & Dropped & Escalated "
             r"& Reports \\", r"\midrule"]
    for name, t in bmv2["trials"].items():
        ev = t["pipeline_events"]
        lines.append(
            f"{CAPTURE_LABEL.get(name, name)} & {t['packets_sent']} & "
            f"{ev['fwd']} & {ev['drop']} & {ev['esc']} & "
            f"{t['reports_received']}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


# ── Figures ───────────────────────────────────────────────────────────

def figure_ablation(groups, out: Path):
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    width, x = 0.35, np.arange(2)
    for i, ent in enumerate(("none", "real")):
        means, stds = [], []
        for aug in (False, True):
            recs = groups.get((ent, aug, "flow-isolated"))
            m, s, _ = agg(recs, "macro_f1") if recs else (np.nan, 0, 0)
            means.append(100 * m)
            stds.append(100 * s)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
               color=PALETTE[i], hatch=HATCHES[i], edgecolor="black",
               linewidth=0.35, label=ENTROPY_LABEL[ent])
    ax.set_xticks(x)
    ax.set_xticklabels(["augmentation off", "augmentation on"])
    ax.set_ylabel("Macro F1 (%)")
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_context_controls(groups, out: Path):
    order = ("none", "shuffle", "class-marginal", "real")
    labels, means, stds = [], [], []
    for ent in order:
        recs = groups.get((ent, True, "flow-isolated"))
        if not recs:
            continue
        m, s, _ = agg(recs, "macro_f1")
        labels.append(ENTROPY_LABEL[ent].replace(" context", ""))
        means.append(100 * m)
        stds.append(100 * s)
    fig, ax = plt.subplots(figsize=(5.0, 2.7))
    ax.bar(labels, means, yerr=stds, capsize=3, color=PALETTE[0], width=0.62)
    ax.set_ylabel("Macro F1 (%)")
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_protocol(groups, out: Path):
    splits = ("random", "flow-isolated", "context-disjoint")
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    width, x = 0.35, np.arange(len(splits))
    for i, ent in enumerate(("none", "real")):
        means, stds = [], []
        for split in splits:
            recs = groups.get((ent, True, split))
            m, s, _ = agg(recs, "macro_f1") if recs else (np.nan, 0, 0)
            means.append(100 * m)
            stds.append(100 * s)
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=3,
               color=PALETTE[i], hatch=HATCHES[i], edgecolor="black",
               linewidth=0.35, label=ENTROPY_LABEL[ent])
    ax.set_xticks(x)
    ax.set_xticklabels([SPLIT_LABEL[s] for s in splits])
    ax.set_ylabel("Macro F1 (%)")
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_confusion(groups, out: Path):
    recs = groups.get(("real", True, "flow-isolated"))
    if not recs:
        return
    cm = np.mean([np.array(r["confusion_matrix"], dtype=float)
                  for r in recs], axis=0)
    cmn = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(8), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(8), CLASSES)
    ax.spines[:].set_visible(False)
    for i in range(8):
        for j in range(8):
            if cmn[i, j] > 0.005:
                ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if cmn[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, fraction=0.046)
    fig.savefig(out)
    plt.close(fig)


def figure_per_class(groups, out: Path):
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    width, x = 0.4, np.arange(len(CLASSES))
    for i, ent in enumerate(("none", "real")):
        recs = groups.get((ent, True, "flow-isolated"))
        if not recs:
            continue
        pc = per_class_f1(recs)
        means = [100 * pc.get(c, (np.nan, 0))[0] for c in CLASSES]
        stds = [100 * pc.get(c, (np.nan, 0))[1] for c in CLASSES]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=2,
               color=PALETTE[i], hatch=HATCHES[i], edgecolor="black",
               linewidth=0.35, label=ENTROPY_LABEL[ent])
    ax.set_xticks(x, CLASSES, rotation=25, ha="right")
    ax.set_ylabel("F1 (%)")
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_baselines(baselines, out: Path):
    """Every estimator on one protocol, with and without the channel."""
    models = [m for m in baselines.get("models", {}) if m != "MajorityClass"]
    if not models:
        return
    models.sort(key=lambda m: baselines["models"][m]["real"]["macro_f1_mean"])
    fig, ax = plt.subplots(figsize=(5.2, 2.7))
    width, x = 0.38, np.arange(len(models))
    for i, ctx in enumerate(("none", "real")):
        means = [100 * baselines["models"][m][ctx]["macro_f1_mean"]
                 for m in models]
        stds = [100 * baselines["models"][m][ctx]["macro_f1_std"]
                for m in models]
        ax.bar(x + (i - 0.5) * width, means, width, yerr=stds, capsize=2,
               color=PALETTE[i], hatch=HATCHES[i], edgecolor="black",
               linewidth=0.35, label=ENTROPY_LABEL[ctx])
    ax.set_xticks(x, models, rotation=18, ha="right")
    ax.set_ylabel("Macro F1 (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="y", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_cost(baselines, groups, bench_path: Path, out: Path):
    """Accuracy against CPU cost per flow, both measured on this corpus."""
    pts = []
    for name, entry in baselines.get("models", {}).items():
        ms = entry.get("real", {}).get("inference_ms_per_flow")
        if name == "MajorityClass" or not ms:
            continue
        pts.append(
            (name, ms, 100 * entry["real"]["macro_f1_mean"], "ensemble"))

    recs = groups.get(("real", True, "flow-isolated"))
    if recs and bench_path.exists():
        bench = {b["name"]: b for b in json.loads(bench_path.read_text())}
        entry = bench.get("BiMamba+Entropy (proposed)")
        if entry:
            pts.append(("BiMamba", entry["latency"]["per_sample_in_batch_ms"],
                        100 * agg(recs, "macro_f1")[0], "sequence"))
    if len(pts) < 2:
        return

    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    lo = min(p[1] for p in pts)
    hi = max(p[1] for p in pts)
    span = max(np.log10(hi / lo), 1e-9)
    label_offsets = {
        "LogisticRegression": (7, -2, "left"),
        "MLP": (-5, 9, "right"),
        "RandomForest": (-6, -14, "right"),
        "XGBoost": (7, -14, "left"),
        "LightGBM": (7, 9, "left"),
        "BiMamba": (-7, -2, "right"),
    }
    for name, ms, f1, kind in sorted(pts, key=lambda p: p[1]):
        seq = kind == "sequence"
        ax.scatter(ms, f1, s=46, zorder=3,
                   marker="D" if seq else "o",
                   color=PALETTE[3] if seq else PALETTE[0])
        dx, dy, align = label_offsets.get(name, (0, 8, "center"))
        ax.annotate(name, (ms, f1), textcoords="offset points",
                    xytext=(dx, dy), ha=align, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(lo / 3, hi * 3)
    ax.margins(y=0.25)
    ax.set_xlabel("CPU cost per flow (ms, log scale)")
    ax.set_ylabel("Macro F1 (%)")
    ax.grid(alpha=0.3, which="major")
    fig.savefig(out)
    plt.close(fig)


def figure_causal(study, out: Path):
    """What the switch estimator measures against what it should measure."""
    by_class = study.get("src_diversity_by_class") or {}
    sep = study.get("ddos_dos_separation") or {}
    if not by_class or not sep:
        return
    ids = sorted(by_class, key=lambda k: -by_class[k]["true_sources_mean"])
    names = [CLASSES[int(i)] for i in ids]
    truth = [by_class[i]["true_sources_mean"] for i in ids]
    est = [by_class[i]["switch_estimate_mean"] for i in ids]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(5.2, 2.5),
                                 gridspec_kw={"width_ratios": [2.1, 1]})
    width, x = 0.38, np.arange(len(ids))
    a1.bar(x - width / 2, truth, width, color=PALETTE[0],
           hatch=HATCHES[0], edgecolor="black", linewidth=0.35,
           label="ground truth")
    a1.bar(x + width / 2, est, width,
           color=PALETTE[1], hatch=HATCHES[1], edgecolor="black",
           linewidth=0.35, label="switch estimate")
    a1.set_xticks(x, names, rotation=25, ha="right")
    a1.set_ylabel("Mean distinct sources")
    a1.legend(frameon=False, fontsize=8, ncol=2, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    a1.grid(axis="y", alpha=0.3)

    for i, (key, lbl) in enumerate((("ground_truth", "ground truth"),
                                    ("switch_estimate", "switch estimate"))):
        d = sep.get(key, {})
        lo, hi = d.get("auc_ci95", [np.nan, np.nan])
        auc = d.get("auc", np.nan)
        a2.errorbar(auc, i, xerr=[[auc - lo], [hi - auc]], fmt="o",
                    color=PALETTE[i], capsize=3, markersize=5)
    a2.axvline(0.5, color="black", lw=0.8, ls="--")
    a2.set_yticks([0, 1], ["truth", "estimate"])
    a2.set_ylim(-0.6, 1.6)
    a2.set_xlabel("DDoS vs DoS AUC")
    a2.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def table_unsw(study, out: Path):
    """The same decomposition on a second corpus."""
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Estimator & No context & Shuffled & Oracle class-conditioned "
             r"& Measured \\", r"\midrule"]
    order = sorted(study["models"],
                   key=lambda m: -study["models"][m]
                   .get("real", {}).get("macro_f1_mean", 0))
    for name in order:
        e = study["models"][name]
        cells = []
        for c in ("none", "shuffle", "class-marginal", "real"):
            d = e.get(c)
            cells.append(fmt(d["macro_f1_mean"], d["macro_f1_std"])
                         if d else NA)
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def figure_unsw(study, out: Path):
    """Contrasts among the context interventions on UNSW-NB15."""
    models = [m for m in study["models"]
              if "decomposition" in study["models"][m]]
    if not models:
        return
    models.sort(key=lambda m: study["models"][m]["decomposition"]
                ["total_context_gain_pp"])

    def part(m, a, b):
        e = study["models"][m]
        return 100 * (e[a]["macro_f1_mean"] - e[b]["macro_f1_mean"])

    caps = np.array([part(m, "shuffle", "none") for m in models])
    prior = np.array([part(m, "class-marginal", "shuffle") for m in models])
    beyond = np.array([part(m, "real", "class-marginal") for m in models])

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    y = np.arange(len(models))
    # the components have mixed signs, so they are grouped rather than
    # stacked; a waterfall would draw the negative one back over the positive
    h = 0.26
    series = ((h, caps, PALETTE[2], "shuffled minus none"),
              (0, prior, PALETTE[1], "oracle minus shuffled"),
              (-h, beyond, PALETTE[3], "measured minus oracle"))
    for i, (off, vals, colour, lbl) in enumerate(series):
        ax.barh(y + off, vals, height=h * 0.92, color=colour,
                hatch=HATCHES[i], edgecolor="black", linewidth=0.35,
                label=lbl)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_yticks(y, models)
    ax.set_xlabel("Macro F1 contrast (percentage points)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(axis="x", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_risk(dp, out: Path):
    """What the switch may decide alone, per certified error rate."""
    pts = [p for p in (dp.get("risk_control") or {}).get("points", [])
           if p.get("threshold") is not None]
    if not pts:
        return
    eps = np.array([100 * p["target_error"] for p in pts])
    cov = np.array([100 * p["test_coverage"] for p in pts])
    err = np.array([100 * p["test_error_on_resolved"] for p in pts])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.7),
                                 gridspec_kw={"wspace": 0.32})
    a1.plot(eps, cov, "o-", ms=4, lw=1.4, color=PALETTE[0])
    a1.set_xlabel("Certified error rate (%)")
    a1.set_ylabel("Flows settled in switch (%)")
    a1.set_xscale("log")
    a1.grid(alpha=0.3, which="both")

    a2.plot(eps, err, "s-", ms=4, lw=1.4, color=PALETTE[3],
            label="error actually measured")
    lim = [eps.min() * 0.8, eps.max() * 1.2]
    a2.plot(lim, lim, "k--", lw=0.9, label="the guarantee")
    a2.set_xscale("log")
    a2.set_yscale("log")
    a2.set_xlabel("Certified error rate (%)")
    a2.set_ylabel("Error on settled flows (%)")
    a2.legend(frameon=False, fontsize=8, loc="upper left")
    a2.grid(alpha=0.3, which="both")
    fig.savefig(out)
    plt.close(fig)


def figure_operating(dp, out: Path):
    """What the escalation threshold buys and what it costs."""
    pts = [p for p in dp.get("operating_points", [])
           if p["accuracy_on_resolved"] is not None]
    if not pts:
        return
    res = np.array([100 * p["resolved_fraction"] for p in pts])
    onres = np.array([100 * p["accuracy_on_resolved"] for p in pts])
    comp = np.array([100 * p["composed_accuracy"] for p in pts])
    ctrl = 100 * dp["controller"]["accuracy"]

    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    ax.plot(res, onres, "o-", ms=3.5, lw=1.3, color=PALETTE[0],
            label="accuracy on flows the switch settles")
    ax.plot(res, comp, "s-", ms=3.5, lw=1.3, color=PALETTE[3],
            label="end-to-end accuracy of the composed system")
    ax.axhline(ctrl, color="black", lw=0.9, ls="--")
    ax.annotate("controller alone", (res.max(), ctrl), xytext=(-4, 5),
                textcoords="offset points", ha="right", fontsize=8)
    ax.set_xlabel("Flows settled in the switch (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.legend(frameon=False, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    ax.grid(alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_dataplane(bmv2, out: Path):
    """Packet disposition and escalation latency measured on the switch."""
    trials = bmv2.get("trials") or {}
    if not trials:
        return
    names = list(trials)
    labels = [CAPTURE_LABEL.get(n, n) for n in names]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.9), sharey=True,
                                 gridspec_kw={"width_ratios": [1.5, 1],
                                              "wspace": 0.08})
    y = np.arange(len(names))
    events = [trials[n].get("pipeline_events", {}) for n in names]
    totals = [max(sum(e.get(k, 0) for k in ("fwd", "drop", "esc")), 1)
              for e in events]
    left = np.zeros(len(names))
    for i, (colour, (k, lbl)) in enumerate(zip(
            (PALETTE[0], PALETTE[1], PALETTE[4]),
            (("fwd", "forwarded"), ("drop", "dropped"),
             ("esc", "escalated")))):
        vals = np.array([100 * e.get(k, 0) / t
                         for e, t in zip(events, totals)])
        a1.barh(y, vals, left=left, height=0.68, color=colour,
                hatch=HATCHES[i], edgecolor="black", linewidth=0.3,
                label=lbl)
        left += vals
    a1.set_yticks(y, labels)
    a1.set_xlabel("Packets (%)")
    a1.set_xlim(0, 100)
    a1.set_xticks([0, 25, 50, 75, 100])
    a1.legend(frameon=False, fontsize=8, ncol=3, loc="lower center",
              bbox_to_anchor=(0.5, 1.0))
    a1.invert_yaxis()

    lat = [trials[n].get("escalation_latency_ms", {}) for n in names]
    med = np.array([d.get("median", np.nan) for d in lat])
    p95 = np.array([d.get("p95", np.nan) for d in lat])
    err = np.nan_to_num(np.maximum(p95 - med, 0))
    a2.barh(y, med, height=0.68, xerr=[np.zeros(len(names)), err],
            color=PALETTE[3], capsize=2,
            error_kw={"lw": 0.8, "ecolor": "black"})
    # the escalation count is the sample size behind each latency bar
    for i, d in enumerate(lat):
        a2.annotate(f"n={d.get('n', 0)}", (p95[i] if np.isfinite(p95[i])
                                           else med[i], i),
                    xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8, color="black")
    a2.set_xlabel("Escalation latency (ms)")
    a2.set_xlim(0, float(np.nanmax(p95)) * 1.3)
    a2.locator_params(axis="x", nbins=5)
    a2.grid(axis="x", alpha=0.3)
    fig.savefig(out)
    plt.close(fig)


def figure_flow_length(corpus: Path, out: Path, seq_len=32):
    """Where the truncation length sits in the packet-count distribution."""
    if not corpus.exists():
        return
    try:
        import h5py
    except ImportError:
        return
    with h5py.File(corpus, "r") as f:
        n = f["sequences"].shape[0]
        counts = np.empty(n, dtype=np.int32)
        for i in range(0, n, 20000):
            blk = f["sequences"][i:i + 20000, :, 0]
            counts[i:i + 20000] = (blk != 0).sum(1)

    lo, hi = int(counts.min()), int(counts.max())
    fig, ax = plt.subplots(figsize=(5.4, 2.7))
    ax.hist(counts, bins=np.arange(lo, hi + 2), color=PALETTE[1],
            edgecolor="none")
    ax.set_yscale("log")
    ax.set_xlim(lo - 1, hi + 2)
    ax.set_xlabel("Packets per flow")
    ax.set_ylabel("Flows")
    ax.grid(axis="y", alpha=0.3)

    ax2 = ax.twinx()
    ax2.spines["right"].set_visible(True)
    xs = np.arange(lo, hi + 1)
    ordered = np.sort(counts)
    cdf = 100 * np.searchsorted(ordered, xs, side="right") / counts.size
    ax2.plot(xs, cdf, color=PALETTE[0], lw=1.5)
    ax2.set_ylabel("Cumulative flows (%)", color=PALETTE[0])
    ax2.set_ylim(0, 104)
    ax2.tick_params(axis="y", colors=PALETTE[0])

    covered = 100 * (counts <= seq_len).mean()
    ax.axvline(seq_len, color=PALETTE[3], lw=1.2, ls="--")
    ax2.annotate(f"truncation at {seq_len}: {covered:.1f}% complete",
                 (seq_len, covered), xytext=(12, -20),
                 textcoords="offset points", fontsize=8, color=PALETTE[3],
                 arrowprops=dict(arrowstyle="-", lw=0.6, color=PALETTE[3]))
    ax.annotate(f"recorded cap\nat {hi} packets", (hi, np.sum(counts == hi)),
                xytext=(-14, 16), textcoords="offset points", fontsize=8,
                ha="right", color="black",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="black"))
    fig.savefig(out)
    plt.close(fig)


def write_macros(summary, baselines, out: Path):
    """
    Emit the handful of figures the prose quotes, as LaTeX macros.

    Sentences in the abstract, introduction and conclusion cite these
    rather than literal numbers, so regenerating the results regenerates
    the prose and the two cannot disagree.
    """
    def pct(v):
        return f"{100 * v:.1f}\\%" if v is not None else "??"

    def pp(v):
        return _num(v) if v is not None else "??"

    def best(context):
        """Highest macro F1 over the ensembles at a given context."""
        vals = [(n, e[context]["macro_f1"])
                for n, e in summary.get("baselines", {}).items()
                if context in e and n not in ("MajorityClass",)]
        return max(vals, key=lambda kv: kv[1]) if vals else (None, None)

    best_ctx_name, best_ctx = best("real")
    _, best_noctx = best("none")
    est_gains = [100 * (e["real"]["macro_f1"] - e["none"]["macro_f1"])
                 for n, e in summary.get("baselines", {}).items()
                 if "real" in e and "none" in e and n != "MajorityClass"]

    ctx_diag = summary.get("context_diagnostics", {})
    ref = ctx_diag.get(REFERENCE_BASELINE, {})
    proto = summary.get("protocol_effects", {})
    eff = summary.get("main_effects", {})
    dp = summary.get("data_plane", {})
    cfg = summary.get("configurations", {})
    p4src = Path("p4/feature_extraction.p4")
    res = p4_resources(p4src) if p4src.exists() else {}

    def seq(key):
        return cfg.get(key, {}).get("macro_f1_mean")

    macros = {
        "BestEnsembleName": best_ctx_name or "??",
        "BestEnsembleCtx": pct(best_ctx),
        "BestEnsembleNoCtx": pct(best_noctx),
        "SeqBestCtx": pct(seq("real|True|flow-isolated")),
        "SeqBestNoCtx": pct(seq("none|True|flow-isolated")),
        "CtxGainEstMin": pp(min(est_gains) if est_gains else None),
        "CtxGainEstMax": pp(max(est_gains) if est_gains else None),
        "ContextEffectAug": pp(eff.get("context_effect_aug=True")),
        "ContextEffectNoAug": pp(eff.get("context_effect_aug=False")),
        "AugEffectNoCtx": pp(eff.get("augmentation_effect_context=none")),
        "AugEffectCtx": pp(eff.get("augmentation_effect_context=real")),
        "CtxGainTotal": pp(ref.get("total_context_gain_pp")),
        "CtxGainPerFlow": pp(ref.get("gain_beyond_class_marginal_pp")),
        "CtxGainClassPrior": pp(ref.get("gain_explained_by_class_marginal_pp")),
        "CtxGainShuffled": pp(ref.get("gain_from_shuffled_context_pp")),
        "CtxGainFromShuffle": pp(ref.get("gain_from_shuffled_context_pp")),
        "ProtoDisjointCtx": pp(proto.get(
            f"{REFERENCE_BASELINE}_context-disjoint_minus_flow_isolated_real_pp")),
        "ProtoDisjointNoCtx": pp(proto.get(
            f"{REFERENCE_BASELINE}_context-disjoint_minus_flow_isolated_none_pp")),
        "ProtoRandomCtx": pp(proto.get(
            f"{REFERENCE_BASELINE}_random_minus_flow_isolated_real_pp")),
        "DpResolved": pct((dp.get("deployment_policy") or {}).get("coverage")),
        "DpResolvedAcc": pct(1 - (dp.get("deployment_policy") or {}).get(
            "false_drop_rate", 1)),
        "RegisterArrays": str(res.get("n_registers", NA)),
        "RegisterState": (f"{res['register_bits'] / 8 / 1024:.0f}"
                          if res else NA),
        "DecisionPackets": str(
            summary.get("data_plane", {}).get("decision_packets", NA)),
        "DpAlone": pct(dp.get("accuracy")),
        "DpAloneMacroF": pct(dp.get("macro_f1")),
        "SysAccuracy": pct((dp.get("composed") or {}).get("accuracy")),
        "SysMacroF": pct((dp.get("composed") or {}).get("macro_f1")),
        "CtrlName": (dp.get("controller") or {}).get("model", NA),
        "CtrlAccuracy": pct((dp.get("controller") or {}).get("accuracy")),
        "RuleInstallMs": _rule_install_ms(),
        "PolicyBudget": pct((dp.get("deployment_policy") or {}).get(
            "false_drop_budget")),
        "PolicyCoverage": pct((dp.get("deployment_policy") or {}).get(
            "coverage")),
        "PolicyFalseDrop": pct((dp.get("deployment_policy") or {}).get(
            "false_drop_rate")),
        "PolicyEntries": str((dp.get("deployment_policy") or {}).get(
            "entries_installed", NA)),
        "PolicyFlatCoverage": _policy(dp, 0.01, "flat_cov"),
        "PolicyAlignedCoverage": _policy(dp, 0.01, "policy_cov"),
        "PolicyCoverageGain": _policy(dp, 0.01, "gain"),
        "PolicyFiveGain": _policy(dp, 0.05, "gain"),
        "PolicyHeldoutFalseDrop": _policy(dp, 0.01, "policy_err"),
        "CorpusFlows": _grouped(corpus_flows(Path("results/controlled"))),
        "ProtoResidual": _residual(proto),
    }

    lines = [r"% Generated by scripts/plot_results.py. Do not edit."]
    # a digit in a control sequence terminates the name and the rest leaks
    # into the document, so reject it here rather than in the LaTeX log
    bad = [k for k in macros if not k.isalpha()]
    assert not bad, f"macro names must be letters only: {bad}"
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k, v in macros.items()]
    out.write_text("\n".join(lines) + "\n")
    logger.info("Macros written to %s", out)


def _policy(dp, budget, field):
    """One field of the aggregated policy comparison."""
    policy = dp.get("policy_comparison") or {}
    point = next((v for k, v in policy.items()
                  if abs(float(k) - budget) < 1e-12), None)
    if point is None:
        return NA
    flat, aligned = point["label_confidence"], point["action_confidence"]
    if field == "flat_cov":
        return f"{100 * flat['coverage_mean']:.1f}\\%"
    if field == "policy_cov":
        return f"{100 * aligned['coverage_mean']:.1f}\\%"
    if field == "policy_err":
        v = aligned["false_drop_mean"]
        return NA if v is None else f"{100 * v:.2f}\\%"
    if field == "gain":
        return f"{point['coverage_gain_pp']:.1f}"
    return NA


def _rule_install_ms():
    """Measured latency of the controller's table write, from the switch run."""
    path = Path("results/bmv2_dataplane.json")
    if not path.exists():
        return NA
    m = json.loads(path.read_text()).get("mitigation") or {}
    v = m.get("rule_install_ms")
    return NA if v is None else f"{v:.0f}"


def corpus_flows(results_dir: Path):
    """
    Total flows retained by the subsample, read from any run's split report.

    Every run in a directory draws the same subsample, so the first readable
    report is authoritative. Returns None when no run has completed.
    """
    for path in sorted(results_dir.glob("*_results.json")):
        rep = json.loads(path.read_text()).get("protocol", {}).get(
            "split_report", {})
        sizes = [rep.get(k, {}).get("n") for k in ("train", "val", "test")]
        if all(isinstance(n, int) for n in sizes):
            return sum(sizes)
    return None


def _grouped(n):
    """Thousands separator written as a LaTeX-safe group."""
    return NA if n is None else f"{n:,}".replace(",", "{,}")


def _residual(proto):
    """
    Difference between context and no-context sensitivity to proxy grouping.

    This is descriptive because rounded context signatures do not recover
    every pair of overlapping source windows.
    """
    stem = f"{REFERENCE_BASELINE}_context-disjoint_minus_flow_isolated"
    with_ctx = proto.get(f"{stem}_real_pp")
    without = proto.get(f"{stem}_none_pp")
    if with_ctx is None or without is None:
        return NA
    return f"{abs(with_ctx - without):.1f}"


def table_causal(study, out: Path):
    """
    Source diversity as the switch measures it, against the truth.

    Separates two questions the offline analysis cannot: whether the
    in-switch estimator is accurate, and whether the quantity it estimates
    differs between the families it is supposed to separate.
    """
    per = study.get("src_diversity_by_class", {})
    recs = json.loads(Path(study["source"]).read_text())["records"]
    rng = np.random.RandomState(0)

    def ci(v):
        v = np.asarray(v, dtype=float)
        boot = v[rng.randint(0, len(v), (2000, len(v)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return f"{v.mean():.2f} [{lo:.2f}, {hi:.2f}]"

    def row(name, rs):
        est = [r["src_diversity"] for r in rs]
        tru = [r["true_victim_sources"] for r in rs]
        q1, q2, q3 = np.percentile(tru, [25, 50, 75])
        return (f"{name} & {len(rs)} & {ci(est)} & {ci(tru)} & "
                f"{q2:.0f} ({q1:.0f} to {q3:.0f})" + r" \\")

    lines = [
        r"\begin{tabular}{lrccc}", r"\toprule",
        r"Family & Flows & In-switch estimate & True distinct sources & "
        r"True median (IQR) \\",
        r"\midrule"]
    for cid in sorted(per, key=int):
        name = CLASSES[int(cid)] if int(cid) < len(CLASSES) else cid
        lines.append(row(name, [r for r in recs if r["label"] == int(cid)]))
    lines += [r"\midrule", row("Total", recs), r"\bottomrule",
              r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


# ── Revision: policy study, baselines, online validation ──────────────
# Every table and macro below is read from a stored result file. Figures are
# emitted as PGFPlots data files that the manuscript plots natively.

TARGETS = ("0.005", "0.01", "0.02", "0.05", "0.1")
SPLIT_NAME = {"flow-isolated": "Flow-isolated",
              "context-disjoint": "Signature-disjoint",
              "blocked": "Blocked", "random": "Random"}
# Result key of the reported calibration rule: the dependence rule chosen on
# development seeds. "blk" holds the earlier block rule, kept for comparison.
CAL = "dep"
OLD = "blk"


def _num(v, d=1, sign=False):
    """A number for LaTeX text; negatives and signed values in math mode."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return NA
    if round(v, d) == 0:
        v = 0.0
    if sign:
        return f"${v:+.{d}f}$"
    return f"${v:.{d}f}$" if v < 0 else f"{v:.{d}f}"


def _pc(v, d=1):
    return NA if v is None else _num(100 * v, d)


def _cis(c, d=1):
    lo, hi = c["ci95"]
    return f"[{_num(lo, d)}, {_num(hi, d)}]"


def _p(p):
    """A p-value for use inside math mode."""
    if p >= 0.001:
        return f"{p:.3f}"
    mant, exp = f"{p:.1e}".split("e")
    return mant + r"\times 10^{" + str(int(exp)) + "}"


def table_policy_main(study, out: Path, macros: dict):
    """Label against action admission, three partitioning rules. Misses are
    counted over flow records and, in parentheses, over connections."""
    s = study["summary"]
    lines = [r"\begin{tabular}{llrrrcrrr}", r"\toprule",
             r"Partition & Target & Label & Action & Gain (95\% CI) & Wins & "
             r"Label misses & Action misses & Frontier gain \\", r"\midrule"]

    def misses(g):
        conn = g.get(f"{CAL}_conn_misses")
        return (f"{g[f'{CAL}_misses']} ({conn})" if conn is not None
                else str(g[f"{CAL}_misses"]))
    for split in ("flow-isolated", "context-disjoint", "blocked"):
        if split not in s:
            continue
        b = s[split]["2-action"]
        for eps in TARGETS:
            gl, ga = b["gates"]["label"][eps], b["gates"]["action"][eps]
            c = b["contrasts"]["action-label"][eps][f"{CAL}_cov"]
            f = b["contrasts"]["action-label"][eps]["front_cov"]
            lines.append(
                f"{SPLIT_NAME[split]} & {100 * float(eps):.1f}\\% & "
                f"{_pc(gl[f'{CAL}_cov_mean'])} & "
                f"{_pc(ga[f'{CAL}_cov_mean'])} & "
                f"{_num(c['mean_points'], sign=True)} "
                f"{_cis(c)} & {c['wins']}/{c['n']}"
                f" & {misses(gl)} & {misses(ga)} & "
                f"{_num(f['mean_points'], sign=True)}" + r" \\")
        lines.append(r"\midrule" if split != "blocked" else "")
    lines = [ln for ln in lines if ln]
    if lines[-1] == r"\midrule":
        lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")

    fi = s["flow-isolated"]["2-action"]
    key = {"0.005": "Half", "0.01": "One", "0.02": "Two", "0.05": "Five",
           "0.1": "Ten"}
    for eps, tag in key.items():
        gl, ga = fi["gates"]["label"][eps], fi["gates"]["action"][eps]
        c = fi["contrasts"]["action-label"][eps]
        macros.update({
            f"PolLabel{tag}": _pc(gl[f"{CAL}_cov_mean"]) + r"\%",
            f"PolAction{tag}": _pc(ga[f"{CAL}_cov_mean"]) + r"\%",
            f"PolGain{tag}": _num(c[f"{CAL}_cov"]["mean_points"]),
            f"PolGainCI{tag}": _cis(c[f"{CAL}_cov"]),
            f"PolGainCI{tag}Low": _num(c[f"{CAL}_cov"]["ci95"][0]),
            f"PolWins{tag}": str(c[f"{CAL}_cov"]["wins"]),
            f"PolMissLabel{tag}": str(gl[f"{CAL}_misses"]),
            f"PolMissAction{tag}": str(ga[f"{CAL}_misses"]),
            f"PolBinLabel{tag}": _pc(gl["cal_cov_mean"]) + r"\%",
            f"PolBinAction{tag}": _pc(ga["cal_cov_mean"]) + r"\%",
            f"PolBinGain{tag}": _num(c["cal_cov"]["mean_points"]),
            f"PolBinGainCI{tag}": _cis(c["cal_cov"]),
            f"PolBinMissLabel{tag}": str(gl["cal_misses"]),
            f"PolBinMissAction{tag}": str(ga["cal_misses"]),
            f"PolFrontLabel{tag}": _pc(gl["front_cov_mean"]) + r"\%",
            f"PolFrontAction{tag}": _pc(ga["front_cov_mean"]) + r"\%",
            f"PolFrontGain{tag}": _num(c["front_cov"]["mean_points"]),
            f"PolFrontGainCI{tag}": _cis(c["front_cov"]),
            f"PolFrontWins{tag}": str(c["front_cov"]["wins"]),
            f"PolP{tag}": _p(c[f"{CAL}_cov"]["wilcoxon_p"]),
        })
    for split, tag in (("context-disjoint", "Sig"), ("blocked", "Blk")):
        b = s[split]["2-action"]
        for eps, t in key.items():
            c = b["contrasts"]["action-label"][eps]
            macros[f"Pol{tag}Gain{t}"] = _num(c[f"{CAL}_cov"]["mean_points"])
            macros[f"Pol{tag}GainCI{t}"] = _cis(c[f"{CAL}_cov"])
            macros[f"Pol{tag}Wins{t}"] = str(c[f"{CAL}_cov"]["wins"])
            macros[f"Pol{tag}FrontGain{t}"] = _num(
                c["front_cov"]["mean_points"])
            macros[f"Pol{tag}MissAction{t}"] = str(
                b["gates"]["action"][eps][f"{CAL}_misses"])
            macros[f"Pol{tag}MissLabel{t}"] = str(
                b["gates"]["label"][eps][f"{CAL}_misses"])
            macros[f"Pol{tag}Label{t}"] = _pc(
                b["gates"]["label"][eps][f"{CAL}_cov_mean"]) + r"\%"
            macros[f"Pol{tag}Action{t}"] = _pc(
                b["gates"]["action"][eps][f"{CAL}_cov_mean"]) + r"\%"
    # The earlier block rule and the dependence rule over every partition,
    # action map, gate and target of the study (30 settings per rule).
    for split, tag in (("flow-isolated", "Fi"), ("context-disjoint", "Sig"),
                       ("blocked", "Blk")):
        for rule, rt in ((OLD, "Old"), (CAL, "Dep")):
            cells = [s[split][m]["gates"][g][eps] for m in s[split]
                     for g in ("label", "action") for eps in TARGETS]
            macros[f"Rule{rt}{tag}Misses"] = str(sum(
                e[f"{rule}_misses"] for e in cells))
            macros[f"Rule{rt}{tag}Over"] = str(sum(
                e[f"{rule}_misses"] > 3 for e in cells))
            macros[f"Rule{rt}{tag}Cov"] = _pc(np.mean(
                [e[f"{rule}_cov_mean"] for e in cells])) + r"\%"
        macros[f"Rule{tag}Settings"] = str(len(cells))
        macros[f"Rule{tag}Evaluations"] = _grouped(
            len(cells) * cells[0]["n_seeds"])
        for m, mt in (("2-action", "Two"), ("3-action", "Three"),
                      ("4-action", "Four")):
            macros[f"RuleDep{tag}Misses{mt}"] = str(sum(
                s[split][m]["gates"][g][eps][f"{CAL}_misses"]
                for g in ("label", "action") for eps in TARGETS))
        conn = [e.get(f"{CAL}_conn_misses") for e in cells]
        if None not in conn:
            macros[f"RuleDep{tag}ConnMisses"] = str(sum(conn))
            macros[f"RuleDep{tag}ConnOver"] = str(sum(c > 3 for c in conn))
        # Per evaluation: does one connection carry the test errors of a miss?
        ev = [r["gates"][m][g][eps] for r in study["per_seed"]
              if r["split"] == split for m in r["gates"]
              for g in ("label", "action") for eps in TARGETS]
        if ev and f"{CAL}_k_te_top" in ev[0]:
            miss = [e for e in ev if not e[f"{CAL}_met"]]
            met = [e for e in ev if e[f"{CAL}_met"] and e[f"{CAL}_k_te"]]

            def share(xs):
                return np.median([e[f"{CAL}_k_te_top"] / e[f"{CAL}_k_te"]
                                  for e in xs]) if xs else None
            macros[f"Conn{tag}Rescued"] = str(sum(e[f"{CAL}_conn_met"]
                                                  for e in miss))
            macros[f"Conn{tag}TopShareMissed"] = _pc(share(miss), 0) + r"\%"
            macros[f"Conn{tag}TopShareMet"] = _pc(share(met), 0) + r"\%"
    for eps, t in key.items():
        for split, tag in (("blocked", "Blk"),):
            for g in ("label", "action"):
                e = s[split]["2-action"]["gates"][g][eps]
                macros[f"Old{tag}{g.capitalize()}{t}"] = _pc(
                    e[f"{OLD}_cov_mean"]) + r"\%"
                macros[f"Old{tag}Miss{g.capitalize()}{t}"] = str(
                    e[f"{OLD}_misses"])
    cfg = study["config"].get("dependence_rule", {})
    macros["WalkStart"] = _grouped(cfg.get("walk_start_flows", 0))
    seeds = study["config"]["seeds"]
    macros.update({"SeedFirst": str(seeds[0]), "SeedLast": str(seeds[-1]),
                   "NSeeds": str(len(seeds)),
                   "KeyLevels": str(study["config"]["key_levels"]),
                   "BootBlocks": str(study["config"]["block_bootstrap"]["blocks"]),
                   "BootResamples": str(
                       study["config"]["block_bootstrap"]["resamples"])})


def table_calibration(study, out: Path):
    """Scores against calibration rules, flow-isolated: the oracle frontier,
    the exact binomial bound and the dependence bound, with paired gains."""
    b = study["summary"]["flow-isolated"]["2-action"]
    lines = [r"\begin{tabular}{rrrrrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{No calibration (frontier)} & "
             r"\multicolumn{3}{c}{Exact binomial bound} & "
             r"\multicolumn{3}{c}{Dependence bound} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Target & Label & Action & Gain & Label & Action & Gain & "
             r"Label & Action & Gain \\", r"\midrule"]
    for eps in TARGETS:
        gl, ga = b["gates"]["label"][eps], b["gates"]["action"][eps]
        c = b["contrasts"]["action-label"][eps]
        lines.append(
            f"{100 * float(eps):.1f}\\% & {_pc(gl['front_cov_mean'])} & "
            f"{_pc(ga['front_cov_mean'])} & "
            f"{_num(c['front_cov']['mean_points'], sign=True)} & "
            f"{_pc(gl['cal_cov_mean'])} ({gl['cal_misses']}) & "
            f"{_pc(ga['cal_cov_mean'])} ({ga['cal_misses']}) & "
            f"{_num(c['cal_cov']['mean_points'], sign=True)} & "
            f"{_pc(gl[f'{CAL}_cov_mean'])} ({gl[f'{CAL}_misses']}) & "
            f"{_pc(ga[f'{CAL}_cov_mean'])} ({ga[f'{CAL}_misses']}) & "
            f"{_num(c[f'{CAL}_cov']['mean_points'], sign=True)}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


# Vertical offset of each calibration rule within a target's row of the
# dumbbell chart; the frontier sits on top because the axis is reversed.
CALIBRATION_RULES = (("front", -0.27), ("cal", 0.0), (CAL, 0.27))


def data_calibration(study, out: Path):
    """PGFPlots data for the calibration figure, one file per rule (frontier,
    exact binomial, dependence): per target, its row position, label and
    action coverage, the paired gain with its interval and the action-gate
    misses."""
    b = study["summary"]["flow-isolated"]["2-action"]
    for r, offset in CALIBRATION_RULES:
        rows = ["target ypos label action gain lo hi miss"]
        for i, eps in enumerate(TARGETS):
            gl, ga = b["gates"]["label"][eps], b["gates"]["action"][eps]
            g = b["contrasts"]["action-label"][eps][f"{r}_cov"]
            miss = ga.get(f"{r}_misses", 0) if r != "front" else 0
            rows.append(f"{100 * float(eps):g} {i + offset:.2f} "
                        f"{100 * gl[f'{r}_cov_mean']:.2f} "
                        f"{100 * ga[f'{r}_cov_mean']:.2f} "
                        f"{g['mean_points']:.2f} {g['ci95'][0]:.2f} "
                        f"{g['ci95'][1]:.2f} {miss}")
        out.with_name(f"{out.stem}_{r}{out.suffix}").write_text(
            "\n".join(rows) + "\n")


def data_controller_sensitivity(c, out: Path):
    """PGFPlots data for the controller sensitivity heat map: one cell per
    configuration in row-major order (trees across, leaves down) with the
    difference in matched recall from the deployed configuration and the
    text printed in the cell. Semicolon-separated, because the text holds
    commas."""
    s, ref = c["summary"], c["config"]["reference"]
    rows = ["x;y;diff;kb;text"]
    for yi, leaves in enumerate(c["config"]["grid"]["leaves"]):
        for xi, trees in enumerate(c["config"]["grid"]["trees"]):
            e = s[f"{trees}x{leaves}"]
            d = e["matched_recall_minus_ref_0.01"]
            kb = e["model_bytes"] / 1000
            text = ("deployed" if f"{trees}x{leaves}" == ref else
                    f"{_num(d['mean_points'], sign=True)} {_cis(d)}")
            rows.append(f"{xi};{yi};{d['mean_points']:.2f};{kb:.0f};"
                        f"{text}\\\\{kb:.0f} kB")
    out.write_text("\n".join(rows) + "\n")


BASELINE_ROWS = (
    ("Deployable in the switch", None),
    ("Cell table, label gate", "table_label"),
    ("Cell table, action gate (ours)", "table_action"),
    ("Cell table, cost-sensitive plug-in$^{a}$", "table_costsens"),
    ("Range-match tree, binary", "tree_binary"),
    ("Range-match tree, label gate", "tree_label"),
    ("Range-match tree, action gate", "tree_action"),
    ("Controller-grade references", None),
    ("GBDT, label gate", "gbdt_label"),
    ("GBDT, action gate", "gbdt_action"),
    ("GBDT, binary (calibrated)", "gbdt_binary"),
    ("GBDT, cost-sensitive plug-in$^{a}$", "gbdt_costsens"),
    ("Multi-task MLP, label head", "mtl_label"),
    ("Multi-task MLP, action head", "mtl_action"),
)


def table_policy_baselines(bl, out: Path, macros: dict):
    """The action gate against binary, multi-task and cost-sensitive rules."""
    s = bl["summary"]
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{1\% target} & \multicolumn{2}{c}{5\% target} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Admission rule & Calibrated (misses) & Frontier & "
             r"Calibrated (misses) & Frontier \\", r"\midrule"]
    for label, key in BASELINE_ROWS:
        if key is None:
            if lines[-1] != r"\midrule":
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{5}}{{l}}{{\emph{{{label}}}}} \\")
            continue
        cells = []
        for eps in ("0.01", "0.05"):
            e = s[key][eps]
            if f"{CAL}_cov_mean" in e:
                cells += [f"{_pc(e[f'{CAL}_cov_mean'])} ({e[f'{CAL}_misses']})",
                          _pc(e["front_cov_mean"])]
            else:
                cells += [f"{_pc(e['plug_cov_mean'])} ({e['plug_misses']})",
                          NA]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")

    tag = {"0.01": "One", "0.05": "Five"}
    for key in ("table_label", "table_action", "table_costsens", "tree_binary",
                "tree_label", "tree_action", "gbdt_label", "gbdt_action",
                "gbdt_binary", "gbdt_costsens", "mtl_label", "mtl_action",
                "table6_action", "table8_action", "table12_action",
                "table6_label", "table8_label", "table12_label"):
        name = "Bl" + "".join(p.capitalize() for p in key.split("_"))
        name = name.replace("6", "Six").replace("8", "Eight").replace(
            "12", "Twelve")
        for eps, t in tag.items():
            e = s[key][eps]
            if f"{CAL}_cov_mean" in e:
                macros[f"{name}{t}"] = _pc(e[f"{CAL}_cov_mean"]) + r"\%"
                macros[f"{name}Miss{t}"] = str(e[f"{CAL}_misses"])
                macros[f"{name}Front{t}"] = _pc(e["front_cov_mean"]) + r"\%"
                macros[f"{name}Bin{t}"] = _pc(e["cal_cov_mean"]) + r"\%"
                macros[f"{name}BinMiss{t}"] = str(e["cal_misses"])
                conn = [r["methods"][key][eps].get(f"{CAL}_conn_met")
                        for r in bl["per_seed"]]
                if None not in conn:
                    macros[f"{name}ConnMiss{t}"] = str(
                        sum(not c for c in conn))
            else:
                macros[f"{name}{t}"] = _pc(e["plug_cov_mean"]) + r"\%"
                macros[f"{name}Miss{t}"] = str(e["plug_misses"])
            c = e.get(f"table_action_minus_{CAL}")
            if c:
                macros[f"{name}Delta{t}"] = _num(c["mean_points"])
                macros[f"{name}DeltaCI{t}"] = _cis(c)
                macros[f"{name}DeltaWins{t}"] = str(c["wins"])
            f = e.get("table_action_minus_front")
            if f:
                macros[f"{name}FrontDelta{t}"] = _num(f["mean_points"])
            if f"{CAL}_cov_mean" in e:
                # Seed-to-seed stability of the calibrated coverage.
                v = np.array([r["methods"][key][eps][f"{CAL}_cov"]
                              for r in bl["per_seed"]])
                macros[f"{name}Sd{t}"] = _pc(v.std())
                macros[f"{name}Min{t}"] = _pc(v.min())
                macros[f"{name}Zero{t}"] = str(int((v == 0).sum()))

    # Paired within-family contrasts over seeds (points, 95% bootstrap CI).
    rng = np.random.RandomState(0)
    pairs = {"GbdtActBin": ("gbdt_action", "gbdt_binary"),
             "TreeActBin": ("tree_action", "tree_binary"),
             "TableActTreeBin": ("table_action", "tree_binary"),
             "GbdtActLab": ("gbdt_action", "gbdt_label"),
             "TreeActLab": ("tree_action", "tree_label"),
             "MtlActLab": ("mtl_action", "mtl_label"),
             "TableActLab": ("table_action", "table_label")}
    for pname, (a, c) in pairs.items():
        for eps, t in tag.items():
            for field, ft in (("front_cov", "Front"), (f"{CAL}_cov", "Blk")):
                d = 100 * np.array([r["methods"][a][eps][field]
                                    - r["methods"][c][eps][field]
                                    for r in bl["per_seed"]])
                boot = d[rng.randint(0, len(d), (2000, len(d)))].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                macros[f"Pair{pname}{ft}{t}"] = _num(d.mean())
                macros[f"Pair{pname}{ft}CI{t}"] = f"[{_num(lo)}, {_num(hi)}]"
                macros[f"Pair{pname}{ft}Wins{t}"] = str(int((d > 0).sum()))


def table_policy_generality(study, unsw, out: Path, macros: dict, grouped=None):
    """Richer action maps and a second corpus, 1% target, dependence rule."""
    rows = []
    s = study["summary"]
    for split in ("flow-isolated", "context-disjoint"):
        for m in ("2-action", "3-action", "4-action"):
            rows.append((f"CICIoT2023, {m}", SPLIT_NAME[split], s[split][m]))
        if grouped:
            rows.append(("CICIoT2023, flood-inspection", SPLIT_NAME[split],
                         grouped["summary"][split]["flood-inspect"]))
    if unsw:
        for m in ("2-action", "4-action"):
            rows.append((f"UNSW-NB15, {m}", "Random",
                        unsw["summary"]["random"][m]))
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"& & \multicolumn{4}{c}{Single 1\% target} & "
             r"\multicolumn{2}{c}{Tiered budgets$^{a}$} \\",
             r"\cmidrule(lr){3-6}\cmidrule(lr){7-8}",
             r"Corpus and policy & Partition & Label & HSC & Binary & "
             r"Action & Label & Action \\", r"\midrule"]
    for name, rule, b in rows:
        g = b["gates"]

        def cov(k):
            e = g[k]["0.01"]
            return f"{_pc(e[f'{CAL}_cov_mean'])} ({e[f'{CAL}_misses']})"
        tl = b.get(f"tiered_{CAL}", {})

        def tiered(k):
            if k not in tl:
                return NA
            miss = sum(tl[k]["misses_by_action"].values())
            return f"{_pc(tl[k]['coverage_mean'])} ({miss})"
        lines.append(
            f"{name} & {rule} & "
            f"{cov('label')} & {cov('hsc')} & {cov('binary')} & {cov('action')}"
            f" & {tiered('label')} & {tiered('action')}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")

    sources = [(study, "2-action", "Two"), (study, "3-action", "Three"),
               (study, "4-action", "Four")]
    if grouped:
        sources += [(grouped, "flood-inspect", "FloodInsp"),
                    (grouped, "flood-ratelimit", "FloodRate")]
    for split, st in (("flow-isolated", ""), ("context-disjoint", "Sig"),
                      ("blocked", "Blk")):
        for src, m, mt in sources:
            b = src["summary"][split][m]
            for eps, t in (("0.01", "One"), ("0.05", "Five")):
                for gname in ("label", "action", "hsc", "binary"):
                    macros[f"Map{st}{mt}{gname.capitalize()}{t}"] = _pc(
                        b["gates"][gname][eps][f"{CAL}_cov_mean"]) + r"\%"
                    macros[f"Map{st}{mt}Miss{gname.capitalize()}{t}"] = str(
                        b["gates"][gname][eps][f"{CAL}_misses"])
                    macros[f"Map{st}{mt}OldMiss{gname.capitalize()}{t}"] = str(
                        b["gates"][gname][eps][f"{OLD}_misses"])
                    conn = b["gates"][gname][eps].get(f"{CAL}_conn_misses")
                    if conn is not None:
                        macros[f"Map{st}{mt}ConnMiss{gname.capitalize()}{t}"] = \
                            str(conn)
                c = b["contrasts"]["action-label"][eps][f"{CAL}_cov"]
                macros[f"Map{st}{mt}Gain{t}"] = _num(c["mean_points"])
                macros[f"Map{st}{mt}GainCI{t}"] = _cis(c)
            macros[f"Map{st}{mt}Within"] = _pc(
                b.get("within_action_share_mean"), 0) + r"\%"
            for tk, tt in ((f"tiered_{CAL}", ""), (f"tiered_{OLD}", "Old")):
                if tk not in b:
                    continue
                for gname in b[tk]:
                    e = b[tk][gname]
                    macros[f"Map{st}{mt}{tt}Tiered{gname.capitalize()}"] = _pc(
                        e["coverage_mean"]) + r"\%"
                    macros[f"Map{st}{mt}{tt}TieredMiss{gname.capitalize()}"] = str(
                        sum(e["misses_by_action"].values()))
                    for a, at in (("1", "Drop"), ("2", "Rate"), ("3", "Redir")):
                        macros[f"Map{st}{mt}{tt}TieredMiss{gname.capitalize()}{at}"] = \
                            str(e["misses_by_action"].get(a, 0))
                    if "conn_misses_by_action" in e:
                        macros[f"Map{st}{mt}{tt}TieredConnMiss{gname.capitalize()}"] = \
                            str(sum(e["conn_misses_by_action"].values()))
        # How often the milder actions of the tiered action gate admit any
        # flow at all, and how much.
        for src, m, mt in sources[1:]:
            per = [r for r in src["per_seed"] if r["split"] == split]
            tiered = [r[f"tiered_{CAL}"][m] for r in per]
            for ref, rt in (("label", ""), ("binary", "Bin")):
                if not tiered or ref not in tiered[0]:
                    continue
                d = 100 * np.array([t["action"]["coverage"]
                                    - t[ref]["coverage"] for t in tiered])
                boot = d[np.random.RandomState(0).randint(
                    0, len(d), (2000, len(d)))].mean(axis=1)
                macros.update({
                    f"Map{st}{mt}TieredGain{rt}": _num(d.mean()),
                    f"Map{st}{mt}TieredGain{rt}CI": _cis(
                        {"ci95": np.percentile(boot, [2.5, 97.5])}),
                    f"Map{st}{mt}TieredGain{rt}Wins": str(int((d > 0).sum()))})
            for a, at in (("1", "Drop"), ("2", "Rate"), ("3", "Redir")):
                v = [r[f"tiered_{CAL}"][m]["action"]["per_action"].get(a)
                     for r in per]
                v = [x for x in v if x]
                if v:
                    macros[f"Map{st}{mt}TieredSeeds{at}"] = str(
                        sum(x["coverage"] > 0 for x in v))
                    macros[f"Map{st}{mt}TieredCov{at}"] = _pc(
                        np.mean([x["coverage"] for x in v]), 1) + r"\%"
                    err = [x["error"] for x in v
                           if x["coverage"] > 0 and x.get("error") is not None]
                    if err:
                        macros[f"Map{st}{mt}TieredErr{at}"] = _pc(
                            np.mean(err), 1) + r"\%"
                # The first validation set the walk tests for that action.
                s0 = [x["start"] for x in v if "start" in x]
                if s0:
                    err = [x["error"] for x in s0]
                    wrong = np.sum([x["wrong_by_true_action"] for x in s0],
                                   axis=0)
                    macros.update({
                        f"Map{st}{mt}StartErr{at}": _pc(np.mean(err), 1)
                        + r"\%",
                        f"Map{st}{mt}StartErrMin{at}": _pc(min(err), 1),
                        f"Map{st}{mt}StartErrMax{at}": _pc(max(err), 1)
                        + r"\%",
                        f"Map{st}{mt}StartBound{at}": _pc(
                            np.mean([x["bound"] for x in s0]), 1) + r"\%",
                        f"Map{st}{mt}StartBenign{at}": _pc(
                            wrong[0] / max(wrong.sum(), 1), 0) + r"\%"})
                    for k, kt in enumerate(("Benign", "Drop", "Rate", "Redir")):
                        macros[f"Map{st}{mt}Start{at}Wrong{kt}"] = _pc(
                            wrong[k] / max(wrong.sum(), 1), 0) + r"\%"
    if unsw:
        u = unsw["summary"]["random"]
        g2 = [u["2-action"]["contrasts"]["action-label"][e][f"{CAL}_cov"]["mean_points"]
              for e in TARGETS]
        macros["UnswMapTwoGainMin"] = _num(min(g2))
        macros["UnswMapTwoGainMax"] = _num(max(g2))
        for m, mt in (("2-action", "Two"), ("4-action", "Four")):
            b = u[m]
            macros[f"UnswMap{mt}Within"] = _pc(
                b.get("within_action_share_mean"), 0) + r"\%"
            for eps, t in (("0.005", "Half"), ("0.01", "One"), ("0.05", "Five"),
                           ("0.1", "Ten")):
                c = b["contrasts"]["action-label"][eps][f"{CAL}_cov"]
                macros[f"UnswMap{mt}Gain{t}"] = _num(c["mean_points"])
                macros[f"UnswMap{mt}GainCI{t}"] = _cis(c)
                macros[f"UnswMap{mt}Wins{t}"] = str(c["wins"])
                macros[f"UnswMap{mt}Label{t}"] = _pc(
                    b["gates"]["label"][eps][f"{CAL}_cov_mean"]) + r"\%"
                macros[f"UnswMap{mt}Action{t}"] = _pc(
                    b["gates"]["action"][eps][f"{CAL}_cov_mean"]) + r"\%"
                macros[f"UnswMap{mt}MissAction{t}"] = str(
                    b["gates"]["action"][eps][f"{CAL}_misses"])
                macros[f"UnswMap{mt}FrontAction{t}"] = _pc(
                    b["gates"]["action"][eps]["front_cov_mean"]) + r"\%"


def data_policy_curves(study, unsw, out: Path):
    """
    Coverage against target for both gates, mean and 95% CI of the mean,
    one PGFPlots table per panel (``out`` is a stem: policy_<panel>.dat).
    """
    rng = np.random.RandomState(0)
    panels = [("fi", study, "flow-isolated"), ("sig", study, "context-disjoint"),
              ("blk", study, "blocked")]
    if unsw:
        panels.append(("unsw", unsw, "random"))
    for tag, src, split in panels:
        per = [r for r in src["per_seed"] if r["split"] == split]
        rows = ["target label labello labelhi action actionlo actionhi"]
        for eps in TARGETS:
            vals = {}
            for g in ("label", "action"):
                v = 100 * np.array([r["gates"]["2-action"][g][eps][f"{CAL}_cov"]
                                    for r in per])
                boot = v[rng.randint(0, len(v), (2000, len(v)))].mean(axis=1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                vals[g] = (v.mean(), lo, hi)
            rows.append(f"{100 * float(eps):g} "
                        + " ".join(f"{x:.2f}" for g in ("label", "action")
                                   for x in vals[g]))
        (out.parent /
         f"{out.stem}_{tag}.dat").write_text("\n".join(rows) + "\n")


def _read(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


def table_online(recon, tiered, out: Path, macros: dict, meter=None):
    """Offline, emulator and BMv2 verdicts on the same timed trace."""
    v = recon["bmv2_validation"]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Quantity & Offline & Emulator & BMv2 \\", r"\midrule",
             rf"\multicolumn{{4}}{{l}}{{\emph{{Two-action table, "
             rf"{v['flows']} held-out flows, {v['packets']} packets, "
             rf"{v['duration_s']:.0f}\,s}}}} \\"]
    lines.append(f"Flows decided & 100.0 & {_pc(v['decided_emulator'])} & "
                 f"{_pc(v['decided_bmv2'])}" + r" \\")
    lines.append("Fast-path coverage & " + " & ".join(
        _pc(v[k]["coverage"]) for k in ("offline", "emulator", "bmv2")) + r" \\")
    lines.append("Selective drop error & " + " & ".join(
        _pc(v[k]["selective_error"], 2) for k in ("offline", "emulator", "bmv2"))
        + r" \\")
    lines.append(f"Verdict agreement with BMv2 & {_pc(v['action_agreement_bmv2_offline'])}"
                 f" & {_pc(v['verdict_agreement_bmv2_emulator'])} & 100.0" + r" \\")
    macros.update({
        "OnFlows": str(v["flows"]), "OnPackets": _grouped(v["packets"]),
        "OnDuration": f"{v['duration_s']:.0f}",
        "OnCovOffline": _pc(v["offline"]["coverage"]) + r"\%",
        "OnCovBmv": _pc(v["bmv2"]["coverage"]) + r"\%",
        "OnErrOffline": _pc(v["offline"]["selective_error"], 2) + r"\%",
        "OnErrBmv": _pc(v["bmv2"]["selective_error"], 2) + r"\%",
        "OnAgreeEmu": _pc(v["verdict_agreement_bmv2_emulator"], 2) + r"\%",
        "OnAgreeOffline": _pc(v["action_agreement_bmv2_offline"], 2) + r"\%",
        "OnDecided": _pc(v["decided_bmv2"], 2) + r"\%",
        "OnPolicyRegs": _pc(v["policy_registers_identical"]) + r"\%",
        "OnCells": _pc(v["report_cells_identical"]) + r"\%",
        "OnLateP": f"{v['replay']['send_lateness_ms']['p99']:.2f}",
    })
    pr = v.get("per_register") or {}
    timing = ("mean_iat_us", "iat_min_us", "iat_max_us", "duration_us")
    if pr:
        macros["OnStaticIdentical"] = _pc(min(
            e["identical"] for k, e in pr.items() if k not in timing)) + r"\%"
        med = [pr[k]["abs_diff_p50"] for k in timing]
        macros["OnTimingMedian"] = f"{min(med)} to {max(med)}"
        macros["OnTimingPNinetyNine"] = (
            f"{max(pr[k]['abs_diff_p99'] for k in timing) / 1000:.1f}")
    diffs = v.get("verdict_differences", [])
    macros["OnVerdictDiffs"] = str(len(diffs))
    macros["OnVerdictDiffsAttackDrop"] = str(
        sum(d["label"] != 0 and d["bmv2"] >= 2 for d in diffs))
    if tiered:
        t = tiered["bmv2_validation"]
        lines.append(rf"\midrule \multicolumn{{4}}{{l}}{{\emph{{Four-action "
                     rf"function-test table, same trace}}}} \\")
        names = {"1": "drop (1\\%)", "2": "rate limit (5\\%)",
                 "3": "redirect (10\\%)"}
        for a, nm in names.items():
            cells = []
            for k in ("offline", "emulator", "bmv2"):
                e = t[k]["per_action"].get(a)
                cells.append(NA if e is None else
                             f"{_pc(e['coverage'])} / {_pc(e['error'], 1)}")
            lines.append(
                f"Coverage / error, {nm} & " + " & ".join(cells) + r" \\")
            e = t["bmv2"]["per_action"].get(a)
            if e:
                tag = ["", "Drop", "Rate", "Redir"][int(a)]
                macros[f"TierCov{tag}"] = _pc(e["coverage"]) + r"\%"
                macros[f"TierErr{tag}"] = _pc(e["error"]) + r"\%"
                macros[f"TierFlows{tag}"] = str(
                    round(e["coverage"] * t["flows"]))
        lines.append(f"Action agreement with BMv2 & "
                     f"{_pc(t['action_agreement_bmv2_offline'])} & "
                     f"{_pc(t['verdict_agreement_bmv2_emulator'])} & 100.0" + r" \\")
        ev = t["pipeline_events"]
        lines.append(f"Rate-limited packets passed / dropped & & & "
                     f"{ev['rate_limit_pass']} / {ev['rate_limit_drop']}" + r" \\")
        lines.append(f"Frames at the inspection port & & & "
                     f"{t.get('frames_at_inspection_port')}" + r" \\")
        if meter:
            mv = meter["pipeline_events"]
            lines.append(f"Meter check: packets discarded / metered & & & "
                         f"{mv['rate_limit_drop']} / "
                         f"{mv['rate_limit_drop'] + mv['rate_limit_pass']}" + r" \\")
        macros.update({
            "TierAgreeEmu": _pc(t["verdict_agreement_bmv2_emulator"]) + r"\%",
            "TierAgreeOffline": _pc(t["action_agreement_bmv2_offline"]) + r"\%",
            "TierCov": _pc(t["bmv2"]["coverage"]) + r"\%",
            "TierRatePass": str(ev["rate_limit_pass"]),
            "TierRateDrop": str(ev["rate_limit_drop"]),
            "TierRedirected": str(ev["redirect"]),
            "TierInspectFrames": str(t.get("frames_at_inspection_port")),
        })
    cl = recon.get("closed_loop")
    if cl:
        lines.append(r"\midrule \multicolumn{4}{l}{\emph{Closed loop: switch "
                     r"plus report adjudicator}} \\")
        lines.append(f"Attack flows stopped (at the switch) & & "
                     f"{_pc(cl['expected_attack_stopped'])} & "
                     f"{_pc(cl['attack_stopped'])} ({_pc(cl['attack_stopped_at_switch'])})"
                     + r" \\")
        lines.append(f"Benign flows stopped & & {_pc(cl['expected_benign_stopped'], 2)}"
                     f" & {_pc(cl['benign_stopped'], 2)}" + r" \\")
        lines.append(f"Per-flow agreement with emulator & & & "
                     f"{_pc(cl['agreement_with_expected'])}" + r" \\")
        ri = cl["rule_install_ms"]
        lines.append(f"Rule install, median / p95 (ms) & & & "
                     f"{ri['p50']:.1f} / {ri['p95']:.1f}" + r" \\")
        lines.append(f"Later packets of blocked flows dropped by rules & & & "
                     f"{cl['of_which_dropped_by_installed_rules']} of "
                     f"{cl['post_decision_packets_of_blocked_flows']}" + r" \\")
        macros.update({
            "ClAttack": _pc(cl["attack_stopped"]) + r"\%",
            "ClAttackSwitch": _pc(cl["attack_stopped_at_switch"]) + r"\%",
            "ClBenign": _pc(cl["benign_stopped"], 2) + r"\%",
            "ClAgree": _pc(cl["agreement_with_expected"]) + r"\%",
            "ClInstallMed": f"{ri['p50']:.1f}", "ClInstallP": f"{ri['p95']:.1f}",
            "ClSwitchDrops": str(cl["switch_drop_flows"]),
            "ClDeferred": str(cl["deferred_flows"]),
            "ClBlocked": str(cl["controller_blocked_flows"]),
            "ClRules": str(cl["mitigation_entries"]),
            "ClPostDropped": str(cl["of_which_dropped_by_installed_rules"]),
            "ClPost": str(cl["post_decision_packets_of_blocked_flows"]),
            "ClBudget": _pc(cl["benign_block_budget"], 0) + r"\%",
            "ClDisagree": str(cl.get("disagreements")),
            "ClDisagreeDeferred": str(
                cl.get("disagreements_on_flows_deferred_in_both")),
        })
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def table_load(recon, out: Path, data: Path, macros: dict):
    """One-way against two-choice flow table as flow arrivals grow."""
    sw = recon["load_sweep"]
    rates = sorted({r["rate_flows_per_s"] for r in sw})
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"& \multicolumn{3}{c}{One-way index} & "
             r"\multicolumn{3}{c}{Two-choice table} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
             r"Flows/s & Undecided & Coverage & Agreement & Undecided & "
             r"Coverage & Agreement \\", r"\midrule"]
    rows = ["rate u1 c1 a1 u2 c2 a2 off"]
    for rate in rates:
        cells, vals = [], []
        for ways in (1, 2):
            sub = [r for r in sw if r["rate_flows_per_s"] == rate
                   and r["ways"] == ways]
            u = np.mean([r["never_decided"] for r in sub])
            c = np.mean([r["online_coverage"] for r in sub])
            a = np.mean([r["verdict_agreement"] for r in sub])
            cells += [_pc(u), _pc(c), _pc(a)]
            vals += [100 * u, 100 * c, 100 * a]
        off = np.mean([r["offline_coverage"] for r in sw
                       if r["rate_flows_per_s"] == rate])
        lines.append(f"{rate:g} & " + " & ".join(cells) + r" \\")
        rows.append(f"{rate:g} " + " ".join(f"{x:.3f}" for x in vals)
                    + f" {100 * off:.3f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    data.write_text("\n".join(rows) + "\n")
    hi, lo = max(rates), min(rates)
    for ways, t in ((1, "One"), (2, "Two")):
        for rate, rt in ((hi, "High"), (lo, "Low")):
            sub = [r for r in sw if r["rate_flows_per_s"]
                   == rate and r["ways"] == ways]
            macros[f"Load{t}Undecided{rt}"] = _pc(
                np.mean([r["never_decided"] for r in sub])) + r"\%"
            macros[f"Load{t}Cov{rt}"] = _pc(
                np.mean([r["online_coverage"] for r in sub])) + r"\%"
    macros["LoadOffline"] = _pc(
        np.mean([r["offline_coverage"] for r in sw])) + r"\%"
    macros["LoadHigh"] = _grouped(int(hi))
    macros["LoadSeeds"] = str(len({r["seed"] for r in sw}))


# The 60 MB capture heads, one per attack family, in the order of the
# families (replay_attack.SLICE_LABEL names them).
HEAD_CAPTURES = (
    ("DDoS-SYN_Flood1", "DDoS SYN flood", "Ddos"),
    ("DoS-TCP_Flood1", "DoS TCP flood", "Dos"),
    ("Mirai-udpplain1", "Mirai UDP flood", "Mirai"),
    ("Recon-PortScan", "Recon port scan", "Recon"),
    ("SqlInjection", "Web, SQL injection", "Sql"),
    ("VulnerabilityScan", "Web, vulnerability scan", "Vuln"),
    ("DNS_Spoofing", "Spoofing DNS", "Dns"),
    ("MITM-ArpSpoofing1", "Spoofing ARP", "Arp"),
    ("DictionaryBruteForce", "BruteForce", "Brute"),
)


def table_capture_heads(runs: dict, out: Path, macros: dict):
    """
    What the system does on the first 60 MB of one capture per attack
    family, from the emulator at recorded timing (results/capture/heads):
    how many flows reach the decision packet, what the switch verdicts
    stop on their own, the flood alerts the new-flow counter raises, the
    entries the controller installs on them and on deferred flows, and
    what the switch and the entries stop together, with the entries taking
    effect at the median install latency measured on BMv2.
    """
    lines = [r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
             r"& & & & & \multicolumn{1}{c}{Switch} & "
             r"\multicolumn{4}{c}{Flood alerts and entries} & \\",
             r"\cmidrule(lr){6-6}\cmidrule(lr){7-10}",
             r"Capture & Packets & Flows & Flows/s & $\geq$10 pkts & Stopped & "
             r"Alerts & First (s) & Sources & Entries & Stopped \\",
             r"\midrule"]
    tot = defaultdict(int)
    n_rows = 0
    for name, label, tag in HEAD_CAPTURES:
        d = runs.get(name)
        if d is None:
            continue
        n_rows += 1
        a = d["all"]["all"]
        e = a["entries"]["p50"]
        e95 = a["entries"]["p95"]
        m = d["entries_model"]["p50"]
        elig = a["flows_reaching_decision"] / a["flows"]
        elig_p = a["packets_in_decidable_flows"] / a["packets"]
        rate = a["flows"] / a["span_s"]
        stopped = a["emulator_fast_attack_packets"] / a["attack_packets"]
        decided = a["emulator_fast"] + a["emulator_deferred"]
        entries = m["pair_entries"] + m["victim_entries"] + m["flow_entries"]
        first = ""
        if e["first_alert_s"] is not None:
            first = (f"{e['first_alert_s']:.3f}" if e["first_alert_s"] < 1
                     else f"{e['first_alert_s']:.1f}")
        lines.append(
            f"{label} & {_grouped(a['packets'])} & {_grouped(a['flows'])} & "
            f"{_grouped(int(round(rate)))} & {_pc(elig, 1)} & {_pc(stopped, 2)} & "
            f"{e['flood_alerts']} & {first} & {e['alert_sources']} & "
            f"{_grouped(entries)} & {_pc(e['dropped_share'], 1)}" + r" \\")
        for k in ("flows", "packets", "flows_reaching_decision",
                  "packets_in_decidable_flows", "emulator_fast",
                  "emulator_deferred", "emulator_controller_blocks",
                  "emulator_fast_attack_packets", "attack_packets",
                  "emulator_reports"):
            tot[k] += a[k]
        tot["alerts"] += e["flood_alerts"]
        tot["entries"] += entries
        tot["stopped_entries"] += e["dropped_packets"]["total"]
        tot["stopped_entries_p95"] += e95["dropped_packets"]["total"]
        bins = a["length_bins"]
        short = bins["1"]["flows"] + bins["2-9"]["flows"]
        macros.update({
            f"Head{tag}Flows": _grouped(a["flows"]),
            f"Head{tag}Packets": _grouped(a["packets"]),
            f"Head{tag}Span": f"{a['span_s']:.0f}",
            f"Head{tag}FlowsPerS": _grouped(int(round(rate))),
            f"Head{tag}Eligible": _pc(elig, 1) + r"\%",
            f"Head{tag}EligiblePackets": _pc(elig_p, 1) + r"\%",
            f"Head{tag}ShortFlows": _pc(short / a["flows"], 1) + r"\%",
            f"Head{tag}Decided": str(decided),
            f"Head{tag}DecidedShare": _pc(
                decided / a["flows_reaching_decision"], 1) + r"\%"
            if a["flows_reaching_decision"] else NA,
            f"Head{tag}Fast": str(a["emulator_fast"]),
            f"Head{tag}Deferred": str(a["emulator_deferred"]),
            f"Head{tag}Blocked": str(a["emulator_controller_blocks"]),
            f"Head{tag}Stopped": _pc(stopped, 2) + r"\%",
            f"Head{tag}StoppedBytes": _pc(
                a["emulator_fast_attack_bytes"] / a["attack_bytes"], 2) + r"\%",
            f"Head{tag}ReportsPerS": f"{a['emulator_reports_per_s']:.1f}",
            f"Head{tag}Unseen": _pc(
                d["all"]["not_in_corpus"]["flows"] / a["flows"], 1) + r"\%",
            # Flood alerts and the controller's entries.
            f"Head{tag}Alerts": str(e["flood_alerts"]),
            f"Head{tag}FirstAlert": f"{e['first_alert_s'] * 1e3:.0f}"
            if e["first_alert_s"] is not None and e["first_alert_s"] < 1
            else (f"{e['first_alert_s']:.1f}" if e["first_alert_s"] is not None
                  else NA),
            f"Head{tag}FirstAlertUnit": "ms"
            if e["first_alert_s"] is not None and e["first_alert_s"] < 1 else "s",
            f"Head{tag}AlertSources": str(e["alert_sources"]),
            f"Head{tag}AlertVictims": str(e["alert_victims"]),
            f"Head{tag}PairEntries": str(m["pair_entries"]),
            f"Head{tag}VictimEntries": str(m["victim_entries"]),
            f"Head{tag}FlowEntries": str(m["flow_entries"]),
            f"Head{tag}Entries": _grouped(entries),
            f"Head{tag}FloodEntryPackets": _grouped(
                e["dropped_packets"]["flood_entries"]),
            f"Head{tag}FlowEntryPackets": _grouped(
                e["dropped_packets"]["flow_entries"]),
            f"Head{tag}StoppedEntries": _pc(e["dropped_share"], 1) + r"\%",
            f"Head{tag}StoppedEntriesP": _pc(e95["dropped_share"], 1) + r"\%",
            f"Head{tag}PassedAfter": _grouped(e["passed_after_first_entry"])
            if e["passed_after_first_entry"] is not None else NA,
        })
        if e["flood_alerts"]:
            last = max(x["t_s"] for x in m["alerts"])
            macros[f"Head{tag}LastAlert"] = (
                f"{last * 1e3:.0f}" if last < 1 else f"{last:.1f}")
            macros[f"Head{tag}LastAlertUnit"] = "ms" if last < 1 else "s"
    if tot["flows"]:
        lines += [r"\midrule",
                  f"All & {_grouped(tot['packets'])} & {_grouped(tot['flows'])} & & "
                  f"{_pc(tot['flows_reaching_decision'] / tot['flows'], 1)} & "
                  f"{_pc(tot['emulator_fast_attack_packets'] / tot['attack_packets'], 2)} & "
                  f"{tot['alerts']} & & & {_grouped(tot['entries'])} & "
                  f"{_pc(tot['stopped_entries'] / tot['packets'], 1)}"
                  + r" \\"]
        macros.update({
            "HeadCaptures": str(n_rows),
            "HeadFlows": _grouped(tot["flows"]),
            "HeadPackets": _grouped(tot["packets"]),
            "HeadEligible": _pc(tot["flows_reaching_decision"] / tot["flows"], 1) + r"\%",
            "HeadEligiblePackets": _pc(
                tot["packets_in_decidable_flows"] / tot["packets"], 1) + r"\%",
            "HeadDecided": _grouped(tot["emulator_fast"] + tot["emulator_deferred"]),
            "HeadFast": str(tot["emulator_fast"]),
            "HeadBlocked": str(tot["emulator_controller_blocks"]),
            "HeadStopped": _pc(
                tot["emulator_fast_attack_packets"] / tot["attack_packets"], 2) + r"\%",
            "HeadAlerts": str(tot["alerts"]),
            "HeadEntries": _grouped(tot["entries"]),
            "HeadStoppedEntries": _pc(tot["stopped_entries"] / tot["packets"], 1) + r"\%",
            "HeadStoppedEntriesP": _pc(
                tot["stopped_entries_p95"] / tot["packets"], 1) + r"\%",
        })
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def flood_alert_facts(benign, heads, bmv2: dict, macros: dict):
    """
    The per-victim new-flow counter: its level against the window peaks of
    the benign captures and of the attack heads that raise no alert
    (results/flood), and the BMv2 runs of the paced flood heads against
    the emulator's expectation (results/capture/heads/*_30k_1ms.json).
    """
    if benign:
        caps = benign["captures"]
        peaks = {n: c["peak_new_flows_per_window"] for n, c in caps.items()}
        unseen = [peaks[n] for n in peaks if n.startswith("BenignTraffic")]
        iot = [peaks[n] for n in peaks if n.startswith("honeypot")]
        macros.update({
            "FloodLevel": _grouped(benign["level_min"]),
            "FloodWindow": f"{benign['window_us'] / 1e6:.0f}",
            "FloodCapacity": _grouped(int(round(
                benign["flow_slots"] / 60))),
            "FloodSlots": _grouped(benign["flow_slots"]),
            "FloodBenignCaptures": str(len(caps)),
            "FloodBenignPeak": str(max(peaks.values())),
            "FloodBenignAlerts": str(sum(c["alerts"] for c in caps.values())),
            "FloodBenignDevPeak": str(peaks.get("benign", 0)),
            "FloodBenignUnseenPeak": str(max(unseen)) if unseen else NA,
            "FloodIotPeak": str(max(iot)) if iot else NA,
            "FloodIotCaptures": str(len(iot)),
            "FloodBenignMargin": f"{benign['level_min'] / max(peaks.values()):.0f}",
        })
    if heads:
        caps = heads["captures"]
        quiet = {n: c["peak_new_flows_per_window"] for n, c in caps.items()
                 if c["alerts"] == 0}
        loud = {n: c["peak_new_flows_per_window"] for n, c in caps.items()
                if c["alerts"]}
        macros.update({
            "FloodQuietHeads": str(len(quiet)),
            "FloodQuietPeak": str(max(quiet.values())) if quiet else NA,
            "FloodLoudHeads": str(len(loud)),
            "FloodLoudPeakMin": _grouped(min(loud.values())) if loud else NA,
        })
        for name, _, tag in HEAD_CAPTURES:
            if name in caps:
                macros[f"Head{tag}PeakNewFlows"] = _grouped(
                    caps[name]["peak_new_flows_per_window"])
    for d in bmv2.values():
        if "flood_alert" in d:
            macros["FloodSourcesMax"] = str(d["flood_alert"]["sources_max"])
            break
    for name, tag in (("DoS-TCP_Flood1", "Dos"), ("DDoS-SYN_Flood1", "Ddos")):
        d = bmv2.get(f"{name}_30k_1ms")
        if not d:
            continue
        a = d["all"]["all"]
        e = a["entries"]["p50"]
        m = d["entries_model"]["p50"]
        b = d["closed_loop"]
        hits = b["mitigation_hits_by_kind"]
        same_sources = sorted(m["alert_sources"]) == sorted(
            b["flood_alert_sources"])
        macros.update({
            f"BmvHead{tag}Packets": _grouped(a["packets"]),
            f"BmvHead{tag}Pace": f"{d['min_gap_us'] / 1e3:.0f}",
            f"BmvHead{tag}ModelAlerts": str(e["flood_alerts"]),
            f"BmvHead{tag}Alerts": str(b["flood_alerts"]),
            f"BmvHead{tag}Sources": str(len(b["flood_alert_sources"])),
            f"BmvHead{tag}SameSources": "the same" if same_sources else "different",
            f"BmvHead{tag}Hits": _grouped(hits.get("flood", 0)),
            f"BmvHead{tag}ModelDrops": _grouped(
                e["dropped_packets"]["flood_entries"]),
            f"BmvHead{tag}SwitchDrops": _grouped(
                b["pipeline_events"]["drop"] - sum(hits.values())),
            f"BmvHead{tag}ModelSwitchDrops": _grouped(
                e["dropped_packets"]["switch"]),
            f"BmvHead{tag}Forwarded": _grouped(b["pipeline_events"]["fwd"]),
            f"BmvHead{tag}ModelForwarded": _grouped(
                a["packets"] - e["dropped_packets"]["total"]),
            f"BmvHead{tag}InstallMax": f"{b['install_ms'][-1]:.0f}"
            if b["install_ms"] else NA,
            f"BmvHead{tag}InstallNext": f"{max(b['install_ms'][:-1]):.0f}"
            if len(b["install_ms"]) > 1 else NA,
        })


SAT_CAP_TAGS = {"100": "Hundred", "50": "Fifty", "30": "Thirty", "20": "Twenty",
                "10": "Ten", "5": "Five", "2": "Two"}
SAT_RATE_TAGS = {"20": "Twenty", "100": "Hundred", "1000": "OneK",
                 "2000": "TwoK", "5000": "FiveK", "10000": "TenK",
                 "20000": "TwentyK"}


def table_saturation(d, live, out: Path, macros: dict):
    """
    The controller under load (results/controller_saturation.json): the
    held-out mix at 100 flows per second through the closed-loop emulator
    with one controller worker behind a bounded queue, capped below the
    report rate; the same worker at rising flow arrival rates; and a
    spoofed-source flood of single-packet flows with the new-flow alert
    off, on with three limits on the sources named per victim, and on
    with a capped controller. ``live`` holds the capped BMv2 runs of the
    same trace (results/bmv2_saturation.json).
    """
    def _f(x, k="%.1f"):
        return NA if x is None else k % x

    def _cap_key(c):
        return str(int(float(c))) if float(c) == int(float(c)) else str(c)

    caps = {_cap_key(k): v for k, v in d["rate_caps"].items()}
    base = caps["0"]
    q0 = base["queue"]
    macros.update({
        "SatFlows": _grouped(d["flows"]),
        "SatBaseRate": _grouped(int(d["base_rate_flows_per_s"])),
        "SatDepth": str(d["queue_depth"]),
        "SatScoreUs": f"{d['scorer_us']['p50']:.0f}",
        "SatScoreUsP": f"{d['scorer_us']['p95']:.0f}",
        "SatInstallMs": f"{d['install_us']['p50'] / 1e3:.1f}",
        "SatReports": _grouped(q0["offered"]),
        "SatOfferedPerS": f"{q0['offered_per_s']:.1f}",
        "SatPeakPerS": _grouped(q0["peak_per_s"]),
        "SatBusy": _pc(q0["busy_share"], 1) + r"\%",
        "SatWindow": f"{q0['report_window_s']:.0f}",
        "SatUncappedAttack": _pc(base["attack_stopped"]) + r"\%",
        "SatUncappedSwitch": _pc(base["attack_stopped_at_switch"]) + r"\%",
        "SatUncappedUnmitigated": _pc(base["attack_deferred_unmitigated"]) + r"\%",
        "SatUncappedBenign": _pc(base["benign_blocked"], 2) + r"\%",
        "SatUncappedBlocked": str(base["controller_blocked"]),
        "SatUncappedDecided": _pc(base["decided"]) + r"\%",
        "SatControllerGain": _num(100 * (base["attack_stopped"]
                                         - base["attack_stopped_at_switch"]), 1),
    })
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"\multicolumn{7}{l}{\emph{Controller capped below the report "
             r"stream (\SatOfferedPerS{} reports/s offered, peaks of "
             r"\SatPeakPerS{}/s, queue of \SatDepth{})}} \\",
             r"Cap (reports/s) & Lost (\%) & Wait p95 (ms) & Attack stopped (\%) "
             r"& Unmitigated (\%) & Benign blocked (\%) & Forwarded \\",
             r"\midrule"]
    for key, label in (("0", "none"),) + tuple((k, k) for k in SAT_CAP_TAGS):
        r = caps.get(key)
        if r is None:
            continue
        q = r["queue"]
        lines.append(
            f"{label} & {_pc(q['lost_share'])} & {_f(q['wait_ms']['p95'] if q['wait_ms'] else None)} & "
            f"{_pc(r['attack_stopped'])} & {_pc(r['attack_deferred_unmitigated'])} & "
            f"{_pc(r['benign_blocked'], 2)} & "
            f"{_grouped(r['packets']['deferred_forwarded_after_decision'])}" + r" \\")
        t = SAT_CAP_TAGS.get(key, "None")
        macros.update({
            f"SatCap{t}Lost": _pc(q["lost_share"]) + r"\%",
            f"SatCap{t}Wait": _f(q["wait_ms"]["p95"] if q["wait_ms"] else None),
            f"SatCap{t}Attack": _pc(r["attack_stopped"]) + r"\%",
            f"SatCap{t}Unmitigated": _pc(r["attack_deferred_unmitigated"]) + r"\%",
            f"SatCap{t}Benign": _pc(r["benign_blocked"], 2) + r"\%",
            f"SatCap{t}Forwarded": _grouped(
                r["packets"]["deferred_forwarded_after_decision"]),
        })
    # Flow arrival rates with the measured worker: text macros only.
    rates = {_cap_key(k): v for k, v in d["flow_rates"].items()}
    unb = {_cap_key(k): v for k, v in d.get("flow_rates_unbounded", {}).items()}
    no_loss = [float(k) for k, v in rates.items() if v["queue"]["lost"] == 0]
    macros["SatRateNoLossMax"] = _grouped(int(max(no_loss))) if no_loss else NA
    for key, t in SAT_RATE_TAGS.items():
        r = rates.get(key)
        if r is None:
            continue
        q = r["queue"]
        macros.update({
            f"SatRate{t}Peak": _grouped(q["peak_per_s"]),
            f"SatRate{t}Lost": _pc(q["lost_share"]) + r"\%",
            f"SatRate{t}Busy": _pc(q["busy_share"], 1) + r"\%",
            f"SatRate{t}Decided": _pc(r["decided"]) + r"\%",
            f"SatRate{t}Attack": _pc(r["attack_stopped"]) + r"\%",
            f"SatRate{t}Switch": _pc(r["attack_stopped_at_switch"]) + r"\%",
            f"SatRate{t}Benign": _pc(r["benign_blocked"], 2) + r"\%",
        })
        u = unb.get(key)
        if u is not None:
            macros[f"SatRate{t}WaitUnbounded"] = _f(u["queue"]["wait_ms"]["p95"], "%.0f")
            macros[f"SatRate{t}QueueUnbounded"] = _grouped(u["queue"]["peak_queue"])
    # Spoofed-source flood.
    fl = d["flood"]
    frates = sorted({float(k.split("_")[0]) for k in fl})
    hi, lo = max(frates), min(frates)

    def fk(rate, tag):
        return fl.get(f"{rate:g}_{tag}")
    lines += [r"\midrule",
              r"\multicolumn{7}{l}{\emph{Spoofed-source flood of single-packet "
              r"flows at \SatFloodRate{} flows/s for \SatFloodDuration{}\,s "
              r"over the same mix}} \\",
              r"Alert, sources limit & Alerts & Victim entry (s) & "
              r"Flood stopped (\%) & Attack stopped (\%) & Benign blocked (\%) & \\",
              r"\midrule"]
    cases = (("no_alert", "off", "Off"), ("alert", "on, 16", "On"),
             ("alert_sources8", "on, 8", "Eight"), ("alert_sources4", "on, 4", "Four"),
             ("alert_capped", "on, 16, controller capped", "Capped"))
    for tag, label, t in cases:
        r = fk(hi, tag)
        if r is None:
            continue
        f = r["flood"]
        lines.append(
            f"{label} & {f['alerts']} & {_f(f['victim_entry_s'], '%.2f')} & "
            f"{_pc(f['stopped_share'])} & {_pc(r['attack_stopped'])} & "
            f"{_pc(r['benign_blocked'], 2)} & " + r"\\")
        macros.update({
            f"SatFlood{t}Alerts": str(f["alerts"]),
            f"SatFlood{t}Victim": _f(f["victim_entry_s"], "%.2f"),
            f"SatFlood{t}Stopped": _pc(f["stopped_share"]) + r"\%",
            f"SatFlood{t}StoppedPackets": _grouped(
                f["stopped_by_pair_entries"] + f["stopped_by_victim_entry"]),
            f"SatFlood{t}Forwarded": _grouped(f["forwarded"]),
            f"SatFlood{t}Attack": _pc(r["attack_stopped"]) + r"\%",
            f"SatFlood{t}Benign": _pc(r["benign_blocked"], 2) + r"\%",
            f"SatFlood{t}Decided": _pc(r["decided"]) + r"\%",
        })
    on = fk(hi, "alert")
    macros.update({
        "SatFloodRate": _grouped(int(hi)),
        "SatFloodLowRate": _grouped(int(lo)),
        "SatFloodFlows": _grouped(on["flood"]["flows"]) if on else NA,
        "SatFloodDuration": f"{on['flood']['flows'] / hi:.0f}" if on else NA,
        "SatFloodFirstAlert": f"{on['flood']['first_alert_s'] * 1e3:.0f}"
        if on and on["flood"]["first_alert_s"] is not None else NA,
        "SatFloodPairEntries": str(on["flood"]["pair_entries"]) if on else NA,
        "SatFloodCap": f"{fk(hi, 'alert_capped')['queue']['rate_cap']:g}"
        if fk(hi, "alert_capped") else NA,
    })
    low = fk(lo, "alert")
    if low is not None:
        macros.update({
            "SatFloodLowVictim": _f(low["flood"]["victim_entry_s"], "%.2f"),
            "SatFloodLowStopped": _pc(low["flood"]["stopped_share"]) + r"\%",
            "SatFloodLowAlerts": str(low["flood"]["alerts"]),
        })
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    # Live BMv2 runs under the same caps.
    if live:
        macros["SatLiveRuns"] = str(len(live["runs"]))
        for name, r in live["runs"].items():
            tag = {"uncapped": "Uncapped", "cap30": "ThirtyCap",
                   "cap10": "TenCap", "cap2": "TwoCap"}.get(name)
            if tag is None:
                continue
            qs = r.get("controller_queue") or {}
            e = r.get("emulator_under_cap") or {}

            def _sec(ms):
                if ms is None:
                    return NA
                return f"{ms / 1e3:.2f}" if ms < 1000 else f"{ms / 1e3:.1f}"
            macros.update({
                f"SatLive{tag}WaitS": _sec(r["queue_wait_ms"]["p95"]
                                           if r.get("queue_wait_ms") else None),
                f"SatLive{tag}ExpWaitS": _sec(e["wait_ms"]["p95"]
                                              if e.get("wait_ms") else None),
            })
            macros.update({
                f"SatLive{tag}Attack": _pc(r["attack_stopped"]) + r"\%",
                f"SatLive{tag}Switch": _pc(r["attack_stopped_at_switch"]) + r"\%",
                f"SatLive{tag}Gain": _num(100 * (r["attack_stopped"]
                                                - r["attack_stopped_at_switch"]), 1),
                f"SatLive{tag}Benign": _pc(r["benign_stopped"], 2) + r"\%",
                f"SatLive{tag}Blocked": str(r["controller_blocked_flows"]),
                f"SatLive{tag}Lost": _pc(qs["lost"] / qs["offered"]) + r"\%"
                if qs.get("offered") else NA,
                f"SatLive{tag}Offered": _grouped(qs.get("offered")),
                f"SatLive{tag}Wait": _f(r["queue_wait_ms"]["p95"] if r.get("queue_wait_ms") else None),
                f"SatLive{tag}Install": _f(r["rule_install_ms"]["p50"] if r.get("rule_install_ms") else None),
                f"SatLive{tag}Agree": _pc(r["agreement_with_expected"]) + r"\%",
                f"SatLive{tag}ExpAttack": _pc(e.get("attack_stopped")) + r"\%",
                f"SatLive{tag}ExpLost": _pc(e.get("reports_lost_share")) + r"\%",
                f"SatLive{tag}ExpBlocked": str(e.get("controller_blocked_flows", NA)),
                f"SatLive{tag}ExpWait": _f(e["wait_ms"]["p95"] if e.get("wait_ms") else None, "%.0f"),
                f"SatLive{tag}Deferred": str(r["deferred_flows"]),
                f"SatLive{tag}Undecided": str(live["flows"] - r["switch_drop_flows"]
                                             - r["deferred_flows"]),
            })
        macros["SatLiveProcessed"] = _pc(
            live["runs"]["uncapped"]["pipeline_events"] and
            sum(live["runs"]["uncapped"]["pipeline_events"].values())
            / live["runs"]["uncapped"]["replay"]["packets_sent"]) + r"\%" \
            if "uncapped" in live["runs"] else NA
def table_capture(cap, out: Path, macros: dict, paced=None):
    """Real capture slices replayed through BMv2 with the adjudicator, at
    recorded timing and, when given, paced to the switch's loss-free rate."""
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"& & & & \multicolumn{3}{c}{Recorded timing} & "
             r"\multicolumn{2}{c}{Agreement} \\",
             r"\cmidrule(lr){5-7}\cmidrule(lr){8-9}",
             r"Capture & Packets & Flows & $\geq$10 packets & Fast path & "
             r"Deferred & Controller blocks & Recorded & Paced \\", r"\midrule"]

    def paced_agree(name):
        if not paced:
            return NA
        r = paced["per_capture"].get(name) if name else paced["all"]["all"]
        return _pc((r.get("open_loop") or {}).get("agreement_on_decidable_flows"))
    for name, r in cap["per_capture"].items():
        cl = r.get("closed_loop") or r.get("open_loop") or {}
        ol = r.get("open_loop") or {}
        lines.append(
            f"{CAPTURE_LABEL.get(name, name.replace('_', ' '))} & {r['packets']} & "
            f"{r['flows']} & {r['flows_reaching_decision']} & {ol.get('fast', NA)} & "
            f"{ol.get('deferred', NA)} & {cl.get('controller_blocked', NA)} & "
            f"{_pc(ol.get('agreement_on_decidable_flows'))} & {paced_agree(name)}"
            + r" \\")
    a = cap["all"]["all"]
    ol, cl = a.get("open_loop", {}), a.get("closed_loop", {})
    lines += [r"\midrule",
              f"All & {a['packets']} & {a['flows']} & {a['flows_reaching_decision']} & "
              f"{ol.get('fast', NA)} & {ol.get('deferred', NA)} & "
              f"{cl.get('controller_blocked', NA)} & "
              f"{_pc(ol.get('agreement_on_decidable_flows'))} & {paced_agree(None)}"
              + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    if paced:
        p = paced["all"]["all"]
        pol, pcl = p.get("open_loop", {}), p.get("closed_loop", {})
        macros.update({
            "CapPacedGap": f"{paced.get('min_gap_us', 0) / 1000:g}",
            "CapPacedRate": _grouped(int(round(1e6 / paced["min_gap_us"])))
            if paced.get("min_gap_us") else NA,
            "CapPacedDuration": f"{paced['trace']['duration_s']:.0f}",
            "CapPacedProcessed": _grouped(paced["open_loop"]["pipeline_packets"]),
            "CapPacedProcessedShare": _pc(paced["open_loop"]["pipeline_packets"]
                                          / paced["trace"]["packets"]) + r"\%",
            "CapPacedDecided": str(pol.get("decidable_flows_decided")),
            "CapPacedAgree": _pc(pol.get("agreement_on_decidable_flows")) + r"\%",
            "CapPacedSameTen": str(pol.get("decidable_flows_same_ten_packets")),
            "CapPacedSameTenAgree": _pc(pol.get("agreement_same_ten_packets")) + r"\%",
            "CapPacedLateP": f"{paced['open_loop']['replay']['send_lateness_ms']['p99']:.1f}",
            "CapPacedFast": str(pol.get("fast")),
            "CapPacedFastBenign": str(pol.get("fast_benign")),
            "CapPacedAttackStopped": _pc(pcl.get("decided_attack_stopped")) + r"\%",
            "CapPacedBenignStopped": _pc(pcl.get("decided_benign_stopped"), 1) + r"\%",
            "CapPacedDecAttack": str(pcl.get("decided_attack_flows")),
            "CapPacedDecBenign": str(pcl.get("decided_benign_flows")),
            "CapPacedBlocked": str(pcl.get("controller_blocked")),
            "CapPacedEmuBenignStopped": _pc(p.get("emulator_decided_benign_stopped"), 1)
            + r"\%",
            "CapPacedEmuAttackStopped": _pc(p.get("emulator_decided_attack_stopped"))
            + r"\%",
            "CapPacedEmuDecBenign": str(p.get("emulator_decided_benign_flows")),
            "CapPacedDiversityAlerts": str(
                (paced.get("open_loop") or {}).get("diversity_alerts")),
        })
    u = cap["all"]["not_seen_by_compiler"]
    macros.update({
        "CapPackets": _grouped(a["packets"]), "CapFlows": _grouped(a["flows"]),
        "CapDecidable": str(a["flows_reaching_decision"]),
        "CapFast": str(ol.get("fast")), "CapDeferred": str(ol.get("deferred")),
        # BMv2's own fast-path share at recorded timing, where it drops packets.
        "CapFastShare": _pc(ol["fast"] / a["flows"]) + r"\%",
        "CapFastBenign": str(ol.get("fast_benign")),
        "CapAgree": _pc(ol.get("agreement_on_decidable_flows")) + r"\%",
        "CapBlocked": str(cl.get("controller_blocked")),
        "CapAttackStopped": _pc(cl.get("decided_attack_stopped")) + r"\%",
        "CapBenignStopped": _pc(cl.get("decided_benign_stopped"), 1) + r"\%",
        "CapDecAttack": str(cl.get("decided_attack_flows")),
        "CapDecBenign": str(cl.get("decided_benign_flows")),
        "CapDuration": f"{cap['trace']['duration_s']:.0f}",
        "CapNCaptures": str(len(cap["per_capture"])),
        "CapUnseenFlows": str(u["flows"]),
        "CapUnseenDecidable": str(u["flows_reaching_decision"]),
        "CapUnseenFast": str((u.get("open_loop") or {}).get("fast")),
        "CapAmbiguous": str(cap["ambiguous_flows"]),
        "CapOversized": _grouped(cap.get("oversized_packets", 0)),
        "CapDiversityAlerts": str((cap.get("open_loop") or {}).get("diversity_alerts")),
        "CapShareDecidable": _pc(a["flows_reaching_decision"] / a["flows"]) + r"\%",
        "CapShareDecidablePackets": _pc(a["packets_in_decidable_flows"]
                                        / a["packets"]) + r"\%",
        # The emulator keeps up with recorded timing, which BMv2 does not.
        "CapEmuFast": str(a["emulator_fast"]),
        "CapEmuFastShare": _pc(a["emulator_fast"] / a["flows"]) + r"\%",
        "CapEmuFastEligible": _pc(a["emulator_fast"]
                                  / a["flows_reaching_decision"]) + r"\%",
        "CapEmuActedPackets": _pc(a["emulator_fast_packets"]
                                  / a["packets"]) + r"\%",
        "CapMinGap": str(cap.get("min_gap_us", 0)),
        "CapTracePackets": _grouped(cap["trace"]["packets"]),
        "CapProcessed": _grouped(cap["open_loop"]["pipeline_packets"]),
        "CapProcessedShare": _pc(cap["open_loop"]["pipeline_packets"]
                                 / cap["trace"]["packets"]) + r"\%",
        "CapDecidedBySwitch": str(ol.get("decidable_flows_decided")),
        "CapSameTen": str(ol.get("decidable_flows_same_ten_packets")),
        "CapSameTenAgree": _pc(ol.get("agreement_same_ten_packets")) + r"\%",
        "CapLateP": f"{cap['open_loop']['replay']['send_lateness_ms']['p99']:.1f}",
        "CapEmuDecBenign": str(a.get("emulator_decided_benign_flows")),
        "CapEmuBenignStopped": _pc(a.get("emulator_decided_benign_stopped"), 1)
        + r"\%",
        "CapEmuAttackStopped": _pc(a.get("emulator_decided_attack_stopped"))
        + r"\%",
        "CapFastBenignRemote": {1: "one", 2: "two", 3: "three"}.get(
            len(a.get("emulator_fast_benign_remote", [])),
            str(len(a.get("emulator_fast_benign_remote", [])))),
        "CapEmuFastBenign": str(a.get("emulator_fast_benign")),
    })


def table_two_tier(tt, out: Path, data: Path, macros: dict):
    """Controller alone against switch plus controller, one benign budget."""
    from scipy.stats import spearmanr
    s = tt["summary"]
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"& & \multicolumn{3}{c}{Matched benign rate} & "
             r"\multicolumn{3}{c}{Validation-selected} \\",
             r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
             r"Rule & Budget & Alone & Split (offload) & Gain (95\% CI) & "
             r"Alone (benign) & Split (benign) & Gain (95\% CI) \\", r"\midrule"]
    for split in ("flow-isolated", "context-disjoint"):
        for i, b in enumerate(("0.01", "0.02", "0.05")):
            e = s[split]["report12"][b]
            m, j = e["matched_best_minus_alone"], e["joint_blk_minus_alone_blk"]
            lines.append(
                f"{SPLIT_NAME[split] if i == 0 else ''} & {100 * float(b):.0f}\\% & "
                f"{_pc(e['matched_alone_recall']['mean'])} & "
                f"{_pc(e['matched_best_recall']['mean'])} "
                f"({_pc(e['matched_best_offload']['mean'], 0)}) & "
                f"{_num(m['mean_points'], sign=True)} {_cis(m)} & "
                f"{_pc(e['alone_blk_recall']['mean'])} "
                f"({_pc(e['alone_blk_benign']['mean'], 2)}) & "
                f"{_pc(e['joint_blk_recall']['mean'])} "
                f"({_pc(e['joint_blk_benign']['mean'], 2)}) & "
                f"{_num(j['mean_points'], sign=True)} {_cis(j)}" + r" \\")
        if split == "flow-isolated":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    rng = np.random.RandomState(0)

    def ci(v):
        v = np.asarray(v, dtype=float)
        boot = v[rng.randint(0, len(v), (2000, len(v)))].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return v.mean(), lo, hi
    for split, st in (("flow-isolated", "fi"), ("context-disjoint", "sig")):
        per = [r for r in tt["per_seed"] if r["split"] == split]
        rows = ["offload r1 lo1 hi1 n1 r2 lo2 hi2 n2"]
        for o in (0.0, 0.1, 0.2, 0.3, 0.4):
            cells = []
            for b in ("0.01", "0.02"):
                if o == 0.0:
                    v = [r["controllers"]["report12"][b]["alone"]["recall_matched"]
                         for r in per]
                else:
                    v = [r["controllers"]["report12"][b]["matched_at_offload"][str(o)]
                         for r in per]
                    v = [x for x in v if x is not None]
                m, lo, hi = ci(100 * np.array(v))
                cells += [f"{m:.2f}", f"{lo:.2f}", f"{hi:.2f}", str(len(v))]
            rows.append(f"{100 * o:g} " + " ".join(cells))
        (data.parent /
         f"{data.stem}_{st}.dat").write_text("\n".join(rows) + "\n")
    for split, st in (("flow-isolated", ""), ("context-disjoint", "Sig")):
        for b, t in (("0.01", "One"), ("0.02", "Two"), ("0.05", "Five")):
            for c, ct in (("report5", "Five"), ("report12", "Twelve")):
                e = s[split][c][b]
                p = f"Tt{st}R{ct}{t}"
                m, j = e["matched_best_minus_alone"], e["joint_blk_minus_alone_blk"]
                macros.update({
                    f"{p}Alone": _pc(e["matched_alone_recall"]["mean"]) + r"\%",
                    f"{p}Split": _pc(e["matched_best_recall"]["mean"]) + r"\%",
                    f"{p}Off": _pc(e["matched_best_offload"]["mean"]) + r"\%",
                    f"{p}Gain": _num(m["mean_points"]), f"{p}GainCI": _cis(m),
                    f"{p}Wins": str(m["wins"]),
                    f"{p}DepAlone": _pc(e["alone_blk_recall"]["mean"]) + r"\%",
                    f"{p}DepSplit": _pc(e["joint_blk_recall"]["mean"]) + r"\%",
                    f"{p}DepAloneBen": _pc(e["alone_blk_benign"]["mean"], 2) + r"\%",
                    f"{p}DepSplitBen": _pc(e["joint_blk_benign"]["mean"], 2) + r"\%",
                    f"{p}DepOff": _pc(e["joint_blk_offload"]["mean"]) + r"\%",
                    f"{p}DepGain": _num(j["mean_points"]), f"{p}DepGainCI": _cis(j),
                    f"{p}DepWins": str(j["wins"]),
                    f"{p}DepAloneOver": str(e["alone_blk_over_budget"]),
                    f"{p}DepSplitOver": str(e["joint_blk_over_budget"]),
                })
                if "alone_blk_connections_over_budget" in e:
                    macros.update({
                        f"{p}DepAloneConn": _pc(
                            e["alone_blk_benign_connections"]["mean"], 2) + r"\%",
                        f"{p}DepSplitConn": _pc(
                            e["joint_blk_benign_connections"]["mean"], 2) + r"\%",
                        f"{p}DepAloneConnOver": str(
                            e["alone_blk_connections_over_budget"]),
                        f"{p}DepSplitConnOver": str(
                            e["joint_blk_connections_over_budget"])})
            r = s[split]["report12_minus_report5"][b]["alone"]
            macros[f"Tt{st}ReportGain{t}"] = _num(r["mean_points"])
            macros[f"Tt{st}ReportGainCI{t}"] = _cis(r)
            macros[f"Tt{st}ReportWins{t}"] = str(r["wins"])
            # At validation-selected thresholds the arms land at different
            # benign rates: how far the recall difference follows that.
            arms = [r["controllers"]["report12"][b] for r in tt["per_seed"]
                    if r["split"] == split]
            rho = spearmanr(
                [a["joint_blk"]["recall"] - a["alone_blk"]["recall"]
                 for a in arms],
                [a["joint_blk"]["benign"] - a["alone_blk"]["benign"]
                 for a in arms])[0]
            macros[f"Tt{st}RTwelve{t}DepRho"] = _num(rho, 2)
    # A few long benign connections, each split into many flow records,
    # decide most of the controller's excess benign rate.
    fi = [r for r in tt["per_seed" if "per_seed" in tt else "rows"]
          if r["split"] == "flow-isolated" and "largest_benign_5tuple_test" in r]
    if fi:
        big = [r["largest_benign_5tuple_test"] for r in fi]
        macros["TtLargestBenignMin"] = str(min(big))
        macros["TtLargestBenignMax"] = str(max(big))
        macros["TtLargestBenignHundred"] = str(sum(b >= 100 for b in big))


def data_map_study(cic, unsw, figures: Path, macros: dict):
    """Gain against crossing mass over random action maps, both corpora."""
    for tag, T, d in (("cic", "Cic", cic), ("unsw", "Unsw", unsw)):
        if not d:
            continue
        s = d["summary"]
        n_attack = max(max(m["map"]) for m in s["maps"])
        for k, kt in ((1, "one"), (2, "two"), (3, "three"), (n_attack, "sep")):
            rows = ["cross gain lo hi"]
            for m in s["maps"]:
                if m["actions"] == k:
                    rows.append(f"{100 * m['crossing_mass']:.2f} "
                                f"{m['gain_points']:.2f} {m['ci95'][0]:.2f} "
                                f"{m['ci95'][1]:.2f}")
            (figures /
             f"mapstudy_{tag}_{kt}.dat").write_text("\n".join(rows) + "\n")
        multi = [m for m in s["maps"] if m["actions"] in (2, 3)]
        n_seeds = len(d["config"]["seeds"])
        macros.update({
            f"MapStudy{T}Maps": str(s["n_maps"]),
            f"MapStudy{T}Multi": str(len(multi)),
            "MapStudyRandom": str(d["config"]["random_maps_per_size"]),
            f"MapStudy{T}Seeds": str(n_seeds),
            f"MapStudy{T}Rho": f"{s['crossing_rho']:.2f}",
            f"MapStudy{T}P": _p(s["crossing_p"]),
            f"MapStudy{T}RhoMulti": f"{s['crossing_rho_multi_action']:.2f}",
            f"MapStudy{T}AggRho": f"{s['spearman_rho']:.2f}",
            f"MapStudy{T}Positive": str(s["positive_gain_maps"]),
            f"MapStudy{T}CIPositive": str(s["ci_above_zero_maps"]),
            f"MapStudy{T}MultiGainMin": _num(min(m["gain_points"] for m in multi)),
            f"MapStudy{T}MultiGainMax": _num(max(m["gain_points"] for m in multi)),
            f"MapStudy{T}MultiCrossMedian": _pc(float(np.median(
                [m["crossing_mass"] for m in multi]))) + r"\%",
            f"MapStudy{T}MultiCrossMax": _pc(max(
                m["crossing_mass"] for m in multi)) + r"\%",
            f"MapStudy{T}MultiFrontMax": _pc(max(
                m["action_front"] for m in multi)) + r"\%",
            f"MapStudy{T}MultiMissRate": _pc(sum(
                m["action_misses"] for m in multi) / (len(multi) * n_seeds)) + r"\%",
        })
        map_selection_facts(d, T, macros)


# Display names of the fine labels in the action maps of the map study.
MAP_CLASS_NAME = {"Recon": "Reconnaissance", "BruteForce": "Brute Force"}


def _map_description(amap):
    """The attack classes of a map grouped by action, in map order; class
    indices stand in for corpora whose label names are not loaded."""
    groups = {}
    for c, a in enumerate(amap):
        if c:
            name = CLASSES[c] if c < len(CLASSES) else str(c)
            groups.setdefault(a, []).append(MAP_CLASS_NAME.get(name, name))
    return "; ".join(", ".join(v) for v in groups.values())


def map_selection_facts(d, T, macros: dict, n_boot=2000):
    """
    The map-selection rule tested on the map study: among a candidate set
    fixed in advance, deploy the map with the largest crossing mass on the
    training cells. Its held-out gain is compared with the mean candidate
    (a random choice) and with the candidate that is best on test outcomes
    (an oracle). Candidate sets: every multi-action map, and the maps with
    exactly two or three attack actions.
    """
    from scipy.stats import spearmanr
    per = d["per_seed"]
    if not per or "crossing_mass" not in per[0]["maps"][0]:
        return
    eps = d["config"]["target"]
    maps = [tuple(m["map"]) for m in per[0]["maps"]]
    n_attack = np.array([len(set(m[1:])) for m in maps])
    gain = 100 * np.array([[m["action_cov"] - m["label_cov"]
                            for m in r["maps"]] for r in per])
    cross = np.array([[m["crossing_mass"] for m in r["maps"]] for r in per])
    acov = 100 * np.array([[m["action_cov"] for m in r["maps"]] for r in per])
    aerr = np.array([[m["action_err"] if m["action_err"] is not None else 0.0
                      for m in r["maps"]] for r in per])
    rng = np.random.RandomState(0)

    def ci(x):
        boot = x[rng.randint(0, len(x), (n_boot, len(x)))].mean(axis=1)
        return {"ci95": np.percentile(boot, [2.5, 97.5])}
    n_seeds = len(per)
    for name, idx in (("Multi", np.flatnonzero(n_attack >= 2)),
                      ("Two", np.flatnonzero(n_attack == 2)),
                      ("Three", np.flatnonzero(n_attack == 3)),
                      ("All", np.arange(len(maps)))):
        if len(idx) < 2:
            continue
        pick = np.array([idx[np.argmax(cross[s, idx])] for s in range(n_seeds)])
        g_pick = gain[np.arange(n_seeds), pick]
        g_rand = gain[:, idx].mean(axis=1)
        g_best = gain[:, idx].max(axis=1)
        c_pick = acov[np.arange(n_seeds), pick]
        e_pick = aerr[np.arange(n_seeds), pick]
        rank = np.array([(gain[s, idx] > g_pick[s]).sum() + 1
                         for s in range(n_seeds)])
        counts = {}
        for i in pick:
            counts[i] = counts.get(i, 0) + 1
        top, top_n = max(counts.items(), key=lambda kv: kv[1])
        P = f"MapSel{T}{name}"
        macros.update({
            f"{P}Candidates": str(len(idx)),
            f"{P}Gain": _num(g_pick.mean()),
            f"{P}GainCI": _cis(ci(g_pick)),
            f"{P}GainMin": _num(g_pick.min()),
            f"{P}Cov": _pc(c_pick.mean() / 100) + r"\%",
            f"{P}Err": _pc(e_pick.mean() / 100, 2) + r"\%",
            f"{P}Misses": str(int((e_pick > eps).sum())),
            f"{P}Random": _num(g_rand.mean()),
            f"{P}RandomCI": _cis(ci(g_rand)),
            f"{P}OverRandom": _num((g_pick - g_rand).mean()),
            f"{P}OverRandomCI": _cis(ci(g_pick - g_rand)),
            f"{P}Oracle": _num(g_best.mean()),
            f"{P}OracleShare": _pc(float(g_pick.sum() / g_best.sum()), 0)
            + r"\%",
            f"{P}RankMedian": str(int(np.median(rank))),
            f"{P}TopQuartile": str(int((rank <= len(idx) / 4).sum())),
            f"{P}BeatsMedian": str(int((g_pick > np.median(
                gain[:, idx], axis=1)).sum())),
            f"{P}TopMap": _map_description(maps[top]),
            f"{P}TopMapSeeds": str(top_n),
            f"{P}TopMapActions": str(int(n_attack[top])),
            f"{P}Distinct": str(len(counts)),
        })
        rho = [spearmanr(cross[s, idx], gain[s, idx]).correlation
               for s in range(n_seeds)]
        macros[f"{P}Rho"] = f"{np.nanmean(rho):.2f}"
    macros[f"MapSel{T}Seeds"] = str(n_seeds)


ACTION_LABEL = {"1": ("Drop", "1"), "2": ("Rate limit", "5"),
                "3": ("Redirect", "10")}


def table_per_action(study, grouped, out: Path, macros: dict, conf=None,
                     conf_grouped=None):
    """
    What the compiler certified under tiered budgets, per map and action, on
    the flow-isolated partition: seeds in which the action received any
    entry, and the mean entries, coverage, selective error and share of
    benign test flows of the certified entries (means over those seeds).
    """
    sources = [(study, "3-action", "CICIoT2023, 3-action", "Three"),
               (study, "4-action", "CICIoT2023, 4-action", "Four")]
    if grouped:
        sources += [(grouped, "flood-inspect", "CICIoT2023, flood-inspection",
                     "FloodInsp"),
                    (grouped, "flood-ratelimit", "CICIoT2023, flood-rate-limit",
                     "FloodRate")]
    confs = {"3-action": conf, "4-action": conf,
             "flood-inspect": conf_grouped, "flood-ratelimit": conf_grouped}
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Policy & Action (budget) & Certified seeds & Entries & Coverage & "
             r"Error & Benign share \\", r"\midrule"]
    for src, m, name, mt in sources:
        t = src["summary"]["flow-isolated"][m][f"tiered_{CAL}"]["action"]
        n_seeds = len(src["config"]["seeds"])
        tc = confs.get(m)
        tc = (tc["summary"]["flow-isolated"][m][f"tiered_{CAL}"]["action"]
              if tc else None)
        first = True
        for a, (an, budget) in ACTION_LABEL.items():
            cert = t["certified_seeds_by_action"].get(a)
            if cert is None:
                continue
            seeds = f"{cert}/{n_seeds}"
            if tc is not None:
                seeds += f" ({tc['certified_seeds_by_action'].get(a)})"
            if cert:
                cells = (f"{t['entries_mean_by_action'][a]:.1f} & "
                         f"{_pc(t['coverage_mean_by_action'][a])} & "
                         f"{_pc(t['error_mean_by_action'][a], 2)} & "
                         f"{_pc(t['benign_rate_mean_by_action'][a], 2)}")
            else:
                cells = " & ".join([NA] * 4)
            lines.append(f"{name if first else ''} & {an} ({budget}\\%) & "
                         f"{seeds} & {cells}" + r" \\")
            first = False
            at = {"1": "Drop", "2": "Rate", "3": "Redir"}[a]
            macros[f"PerAct{mt}{at}Seeds"] = str(cert)
            if cert:
                macros.update({
                    f"PerAct{mt}{at}Entries": f"{t['entries_mean_by_action'][a]:.1f}",
                    f"PerAct{mt}{at}Cov": _pc(t["coverage_mean_by_action"][a])
                    + r"\%",
                    f"PerAct{mt}{at}Err": _pc(t["error_mean_by_action"][a], 2)
                    + r"\%",
                    f"PerAct{mt}{at}Benign": _pc(
                        t["benign_rate_mean_by_action"][a], 2) + r"\%"})
        macros[f"PerAct{mt}Total"] = _pc(t["coverage_mean"]) + r"\%"
        # The binary gate under the same tiered budgets, for the case in
        # which only one action is certified.
        b = src["summary"]["flow-isolated"][m][f"tiered_{CAL}"].get("binary")
        if b and b["certified_seeds_by_action"].get("1"):
            macros.update({
                f"PerAct{mt}BinDropSeeds": str(b["certified_seeds_by_action"]["1"]),
                f"PerAct{mt}BinDropEntries": f"{b['entries_mean_by_action']['1']:.1f}",
                f"PerAct{mt}BinDropCov": _pc(b["coverage_mean_by_action"]["1"])
                + r"\%",
                f"PerAct{mt}BinTotal": _pc(b["coverage_mean"]) + r"\%"})
        lines.append(r"\midrule")
    lines.pop()
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def direct_action_facts(d, macros: dict):
    """
    Pooling a fine-label model by action against fitting the same family
    directly on the action labels, under tiered budgets. For the cell table
    the two are the same computation.
    """
    s = d["summary"]
    tag = {"3-action": "Three", "4-action": "Four", "flood-inspect": "FloodInsp"}
    gbdt, tree = [], []
    for m, mt in tag.items():
        if m not in s:
            continue
        b = s[m]
        macros[f"Direct{mt}TableDiff"] = f"{b['table_max_abs_score_diff']:.0e}"
        for fam, ft, acc in (("gbdt", "Gbdt", gbdt), ("tree", "Tree", tree)):
            c = b[f"{fam}_pooled_minus_direct_tiered"]
            acc.append(c)
            macros.update({
                f"Direct{mt}{ft}Diff": _num(c["mean_points"]),
                f"Direct{mt}{ft}DiffCI": _cis(c),
                f"Direct{mt}{ft}Adv": _num(-c["mean_points"]),
                f"Direct{mt}{ft}AdvCI": _cis(
                    {"ci95": [-c["ci95"][1], -c["ci95"][0]]}),
                f"Direct{mt}{ft}Wins": f"{c['wins']}/{c['n']}",
                f"Direct{mt}{ft}Pooled": _pc(
                    b[f"{fam}_pooled"]["tiered_coverage_mean"]) + r"\%",
                f"Direct{mt}{ft}Direct": _pc(
                    b[f"{fam}_direct"]["tiered_coverage_mean"]) + r"\%"})
    for ft, acc in (("Gbdt", gbdt), ("Tree", tree)):
        if acc:
            macros[f"Direct{ft}AbsMax"] = _num(max(
                abs(c["mean_points"]) for c in acc))
            macros[f"Direct{ft}ZeroIntervals"] = str(sum(
                c["ci95"][0] <= 0 <= c["ci95"][1] for c in acc))
            macros[f"Direct{ft}Maps"] = str(len(acc))
    per = d["per_seed"]
    fine = np.mean([r["fit_seconds"]["gbdt_fine"] for r in per])
    direct = np.mean([r["maps"][m]["fit_seconds_gbdt_direct"]
                      for r in per for m in r["maps"]])
    repool = np.mean([r["maps"][m]["compile_seconds_pooled_table"]
                      for r in per for m in r["maps"]])
    macros.update({"DirectFitFine": f"{fine:.1f}",
                   "DirectFitDirect": f"{direct:.1f}",
                   "DirectRepoolMs": f"{1000 * repool:.0f}",
                   "DirectSeeds": str(len(per))})


def controller_sensitivity_facts(c, out: Path, macros: dict):
    """The adjudicator over tree counts and leaf limits: recall at a matched
    benign block rate against the deployed configuration, size and time."""
    s, ref = c["summary"], c["config"]["reference"]
    cfgs = sorted(s, key=lambda k: tuple(int(v) for v in k.split("x")))
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"Trees & Leaves & Recall at 1\% & Difference (95\% CI) & "
             r"Recall at 5\% & Model (kB) & Time per report ($\mu$s) \\",
             r"\midrule"]
    for k in cfgs:
        e, d = s[k], s[k]["matched_recall_minus_ref_0.01"]
        trees, leaves = k.split("x")
        diff = ("reference" if k == ref else
                f"{_num(d['mean_points'], sign=True)} {_cis(d)}")
        lines.append(f"{trees} & {leaves} & {_pc(e['matched_recall_mean_0.01'])}"
                     f" & {diff} & {_pc(e['matched_recall_mean_0.05'])} & "
                     f"{e['model_bytes'] / 1000:.0f} & "
                     f"{e['single_us_per_report']:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")
    r1 = {k: s[k]["matched_recall_mean_0.01"] for k in cfgs}
    r5 = {k: s[k]["matched_recall_mean_0.05"] for k in cfgs}
    others = [k for k in cfgs if k != ref]
    diffs = [s[k]["matched_recall_minus_ref_0.01"] for k in others]
    rt, rl = ref.split("x")
    macros.update({
        "CtlSensConfigs": str(len(cfgs)),
        "CtlSensOthers": str(len(others)),
        "CtlSensSeeds": str(len(c["config"]["seeds"])),
        "CtlSensTrees": ", ".join(str(t) for t in c["config"]["grid"]["trees"]),
        "CtlSensLeaves": ", ".join(str(t) for t in c["config"]["grid"]["leaves"]),
        "CtlSensRefTrees": rt, "CtlSensRefLeaves": rl,
        "CtlSensRefRecallOne": _pc(r1[ref]) + r"\%",
        "CtlSensMinRecallOne": _pc(min(r1.values())) + r"\%",
        "CtlSensMaxRecallOne": _pc(max(r1.values())) + r"\%",
        "CtlSensMinRecallFive": _pc(min(r5.values())) + r"\%",
        "CtlSensMaxRecallFive": _pc(max(r5.values())) + r"\%",
        "CtlSensZeroIntervals": str(sum(
            d["ci95"][0] <= 0 <= d["ci95"][1] for d in diffs)),
        "CtlSensAbsMaxDiff": _num(max(abs(d["mean_points"]) for d in diffs)),
        "CtlSensRefBytes": f"{s[ref]['model_bytes'] / 1000:.0f}",
        "CtlSensRefLeavesTotal": _grouped(int(s[ref]["leaves_total"])),
        "CtlSensRefSingleUs": f"{s[ref]['single_us_per_report']:.0f}",
        "CtlSensMinSingleUs": f"{min(s[k]['single_us_per_report'] for k in cfgs):.0f}",
        "CtlSensMaxSingleUs": f"{max(s[k]['single_us_per_report'] for k in cfgs):.0f}",
        "CtlSensRefBatchUs": f"{s[ref]['batch_us_per_report']:.1f}",
        "CtlSensPythonUs": f"{s[ref]['python_scorer_us_per_report']:.0f}",
        "CtlSensSmallBytes": f"{s['50x31']['model_bytes'] / 1000:.0f}",
        "CtlSensSmallRecallOne": _pc(r1["50x31"]) + r"\%",
        "CtlSensSmallDiff": _num(s["50x31"]["matched_recall_minus_ref_0.01"]
                                 ["mean_points"], sign=True),
        "CtlSensSmallDiffCI": _cis(s["50x31"]["matched_recall_minus_ref_0.01"]),
        "CtlSensLargestBytes": f"{s[cfgs[-1]]['model_bytes'] / 1000:.0f}",
    })


def dev_calibration_facts(dev, macros: dict):
    """
    The candidate calibration rules on the development seeds: settings with
    more than three misses, total misses and mean coverage, for the earlier
    block rule and the chosen dependence rule, over all three partitions.
    """
    rows = [r for r in dev["per_seed"] if "candidates" in r]
    if not rows:
        return
    stats = {}
    for r in rows:
        for m, gates in r["candidates"].items():
            for g, by_eps in gates.items():
                for eps, res in by_eps.items():
                    for cand, (cov, _, met) in res.items():
                        key = (r["split"], m, g, eps, cand)
                        s = stats.setdefault(key, [0.0, 0, 0])
                        s[0] += cov
                        s[1] += 0 if met else 1
                        s[2] += 1
    for cand, tag in (("blk|target", "Old"), ("all|common", "Dep")):
        cells = [v for k, v in stats.items() if k[4] == cand]
        macros[f"Dev{tag}Over"] = str(sum(v[1] > 3 for v in cells))
        macros[f"Dev{tag}Misses"] = str(sum(v[1] for v in cells))
        macros[f"Dev{tag}Cov"] = _pc(
            np.mean([v[0] / v[2] for v in cells])) + r"\%"
        for split, st in (("flow-isolated", "Fi"), ("context-disjoint", "Sig"),
                          ("blocked", "Blk")):
            sub = [v for k, v in stats.items() if k[4] ==
                   cand and k[0] == split]
            macros[f"Dev{tag}{st}Over"] = str(sum(v[1] > 3 for v in sub))
            macros[f"Dev{tag}{st}Misses"] = str(sum(v[1] for v in sub))
    macros["DevCandidates"] = str(len({k[4] for k in stats}))


def partition_facts(study, macros: dict, dev=None, grid=None):
    """
    Where the largest context-disjoint group falls and what it does to the
    target misses, and the capture scenarios the blocked rule tests unseen.
    """
    rows = [r for r in study["per_seed"] if r["split"] == "context-disjoint"
            and "largest_unit" in r]
    if rows:
        where = [r["largest_unit"]["partition"] for r in rows]
        size = [r["largest_unit"]["flows"] for r in rows]
        ben = [r["largest_unit"]["classes"][0] for r in rows]
        macros.update({
            "GiantMin": _grouped(min(size)), "GiantMax": _grouped(max(size)),
            "GiantClasses": str(min(sum(c > 0 for c in r["largest_unit"]["classes"])
                                    for r in rows)),
            "GiantBenignMin": _grouped(min(ben)), "GiantBenignMax": _grouped(max(ben)),
            "GiantTest": str(where.count("test")),
            "GiantVal": str(where.count("validation")),
            "GiantTrain": str(where.count("train")),
        })
        # Misses of the dependence rule over all maps, gates and targets,
        # split by where the component fell.
        cells = [(m, g, e) for m in rows[0]["gates"] for g in ("label", "action")
                 for e in TARGETS]

        def misses(sub):
            return sum(not r["gates"][m][g][e][f"{CAL}_met"]
                       for r in sub for m, g, e in cells)
        test = [r for r in rows if r["largest_unit"]["partition"] == "test"]
        rest = [r for r in rows if r["largest_unit"]["partition"] != "test"]
        val = [r for r in rows if r["largest_unit"]
               ["partition"] == "validation"]
        macros.update({
            "GiantCells": str(len(cells)),
            "GiantTestMisses": str(misses(test)),
            "GiantTestEvals": _grouped(len(test) * len(cells)),
            "GiantRestMisses": str(misses(rest)),
            "GiantRestSeeds": str(len(rest)),
            "GiantRestEvals": _grouped(len(rest) * len(cells)),
            "GiantValMisses": str(misses(val)),
            "GiantTestSize": _grouped(int(np.mean([r["n_test"] for r in test])))
            if test else NA,
            "GiantRestSize": _grouped(int(np.mean([r["n_test"] for r in rest]))),
        })
        # Share of the subsample, which every seed draws at the same size.
        total = corpus_flows(Path("results/controlled"))
        if total:
            share = [s / total for s in size]
            macros["GiantShareMin"] = _pc(min(share), 0) + r"\%"
            macros["GiantShareMax"] = _pc(max(share), 0) + r"\%"
    blk = [r for r in study["per_seed"] if r["split"] == "blocked"
           and "unseen_scenarios" in r]
    if blk:
        u = [r["unseen_scenarios"] for r in blk]
        names = sorted({n for x in u for n in x["test_scenarios"]})
        macros.update({
            "BlkUnseenAttackMin": _pc(min(x["test_attack_share"] for x in u)) + r"\%",
            "BlkUnseenAttackMax": _pc(max(x["test_attack_share"] for x in u)) + r"\%",
            "BlkUnseenAttack": _pc(float(np.mean(
                [x["test_attack_share"] for x in u]))) + r"\%",
            "BlkUnseenScenarios": str(len(names)),
            "BlkUnseenValidation": _pc(float(np.mean(
                [x["validation_share"] for x in u]))) + r"\%",
        })
        fi = [r for r in study["per_seed"] if r["split"] == "flow-isolated"
              and "unseen_scenarios" in r]
        macros["FiUnseenAttack"] = _pc(float(np.mean(
            [r["unseen_scenarios"]["test_attack_share"] for r in fi]))) + r"\%"
    if dev:
        dw = [r["largest_unit"]["partition"] for r in dev["per_seed"]
              if r["split"] == "context-disjoint" and "largest_unit" in r]
        macros["GiantTestDev"] = str(dw.count("test"))
    if grid:
        for tag, v in grid.items():
            macros[tag] = v
    # The longest benign connection under flow-isolated partitioning, and the
    # seed in which the action gate admits nothing at the 1% target.
    fi = [r for r in study["per_seed"] if r["split"] == "flow-isolated"
          and "largest_benign_connection" in r]
    if fi:
        lb = [r["largest_benign_connection"] for r in fi]
        macros["LongBenignRecords"] = str(lb[0]["corpus_records"])
        macros["LongBenignTestSeeds"] = str(sum(x["test"] > 0 for x in lb))
        macros["LongBenignValSeeds"] = str(
            sum(x["validation"] > 0 for x in lb))
        zero = [r for r in fi
                if r["gates"]["2-action"]["action"]["0.01"][f"{CAL}_cov"] == 0.0]
        if zero:
            macros["LongBenignValRecords"] = str(
                zero[0]["largest_benign_connection"]["validation"])
            macros["ZeroCoverageSeed"] = str(zero[0]["seed"])
    blk = [r for r in study["per_seed"] if r["split"] == "blocked"]
    miss = [r for r in blk if not r["gates"]["2-action"]["action"]["0.005"]
            [f"{CAL}_met"] and f"{CAL}_n_val" in r["gates"]["2-action"]["action"]
            ["0.005"]]
    if miss:
        e = miss[0]["gates"]["2-action"]["action"]["0.005"]
        macros.update({"BlkMissSeed": str(miss[0]["seed"]),
                       "BlkMissValAdmitted": _grouped(e[f"{CAL}_n_val"]),
                       "BlkMissTestAdmitted": _grouped(e[f"{CAL}_n_te"]),
                       "BlkMissValBenign": str(e[f"{CAL}_k_val"]),
                       "BlkMissTestBenign": str(e[f"{CAL}_k_te"])})


def confirmation_facts(macros: dict, root=Path("results/confirm")):
    """
    The paper's numbers on seeds 60 to 89, run once after every choice was
    fixed, under the same macro names with a Conf prefix.
    """
    import shutil
    import tempfile
    study = _read(root / "policy_study_seeds60-89.json")
    if not study:
        return
    tmp, scratch = {}, Path(tempfile.mkdtemp())
    try:
        table_policy_main(study, scratch / "p.tex", tmp)
        partition_facts(study, tmp)
        grouped = _read(root / "policy_study_grouped_seeds60-89.json")
        table_policy_generality(study, _read(root / "unsw_policy_seeds60-89.json"),
                                scratch / "g.tex", tmp, grouped)
        table_per_action(study, grouped, scratch / "a.tex", tmp)
        direct = _read(root / "direct_action_study_seeds60-89.json")
        if direct:
            direct_action_facts(direct, tmp)
        bl = _read(root / "baselines_study_seeds60-89.json")
        if bl:
            table_policy_baselines(bl, scratch / "b.tex", tmp)
        tt = _read(root / "two_tier_study_seeds60-89.json")
        if tt:
            table_two_tier(tt, scratch / "t.tex", scratch / "t.dat", tmp)
        cic = _read(root / "map_study_seeds60-89.json")
        if cic:
            data_map_study(cic, _read(root / "unsw_map_study_seeds60-89.json"),
                           scratch, tmp)
    finally:
        shutil.rmtree(scratch)
    macros.update({"Conf" + k: v for k, v in tmp.items()})


def benign_capture_facts(cap, macros: dict):
    """Unseen benign captures replayed through the closed loop."""
    from scipy.stats import beta
    a = cap["all"]["all"]
    ol, cl = a.get("open_loop", {}), a.get("closed_loop", {})
    n, rate = cl.get("decided_benign_flows"), cl.get("decided_benign_stopped")
    k = int(round(rate * n)) if n else 0
    lo = beta.ppf(0.025, k, n - k + 1) if k else 0.0
    hi = beta.ppf(0.975, k + 1, n - k) if n else None
    new = cap["all"]["not_in_corpus"].get("closed_loop", {})
    macros.update({
        "CapBenPackets": _grouped(cap["trace"]["packets"]),
        "CapBenDuration": f"{cap['trace']['duration_s']:.0f}",
        "CapBenFlows": _grouped(a["flows"]),
        "CapBenDecidable": str(a["flows_reaching_decision"]),
        "CapBenProcessedShare": _pc(cap["open_loop"]["pipeline_packets"]
                                    / cap["trace"]["packets"]) + r"\%",
        "CapBenAgree": _pc(ol.get("agreement_on_decidable_flows")) + r"\%",
        "CapBenDecided": str(n),
        "CapBenStoppedCount": str(k),
        "CapBenStopped": _pc(rate, 1) + r"\%",
        "CapBenStoppedCI": f"[{_pc(lo, 1)}, {_pc(hi, 1)}]",
        "CapBenFastBenign": str(ol.get("fast_benign")),
        "CapBenBlocked": str(cl.get("controller_blocked")),
        "CapBenEmuStopped": _pc(a.get("emulator_decided_benign_stopped"), 1)
        + r"\%",
        "CapBenNewDecided": str(new.get("decided_benign_flows")),
        "CapBenNewStopped": _pc(new.get("decided_benign_stopped"), 1) + r"\%",
        "CapBenRemote": str(len(a.get("emulator_fast_benign_remote", []))),
    })


def grid_paired_facts(macros: dict, root=Path("results/controlled")):
    """
    Seed-paired contrasts of the controlled grid for the sequence model and
    the reference ensemble: mean difference in macro F1 points, 95%
    bootstrap interval over seeds, wins and a two-sided signed-rank test.
    A seed fixes the partition, so the two runs of a pair share test flows.
    """
    from scipy.stats import wilcoxon
    groups = load_grid(root)
    seq = {k: {r["protocol"]["seed"]: r["test"]["macro_f1"] for r in v}
           for k, v in groups.items()}
    bl = _read(root / "baselines.json") or {}
    lgb = {}
    for label, e in bl.get("models", {}).get(REFERENCE_BASELINE, {}).items():
        ctx, _, split = label.partition("@")
        lgb[(ctx, True, split or "flow-isolated")] = {
            r["seed"]: r["macro_f1"] for r in e.get("runs", []) if "seed" in r}
    fi = "flow-isolated"
    contrasts = {
        "CtxAug": (("real", True, fi), ("none", True, fi)),
        "CtxNoAug": (("real", False, fi), ("none", False, fi)),
        "AugNoCtx": (("none", True, fi), ("none", False, fi)),
        "AugCtx": (("real", True, fi), ("real", False, fi)),
        "PerFlow": (("real", True, fi), ("class-marginal", True, fi)),
        "ClassPrior": (("class-marginal", True, fi), ("none", True, fi)),
        "Shuffle": (("shuffle", True, fi), ("none", True, fi)),
        "RowNoCtx": (("none", True, "random"), ("none", True, fi)),
        "RowCtx": (("real", True, "random"), ("real", True, fi)),
        "DisjNoCtx": (("none", True, "context-disjoint"), ("none", True, fi)),
        "DisjCtx": (("real", True, "context-disjoint"), ("real", True, fi)),
    }
    rng = np.random.RandomState(0)

    def paired(prefix, runs, pairs):
        sizes = []
        for name, (a, b) in pairs.items():
            common = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
            if len(common) < 3:
                continue
            d = 100 * np.array([runs[a][s] - runs[b][s] for s in common])
            boot = d[rng.randint(0, len(d), (2000, len(d)))].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
            sizes.append(len(common))
            macros.update({
                f"{prefix}{name}": _num(d.mean()),
                f"{prefix}{name}CI": f"[{_num(lo)}, {_num(hi)}]",
                f"{prefix}{name}Wins": str(int((d > 0).sum())),
                f"{prefix}{name}N": str(len(common)),
                f"{prefix}{name}P": _p(p)})
        if sizes:
            macros[f"{prefix}Seeds"] = str(min(sizes))

    for est, runs in (("Seq", seq), ("Lgb", lgb)):
        paired(f"Grid{est}", runs, contrasts)
    # The same controls on UNSW-NB15, for the estimator the paper reports.
    models = (_read("results/unsw_nb15/decomposition.json") or {}).get(
        "models", {})
    best = max((m for m in models if "decomposition" in models[m]),
               key=lambda m: models[m]["real"]["macro_f1_mean"], default=None)
    if best and "by_seed" in models[best]["real"]:
        runs = {c: {int(s): v for s, v in models[best][c]["by_seed"].items()}
                for c in ("none", "real", "class-marginal", "shuffle")}
        paired("GridUnsw", runs, {
            "CtxTotal": ("real", "none"), "PerFlow": ("real", "class-marginal"),
            "ClassPrior": ("class-marginal", "none"),
            "Shuffle": ("shuffle", "none")})


# IoT-23 honeypot captures by device, in the order the text names them.
IOT23_DEVICES = (("iot23_echo", "Echo"), ("iot23_hue", "Hue"),
                 ("iot23_somfy", "Somfy"))


def benign_iot23_facts(d, macros: dict):
    """
    Benign flows from another network (IoT-23 honeypot captures) through
    the tables compiled on each reporting seed: drops under the two-action
    map, and drops and other actions under the tiered maps.
    """
    rows = d["per_seed"]
    flows = d["config"]["captures"]
    total = sum(flows.values())
    macros["IotFlows"] = _grouped(total)
    macros["IotSeeds"] = str(len(rows))
    pooled = []
    for cap, tag in IOT23_DEVICES:
        n = flows[cap]
        two = np.array([r["corpus"]["unseen_benign_dropped"][cap] for r in rows])
        pooled.append(two * n)
        macros.update({
            f"Iot{tag}Flows": _grouped(n),
            f"IotTwo{tag}": _pc(two.mean()) + r"\%",
            f"IotTwo{tag}Max": _pc(two.max()) + r"\%",
            f"IotTwo{tag}Seeds": str(int((two > 0).sum())),
            f"IotTwo{tag}Dropped": f"{two.mean() * n:.0f}",
        })
        for m, mt in (("4-action", "Four"), ("flood-inspect", "FloodInsp"),
                      ("3-action", "Three"), ("spoof-redirect", "Spoof")):
            drop = np.array([r["corpus"]["tiered"][m][cap]["dropped"] for r in rows])
            anya = np.array([r["corpus"]["tiered"][m][cap]["any_action"] for r in rows])
            macros.update({
                f"Iot{mt}Drop{tag}": _pc(drop.mean()) + r"\%",
                f"Iot{mt}Drop{tag}Max": _pc(drop.max()) + r"\%",
                f"Iot{mt}Drop{tag}Seeds": str(int((drop > 0).sum())),
                f"Iot{mt}Any{tag}": _pc(anya.mean()) + r"\%",
                f"Iot{mt}Any{tag}Max": _pc(anya.max()) + r"\%",
                f"Iot{mt}Any{tag}Seeds": str(int((anya > 0).sum())),
            })
    pooled = np.sum(pooled, axis=0) / total
    macros["IotTwoPooled"] = _pc(pooled.mean()) + r"\%"
    macros["IotTwoPooledMax"] = _pc(pooled.max()) + r"\%"
    # Seeds in which the flood-only map dropped any flow of any capture.
    four = [sum(r["corpus"]["tiered"]["4-action"][cap]["dropped"]
                for cap, _ in IOT23_DEVICES) for r in rows]
    macros["IotFourDropSeeds"] = str(sum(v > 0 for v in four))
    # Upper 95% bound on a rate observed as zero (rule of three).
    macros["IotZeroBoundAll"] = _pc(3 / total) + r"\%"
    macros["IotZeroBoundSomfy"] = _pc(3 / flows["iot23_somfy"], 0) + r"\%"


def benign_refresh_facts(br, macros: dict):
    """Unseen benign captures: drops, and adding one capture to training."""
    s = br["summary"]
    c = s["corpus"]
    macros.update({
        "BenRefreshSeeds": str(len(br["config"]["seeds"])),
        "BenRefreshCorpusDropped": _pc(c["corpus_benign_dropped"], 1) + r"\%",
        "BenRefreshTwo": _pc(c["unseen_benign_dropped"]["benign_unseen2"], 1)
        + r"\%",
        "BenRefreshThree": _pc(c["unseen_benign_dropped"]["benign_unseen3"], 1)
        + r"\%",
        "BenRefreshFlowsTwo": str(br["config"]["captures"]["benign_unseen2"]),
        "BenRefreshFlowsThree": str(br["config"]["captures"]["benign_unseen3"]),
    })
    app = c.get("dropped_in_application_attack_cells")
    if app:
        macros["BenRefreshAppCells"] = _pc(
            np.mean(list(app.values())), 0) + r"\%"
    major = c.get("dropped_by_cell_majority")
    if major:
        # Pooled over both captures and all seeds; 2 is DoS, 6 is spoofing.
        w = np.sum(list(major.values()), axis=0)
        macros["BenRefreshSpoofCells"] = _pc(w[6] / w.sum(), 0) + r"\%"
        macros["BenRefreshDosCells"] = _pc(w[2] / w.sum(), 0) + r"\%"
    for m, mt in (("3-action", "Three"), ("4-action", "Four"),
                  ("spoof-redirect", "Spoof"), ("flood-inspect", "Insp")):
        t = c.get("tiered", {}).get(m)
        if t:
            for n, nt in (("benign_unseen2", "Two"), ("benign_unseen3", "Three")):
                macros[f"BenRefreshTier{mt}{nt}"] = _pc(
                    t[n]["dropped"], 1) + r"\%"
                macros[f"BenRefreshTier{mt}Other{nt}"] = _pc(
                    t[n]["any_action"] - t[n]["dropped"], 1) + r"\%"
            if "corpus" in t:
                d = t["vs_two_action"]["coverage"]
                macros.update({
                    f"BenRefreshTier{mt}Cov": _pc(t["corpus"]["coverage"])
                    + r"\%",
                    f"BenRefreshTier{mt}CovDelta": _num(d["mean_points"]),
                    f"BenRefreshTier{mt}CovDeltaCI": _cis(d),
                    f"BenRefreshTier{mt}Err": _pc(
                        t["corpus"]["selective_error"], 2) + r"\%",
                    f"BenRefreshTier{mt}CorpusBen": _pc(
                        t["corpus"]["benign_dropped"], 1) + r"\%"})
    macros["BenRefreshCov"] = _pc(c["coverage"]) + r"\%"
    for added, other, tag in (("benign_unseen2", "benign_unseen3", "Three"),
                              ("benign_unseen3", "benign_unseen2", "Two")):
        e = s[f"with_{added}"][f"dropped_{other}"]
        macros.update({
            f"BenRefresh{tag}After": _pc(e["after"], 1) + r"\%",
            f"BenRefresh{tag}Delta": _num(e["mean_points"]),
            f"BenRefresh{tag}DeltaCI": f"[{_num(e['ci95'][0])}, {_num(e['ci95'][1])}]",
            f"BenRefresh{tag}Lower": str(e["lower_in"])})
    cov = s["with_benign_unseen2"]["coverage"]
    macros["BenRefreshCovDelta"] = _num(cov["mean_points"])
    macros["BenRefreshCovDeltaCI"] = f"[{_num(cov['ci95'][0])}, {_num(cov['ci95'][1])}]"


def capture_label_noise(benign, attack, macros: dict):
    """
    Flows of the attack capture slices, labelled by capture, that reach an
    external endpoint the unseen benign captures also reach.
    """
    import ipaddress

    def remote(key):
        return [a for a in (str(ipaddress.IPv4Address(k)) for k in key[:2])
                if not a.startswith("192.168.")]
    seen = {a for f in benign["flows"] if f["packets"] >= 10
            for a in remote(f["key"])}
    macros["CapNoiseEndpoints"] = str(len(seen))
    for cap, tag in (("recon", "Recon"), ("spoofing_dns", "Dns"),
                     ("bruteforce", "Brute")):
        fl = [f for f in attack["flows"] if f["captures"] == [cap]
              and f["label"] not in (None, 0) and f["packets"] >= 10]
        macros[f"CapNoise{tag}Flows"] = str(len(fl))
        macros[f"CapNoise{tag}Shared"] = str(sum(
            any(a in seen for a in remote(f["key"])) for f in fl))


def revision_outputs(tables: Path, figures: Path):
    """Tables, PGFPlots data and macros for the revised manuscript."""
    macros = {}
    study = _read("results/policy_study.json")
    unsw = _read("results/unsw_nb15/policy.json")
    bl = _read("results/baselines_study.json")
    dev = _read("results/dev/policy_study_dev_seeds0-29.json")
    if study:
        table_policy_main(study, tables / "policy_main.tex", macros)
        table_calibration(study, tables / "calibration.tex")
        data_calibration(study, figures / "calibration.dat")
        grouped = _read("results/policy_study_grouped.json")
        table_policy_generality(study, unsw, tables / "policy_generality.tex",
                                macros, grouped)
        table_per_action(study, grouped, tables / "per_action.tex", macros,
                         _read("results/confirm/policy_study_seeds60-89.json"),
                         _read("results/confirm/policy_study_grouped_seeds60-89.json"))
        data_policy_curves(study, unsw, figures / "policy.dat")
        # Macro F1 of the context-disjoint BiMamba runs per seed: the seed
        # whose test partition holds the largest group stands apart.
        grid = {}
        for ent, tag in (("none", "NoCtx"), ("real", "Ctx")):
            for sd, st in ((0, "Zero"), (1, "One"), (2, "Two")):
                g = _read(f"results/controlled/bimamba_{ent}_aug_context-disjoint"
                          f"_s{sd}_results.json")
                if g:
                    grid[f"GridSig{tag}Seed{st}"] = _pc(
                        g.get("test", g)["macro_f1"])
        partition_facts(study, macros, dev, grid)
    if dev:
        dev_calibration_facts(dev, macros)
    data_map_study(_read("results/map_study.json"),
                   _read("results/unsw_nb15/map_study.json"), figures, macros)
    if bl:
        table_policy_baselines(bl, tables / "policy_baselines.tex", macros)
    direct = _read("results/direct_action_study.json")
    if direct:
        direct_action_facts(direct, macros)
    ctl = _read("results/controller_sensitivity.json")
    if ctl:
        controller_sensitivity_facts(ctl, tables / "controller_sensitivity.tex",
                                     macros)
        data_controller_sensitivity(ctl, figures / "ctl_sens.dat")
    recon = _read("results/online_reconciliation.json")
    tiered = _read("results/online_reconciliation_tiered.json")
    meter, meter_trace = (_read("results/bmv2_meter_check.json"),
                          _read("results/trace/meter_check.json"))
    if recon and "bmv2_validation" in recon:
        table_online(recon, tiered, tables / "online.tex", macros, meter)
    if meter and meter_trace:
        ev = meter["pipeline_events"]
        sent = ev["rate_limit_pass"] + ev["rate_limit_drop"]
        macros.update({
            "MeterFlows": str(meter_trace["flows"]),
            "MeterPackets": str(meter_trace["packets_per_flow_after_decision"]),
            "MeterRate": _grouped(int(meter_trace["rate_pps"])),
            "MeterMetered": _grouped(sent),
            "MeterDrop": _grouped(ev["rate_limit_drop"]),
            "MeterDropShare": _pc(ev["rate_limit_drop"] / sent if sent else None)
            + r"\%"})
    if recon and recon.get("load_sweep"):
        table_load(recon, tables / "load.tex",
                   figures / "load_sweep.dat", macros)
    cap = _read("results/capture_replay.json")
    if cap:
        table_capture(cap, tables / "capture.tex", macros,
                      _read("results/capture_replay_paced.json"))
    heads = {p.stem: json.loads(p.read_text())
             for p in sorted(Path("results/capture/heads").glob("*.json"))}
    if heads:
        table_capture_heads(heads, tables / "capture_heads.tex", macros)
    flood_alert_facts(_read("results/flood/benign.json"),
                      _read("results/flood/heads.json"), heads, macros)
    sat = _read("results/controller_saturation.json")
    if sat:
        table_saturation(sat, _read("results/bmv2_saturation.json"),
                         tables / "saturation.tex", macros)
    # Packets the software switch processes at constant offered rates.
    rates = {}
    for r in (250, 500, 1000, 2000, 5000):
        run = _read(f"results/throughput/rate{r}.json")
        if run:
            rates[r] = (sum(run["pipeline_events"].values()),
                        run["replay"]["packets_sent"])
    if rates:
        for r, tag in ((250, "TwoFifty"), (500, "FiveHundred"), (1000, "One"),
                       (2000, "Two"), (5000, "Five")):
            if r in rates:
                macros[f"BmvRate{tag}"] = _pc(
                    rates[r][0] / rates[r][1]) + r"\%"
        macros["BmvRatePackets"] = _grouped(next(iter(rates.values()))[1])
        macros["BmvCapacity"] = _grouped(int(round(max(
            p / (n / r) for r, (p, n) in rates.items()), -1)))
    tt = _read("results/two_tier_study.json")
    if tt:
        table_two_tier(tt, tables / "two_tier.tex", figures / "twotier.dat",
                       macros)
    table_resources(Path("p4"), tables / "resources.tex", macros)
    tna = Path("p4/tna/compiled/tofino2")
    if (tna / "metrics.json").exists():
        tna_resources_facts(tna, tables / "resources_tna.tex", macros)
    tofino_model_facts(_read("results/tofino_model.json"),
                       _read("results/tofino_model_trace.json"), macros)
    dp = _read("results/dataplane_classifier.json")
    if dp:
        d = dp["deployment_policy"]
        macros.update({
            "DeployEntries": str(d["entries_installed"]),
            "DeployCoverage": _pc(d["coverage"]) + r"\%",
            "DeployError": _pc(d["false_drop_rate"], 2) + r"\%",
            "DeployThreshold": f"{d['threshold']:.2f}",
            "DeploySeed": str(dp["protocol"]["seed"]),
            "DeployCells": _grouped(dp["quantisation"]["cells_occupied"]),
            "DeployKeySpace": _grouped(dp["quantisation"]["key_space"]),
            "DeployCtlThreshold": f"{d['report_adjudicator']['threshold']:.2f}",
            "DeployCtlBudget": _pc(d["report_adjudicator"]["benign_block_budget"], 0)
            + r"\%",
        })
        if dp.get("tiered_policy"):
            t = dp["tiered_policy"]
            macros["TierEntries"] = str(t["entries"])
            for a, nm in (("1", "Drop"), ("2", "Rate"), ("3", "Redir")):
                macros[f"TierEntries{nm}"] = str(t["per_action"][a]["entries"])
        fn = _read("p4/generated/tiered_functional/classifier_entries.json")
        if fn:
            macros["TierFunctionalEntries"] = str(fn["entries_installed"])
    tune = _read("results/controlled/tune/summary.json")
    if tune and tune.get("selected"):
        sel, cfg = tune["selected"], tune["selected"]["config"]
        macros.update({
            "TuneConfigs": str(len(tune["trials"])),
            "TuneBestVal": _pc(sel["best_val_f1"]) + r"\%",
            "TuneValRange": f"{_pc(min(t['best_val_f1'] for t in tune['trials']))}"
            f" to {_pc(max(t['best_val_f1'] for t in tune['trials']))}",
            "TuneDModel": cfg["--d-model"], "TuneLayers": cfg["--n-layers"],
            "TuneDState": cfg["--d-state"], "TuneExpand": cfg["--expand"],
            "TuneDropout": cfg["--dropout"], "TuneLR": cfg["--lr"],
            "TuneEpochs": cfg["--epochs"]})
        for ent, tag in (("none", "NoCtx"), ("real", "Ctx")):
            v = [e["macro_f1"] for k, e in tune.get("final", {}).items()
                 if k.startswith(ent + "|")]
            if v:
                macros[f"TuneTest{tag}"] = (f"{100 * np.mean(v):.1f} $\\pm$ "
                                            f"{100 * np.std(v):.1f}")
                macros[f"TuneSeeds{tag}"] = str(len(v))
            run = _read(f"results/controlled/tune/tuned_{ent}_s0_results.json")
            if run:
                macros[f"TuneParams{tag}"] = (
                    f"{run['architecture']['n_params'] / 1000:.1f}K")
    # Until the random search has finished, mark its numbers as missing so
    # that a draft still compiles and nothing can pass for a result.
    for k in ("TuneConfigs", "TuneBestVal", "TuneValRange", "TuneDModel",
              "TuneLayers", "TuneDState", "TuneExpand", "TuneDropout", "TuneLR",
              "TuneEpochs", "TuneTestCtx", "TuneTestNoCtx", "TuneParamsCtx",
              "TuneParamsNoCtx"):
        macros.setdefault(k, r"\textbf{??}")
    rep = _read("results/causal/replay_summary.json")
    if rep:
        sm = rep["summary"]

        def rng3(k, d=3, pp=False):
            e = sm[k]
            return (f"{_num(e['min'], d)} to {_num(e['max'], d)}")
        macros.update({
            "CausalReplays": str(len(rep["replays"])),
            "CausalSplits": str(rep["split_seeds_per_replay"]),
            "CausalRRange": rng3("estimator_vs_truth_pearson_r"),
            "CausalTrueAUCRange": rng3("auc_truth"),
            "CausalEstAUCRange": rng3("auc_switch"),
            "CausalGainMean": _num(sm["diversity_gain_pp"]["mean"]),
            "CausalGainRange": rng3("diversity_gain_pp", 1),
            "CausalBeyondMean": _num(sm["beyond_class_marginal_pp"]["mean"]),
            "CausalBeyondRange": rng3("beyond_class_marginal_pp", 1),
            "CausalFlowsRange": f"{min(r['n_flows'] for r in rep['replays'])} to "
            f"{max(r['n_flows'] for r in rep['replays'])}",
        })
    confirmation_facts(macros)
    grid_paired_facts(macros)
    benign = _read("results/capture_benign_unseen.json")
    if benign:
        benign_capture_facts(benign, macros)
    tier = _read("results/capture_benign_unseen_tiered.json")
    if tier:
        cl = tier["all"]["all"].get("closed_loop", {})
        macros.update({
            "CapBenTierDecided": str(cl.get("decided_benign_flows")),
            "CapBenTierStopped": _pc(cl.get("decided_benign_stopped"), 1) + r"\%",
            "CapBenTierFast": str(cl.get("fast_benign")),
            "CapBenTierBlocked": str(cl.get("controller_blocked")),
            "CapBenTierAgree": _pc(cl.get("agreement_on_decidable_flows")) + r"\%"})
    spoof = _read("results/capture_benign_unseen_spoof.json")
    if spoof:
        cl = spoof["all"]["all"].get("closed_loop", {})
        macros.update({
            "CapBenSpoofDecided": str(cl.get("decided_benign_flows")),
            "CapBenSpoofStopped": _pc(cl.get("decided_benign_stopped"), 1)
            + r"\%",
            "CapBenSpoofFast": str(cl.get("fast_benign")),
            "CapBenSpoofBlocked": str(cl.get("controller_blocked")),
            "CapBenSpoofAgree": _pc(cl.get("agreement_on_decidable_flows"))
            + r"\%"})
    br = _read("results/benign_refresh.json")
    if br:
        benign_refresh_facts(br, macros)
    iot = _read("results/benign_iot23.json")
    if iot:
        benign_iot23_facts(iot, macros)
    brc = _read("results/confirm/benign_refresh_seeds60-89.json")
    if brc:
        tmp = {}
        benign_refresh_facts(brc, tmp)
        macros.update({"Conf" + k: v for k, v in tmp.items()})
    ben_man = _read("results/capture/benign_unseen.json")
    att_man = _read("results/capture/slices_merged.json")
    if ben_man and att_man:
        capture_label_noise(ben_man, att_man, macros)
    bad = [k for k in macros if not k.isalpha()]
    assert not bad, f"macro names must be letters only: {bad}"
    lines = [r"% Generated by scripts/plot_results.py. Do not edit."]
    lines += [rf"\newcommand{{\{k}}}{{{v}}}" for k,
              v in sorted(macros.items())]
    (tables / "macros_revision.tex").write_text("\n".join(lines) + "\n")
    logger.info("Revision outputs: %d macros", len(macros))
    return macros


def main():
    ap = argparse.ArgumentParser(description="Aggregate controlled results")
    ap.add_argument("--results", default="results/controlled")
    ap.add_argument(
        "--tables", default="reports/tables")
    ap.add_argument("--figures", default="reports/figures")
    args = ap.parse_args()

    res = Path(args.results)
    tables = Path(args.tables)
    figures = Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    (figures / "data").mkdir(parents=True, exist_ok=True)

    groups = load_grid(res)
    logger.info("Loaded %d configurations (%d runs)", len(groups),
                sum(len(v) for v in groups.values()))
    for key, recs in sorted(groups.items()):
        m, s, n = agg(recs, "macro_f1")
        logger.info("  %-16s aug=%-5s %-16s n=%d macro_f1=%.4f +- %.4f",
                    key[0], key[1], key[2], n, m, s)

    summary = {"configurations": {
        "|".join(map(str, k)): {
            "n_seeds": len(v),
            "accuracy_mean": agg(v, "accuracy")[0],
            "accuracy_std": agg(v, "accuracy")[1],
            "macro_f1_mean": agg(v, "macro_f1")[0],
            "macro_f1_std": agg(v, "macro_f1")[1],
            "per_class_f1": per_class_f1(v),
        } for k, v in groups.items()}}

    base_path = res / "baselines.json"
    baselines = json.loads(base_path.read_text()) if base_path.exists() else {}

    if groups:
        summary["main_effects"] = table_ablation(
            groups, tables / "ablation.tex")
        figure_ablation(groups, figures / "ablation.pdf")
        figure_context_controls(groups, figures / "context_controls.pdf")
        figure_protocol(groups, figures / "protocol.pdf")
        figure_confusion(groups, figures / "confusion.pdf")
        figure_per_class(groups, figures / "per_class_f1.pdf")

    # The context and protocol contrasts are reported for the sequence model
    # and for the reference ensemble together, so both are written whenever
    # either source of runs is present.
    if groups or baselines:
        summary["context_diagnostics"] = table_context_controls(
            groups, baselines, tables / "context_controls.tex")
        summary["protocol_effects"] = table_protocol(
            groups, baselines, tables / "protocol.tex")

    if baselines:
        table_baselines(baselines, groups, tables / "baselines.tex",
                        Path("results/benchmark_cpu.json"))
        figure_baselines(baselines, figures / "baselines.pdf")
        figure_cost(baselines, groups, Path("results/benchmark_cpu.json"),
                    figures / "cost.pdf")
        summary["baselines"] = {
            n: {c: {"accuracy": e[c]["accuracy_mean"],
                    "macro_f1": e[c]["macro_f1_mean"]}
                for c in e} for n, e in baselines["models"].items()}

    figure_flow_length(Path(baselines.get("protocol", {}).get(
        "corpus", "data/sequences/entropy_flows_v2.h5")),
        figures / "flow_length.pdf")

    dp_path = Path("results/dataplane_classifier.json")
    p4_path = Path("p4/feature_extraction.p4")
    if dp_path.exists() and p4_path.exists():
        dp = json.loads(dp_path.read_text())
        table_dataplane(dp, None, p4_path, tables / "dataplane.tex")
        summary["data_plane"] = {
            "accuracy": dp["test"]["accuracy"],
            "macro_f1": dp["test"]["macro_f1"],
            "decision_packets": dp["protocol"].get("decision_packets"),
            "resolved_fraction": dp["escalation"]["resolved_fraction"],
            "accuracy_on_resolved": dp["escalation"]["accuracy_on_resolved"],
            "feature_importance": dp["feature_importance"],
            "composed": dp.get("composed"),
            "controller": dp.get("controller"),
            "deployment_policy": dp.get("deployment_policy"),
        }

    write_macros(summary, baselines, tables / "macros.tex")

    # A second baseline suite fitted at a smaller per-class cap, used only to
    # show that the context decomposition does not depend on corpus size.
    scale_path = Path("results/scale_check_2500/baselines_2500.json")
    if scale_path.exists():
        sc = json.loads(scale_path.read_text())
        best = max(((n, e) for n, e in sc["models"].items()
                    if "real" in e and n != "MajorityClass"),
                   key=lambda kv: kv[1]["real"]["macro_f1_mean"])
        name, e = best
        gain = 100 * (e["real"]["macro_f1_mean"] - e["none"]["macro_f1_mean"])
        prior = 100 * (e["class-marginal"]["macro_f1_mean"]
                       - e["none"]["macro_f1_mean"])
        summary["scale_check"] = {
            "cap": sc["protocol"]["subsample_per_class"],
            "estimator": name,
            "context_gain_pp": round(gain, 2),
            "explained_by_class_marginal_pp": round(prior, 2),
            "beyond_class_marginal_pp": round(gain - prior, 2),
        }
        with open(tables / "macros.tex", "a") as fh:
            for k, v in (("ScaleFlows",
                          _grouped(corpus_flows(scale_path.parent))),
                         ("ScaleCtxGain", f"{gain:.1f}"),
                         ("ScaleCtxPrior", f"{prior:.1f}"),
                         ("ScaleCtxBeyond", f"{gain - prior:.1f}")):
                fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
        logger.info("Scale check at cap %s: gain %.1f, beyond control %.1f",
                    sc["protocol"]["subsample_per_class"], gain, gain - prior)

    causal_path = Path("results/causal_feature_study.json")
    if causal_path.exists():
        study = json.loads(causal_path.read_text())
        table_causal(study, tables / "causal.tex")
        figure_causal(study, figures / "causal.pdf")
        summary["causal_feature_study"] = study
        with open(tables / "macros.tex", "a") as fh:
            sep = study.get("ddos_dos_separation") or {}
            est, tru = sep.get("switch_estimate", {}), sep.get(
                "ground_truth", {})
            eci = est.get("auc_ci95", [0.0, 0.0])
            tci = tru.get("auc_ci95", [0.0, 0.0])
            for name, val in (
                    ("CausalFlows", study.get("n_flows")),
                    ("CausalPackets", study.get("packets_replayed")),
                    ("CausalEstimatorR",
                     f"{study.get('estimator_vs_truth_pearson_r', 0):.3f}"),
                    ("CausalEstAUC", f"{est.get('auc', 0):.3f}"),
                    ("CausalTrueAUC", f"{tru.get('auc', 0):.3f}"),
                    ("CausalEstAUCCI", f"[{eci[0]:.3f}, {eci[1]:.3f}]"),
                    ("CausalTrueAUCCI", f"[{tci[0]:.3f}, {tci[1]:.3f}]"),
                    ("CausalEstDDoS", f"{est.get('mean_a', 0):.2f}"),
                    ("CausalEstDoS", f"{est.get('mean_b', 0):.2f}"),
                    ("CausalTrueDDoS", f"{tru.get('mean_a', 0):.2f}"),
                    ("CausalTrueDoS", f"{tru.get('mean_b', 0):.2f}"),
                    ("CausalDivGain",
                     f"{study.get('diversity_gain_pp', 0):.1f}"),
                    ("CausalDivBeyond",
                     f"{study.get('diversity_gain_beyond_class_marginal_pp', 0):.1f}")):
                fh.write(f"\\newcommand{{\\{name}}}{{{val}}}\n")
        logger.info("Causal feature study folded in (%s flows)",
                    study.get("n_flows"))

    unsw_path = Path("results/unsw_nb15/decomposition.json")
    if unsw_path.exists():
        unsw = json.loads(unsw_path.read_text())
        table_unsw(unsw, tables / "unsw.tex")
        figure_unsw(unsw, figures / "unsw.pdf")
        best = max((m for m in unsw["models"]
                    if "decomposition" in unsw["models"][m]),
                   key=lambda m: unsw["models"][m]["real"]["macro_f1_mean"],
                   default=None)
        if best:
            d = unsw["models"][best]["decomposition"]
            summary["unsw_replication"] = {"estimator": best, **d}
            with open(tables / "macros.tex", "a") as fh:
                for k, v in (
                        ("UnswRows", _grouped(unsw["n_rows"])),
                        ("UnswClasses", len(unsw["classes"])),
                        ("UnswFeatures", len(unsw["context_features"])),
                        ("UnswBest", best),
                        ("UnswCtxGain",
                         _num(d['total_context_gain_pp'])),
                        ("UnswCtxPrior",
                         _num(d['explained_by_class_marginal_pp'])),
                        ("UnswCtxBeyond",
                         _num(d['beyond_class_marginal_pp']))):
                    fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
            logger.info("UNSW-NB15 replication: %s gain %.1f, beyond %.1f",
                        best, d["total_context_gain_pp"],
                        d["beyond_class_marginal_pp"])

    revision_outputs(tables, figures / "data")
    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", res / "summary.json")
    logger.info("Tables in %s, figures in %s", tables, figures)


if __name__ == "__main__":
    main()
