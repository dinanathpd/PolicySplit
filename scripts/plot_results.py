"""
plot_results.py - Aggregate the controlled runs into tables and figures.

Reads every result file produced by the experiment grid, the baseline
suite, the data-plane classifier and the BMv2 test bed, aggregates
repeated seeds, and writes

  * results/controlled/summary.json  - one machine-readable record,
  * reports/<venue>/tables/*.tex     - LaTeX tables the manuscript
                                       includes verbatim,
  * reports/figures/*.pdf            - vector figures.

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
    # match the Times body text of the manuscript; Liberation Serif is a
    # TrueType clone with Times metrics, so it subsets cleanly into the PDF
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Nimbus Roman",
                   "TeX Gyre Termes", "FreeSerif", "DejaVu Serif"],
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
    "bruteforce": "BruteForce",
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


def table_baselines(baselines, groups, out: Path):
    """Estimators compared on one corpus, one partition, one feature set."""
    lines = [
        r"\begin{tabular}{lcccc}", r"\toprule",
        r"& \multicolumn{2}{c}{No context} & \multicolumn{2}{c}{Measured "
        r"context} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r"Estimator & Accuracy (\%) & Macro F1 (\%) & Accuracy (\%) "
        r"& Macro F1 (\%) \\", r"\midrule"]
    for name, entry in baselines["models"].items():
        cells = []
        for ctx in ("none", "real"):
            e = entry.get(ctx)
            cells += ([fmt(e["accuracy_mean"], e["accuracy_std"]),
                       fmt(e["macro_f1_mean"], e["macro_f1_std"])]
                      if e else [NA, NA])
        lines.append(f"{name} & " + " & ".join(cells) + r" \\")

    lines.append(r"\midrule")
    no_ctx = groups.get(("none", True, "flow-isolated"))
    ctx = groups.get(("real", True, "flow-isolated"))
    if no_ctx and ctx:
        cells = [fmt(*agg(no_ctx, "accuracy")[:2]),
                 fmt(*agg(no_ctx, "macro_f1")[:2]),
                 fmt(*agg(ctx, "accuracy")[:2]),
                 fmt(*agg(ctx, "macro_f1")[:2])]
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

    rows = [
        ("Register arrays", f"{res['n_registers']}"),
        ("Register state", f"{res['register_bits'] / 8 / 1024:.1f} KiB"),
        ("Classifier table capacity",
         f"{tables.get('flow_classifier', 0)} entries"),
        ("Candidate policy entries (offline)",
         f"{dp['table']['entries_installed']}"),
        ("Candidate key space covered (offline)",
         f"{dp['table']['entries_installed']}/{dp['table']['key_space']}"),
        ("Mitigation table capacity",
         f"{tables.get('mitigation', 0)} entries"),
        ("Fine-label accuracy (offline diagnostic)",
         f"{100 * dp['test']['accuracy']:.1f}\\%"),
        ("Fine-label macro F1 (offline diagnostic)",
         f"{100 * dp['test']['macro_f1']:.1f}\\%"),
        ("Fast-path policy", "drop only"),
        ("Selective drop-error budget (offline)",
         f"{100 * dp['deployment_policy']['false_drop_budget']:.1f}\\%"),
        ("Candidate drop coverage (offline)",
         f"{100 * dp['deployment_policy']['coverage']:.1f}\\%"),
        ("Benign share among candidate drops",
         f"{100 * dp['deployment_policy']['false_drop_rate']:.1f}\\%"),
    ]
    if bmv2:
        rows.append(("Classifier entries installed in replay",
                     f"{bmv2['control_state']['classifier_entries']}"))
        lat = [t["escalation_latency_ms"]["median"]
               for t in bmv2["trials"].values()
               if t["escalation_latency_ms"]["median"] is not None]
        if lat:
            rows.append(("Escalation latency (median over captures)",
                         f"{np.median(lat):.1f} ms"))
        if bmv2.get("mitigation"):
            rows.append(("Mitigation rule install",
                         f"{bmv2['mitigation']['rule_install_ms']:.0f} ms"))
            rows.append(("Packets dropped by installed rule",
                         f"{bmv2['mitigation']['mitigation_hits_after_rule']}"))

    lines = [r"\begin{tabular}{ll}", r"\toprule",
             r"Quantity & Value \\", r"\midrule"]
    lines += [f"{k} & {v} \\\\" for k, v in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


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

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.5),
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
        return f"{v:.1f}" if v is not None else "??"

    def best(context):
        """Highest macro F1 over the ensembles at a given context."""
        vals = [(n, e[context]["macro_f1"])
                for n, e in summary.get("baselines", {}).items()
                if context in e and n not in ("MajorityClass",)]
        return max(vals, key=lambda kv: kv[1]) if vals else (None, None)

    best_ctx_name, best_ctx = best("real")
    _, best_noctx = best("none")

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
    lines = [
        r"\begin{tabular}{lrrr}", r"\toprule",
        r"Family & Flows & In-switch estimate & True distinct sources \\",
        r"\midrule"]
    for cid in sorted(per, key=int):
        e = per[cid]
        name = CLASSES[int(cid)] if int(cid) < len(CLASSES) else cid
        lines.append(f"{name} & {e['n']} & "
                     f"{e['switch_estimate_mean']:.2f} & "
                     f"{e['true_sources_mean']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Aggregate controlled results")
    ap.add_argument("--results", default="results/controlled")
    ap.add_argument(
        "--tables", default="reports/discover-cybersecurity/tables")
    ap.add_argument("--figures",
                    default="reports/discover-cybersecurity/figures")
    args = ap.parse_args()

    res = Path(args.results)
    tables = Path(args.tables)
    figures = Path(args.figures)
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

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
        table_baselines(baselines, groups, tables / "baselines.tex")
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
    bmv2_path = Path("results/bmv2_dataplane.json")
    p4_path = Path("p4/feature_extraction.p4")
    if dp_path.exists() and p4_path.exists():
        dp = json.loads(dp_path.read_text())
        bmv2 = json.loads(bmv2_path.read_text()
                          ) if bmv2_path.exists() else None
        table_dataplane(dp, bmv2, p4_path, tables / "dataplane.tex")
        policy = aggregate_policy(sorted(Path("results").glob(
            "dataplane_policy_seed*.json")))
        if policy:
            table_policy(policy, tables / "policy.tex")
            figure_policy(policy, figures / "policy.pdf")
        if bmv2:
            table_bmv2_trials(bmv2, tables / "bmv2_trials.tex")
            figure_dataplane(bmv2, figures / "dataplane.pdf")
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
            "policy_comparison": policy,
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
                         f"{d['total_context_gain_pp']:.1f}"),
                        ("UnswCtxPrior",
                         f"{d['explained_by_class_marginal_pp']:.1f}"),
                        ("UnswCtxBeyond",
                         f"{d['beyond_class_marginal_pp']:.1f}")):
                    fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
            logger.info("UNSW-NB15 replication: %s gain %.1f, beyond %.1f",
                        best, d["total_context_gain_pp"],
                        d["beyond_class_marginal_pp"])

    (res / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Summary written to %s", res / "summary.json")
    logger.info("Tables in %s, figures in %s", tables, figures)


if __name__ == "__main__":
    main()
