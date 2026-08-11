"""
run_experiment.py - Driver for the controlled experiment grid.

Every run in the grid shares one corpus, one architecture, one optimiser
setting and one epoch budget. Runs differ only in the factors declared in
GRID, so any difference in a reported metric is attributable to the factor
that was varied. This is what makes the ablation an ablation rather than a
comparison of two differently-tuned systems.

Usage:
    python scripts/run_experiment.py --stage ablation
    python scripts/run_experiment.py --stage all --dry-run
"""

import argparse
import itertools
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

CORPUS = "data/sequences/entropy_flows_v2.h5"
RESULTS_DIR = "results/controlled"
MODELS_DIR = "models/controlled"

# Held fixed across every run in the grid. Corpus, subsample and truncation
# match the baseline suite exactly so the two are directly comparable; the
# remaining entries are capacity and optimiser settings, which are shared by
# every cell and therefore cannot favour one.
BASE = [
    "--data", CORPUS,
    "--d-model", "64",
    "--n-layers", "4",
    "--d-state", "8",
    "--expand", "1",
    "--seq-len", "32",
    "--subsample-per-class", "12000",
    "--epochs", "30",
    "--batch-size", "512",
    "--lr", "6e-4",
    "--patience", "12",
    "--focal-gamma", "2.0",
    "--compile",
    "--checkpoint-every", "0",
]

SEEDS = (0, 1, 2)

# stage -> list of (entropy_mode, augment, split)
GRID = {
    # Isolates the context channel and the augmentation policy. The
    # earlier comparison varied both at once; this varies one at a time.
    "ablation": [
        (ent, aug, "flow-isolated")
        for ent, aug in itertools.product(("none", "real"), (False, True))
    ],
    # Quantifies how much of the context channel's contribution survives
    # once its per-flow correspondence is destroyed.
    "context-controls": [
        ("class-marginal", True, "flow-isolated"),
        ("shuffle", True, "flow-isolated"),
    ],
    # Repeats the headline configurations under a protocol in which flows
    # sharing a context window cannot straddle the split, and under the
    # row-level protocol whose optimism is being quantified.
    "protocol": [
        ("none", True, "context-disjoint"),
        ("real", True, "context-disjoint"),
        ("none", True, "random"),
        ("real", True, "random"),
    ],
}


def run_name(ent: str, aug: bool, split: str, seed: int) -> str:
    return f"bimamba_{ent}_{'aug' if aug else 'noaug'}_{split}_s{seed}"


def build_command(ent: str, aug: bool, split: str, seed: int) -> list:
    cmd = [sys.executable, "scripts/train_bimamba.py", *BASE,
           "--entropy-mode", ent,
           "--split", split,
           "--seed", str(seed),
           "--name", run_name(ent, aug, split, seed),
           "--results-dir", RESULTS_DIR,
           "--output-dir", MODELS_DIR]
    if aug:
        cmd.append("--augment")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Controlled experiment grid")
    parser.add_argument("--stage", default="all",
                        choices=[*GRID.keys(), "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true",
                        help="Re-run configurations that already have results")
    args = parser.parse_args()

    stages = list(GRID) if args.stage == "all" else [args.stage]
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    jobs = [(ent, aug, split, seed)
            for stage in stages
            for ent, aug, split in GRID[stage]
            for seed in args.seeds]

    logger.info("%d run(s) queued across stage(s): %s",
                len(jobs), ", ".join(stages))

    env = dict(os.environ, PYTHONPATH=".",
               PYTORCH_ALLOC_CONF="expandable_segments:True")

    for i, job in enumerate(jobs, 1):
        name = run_name(*job)
        out = Path(RESULTS_DIR) / f"{name}_results.json"
        if out.exists() and not args.force:
            logger.info("[%d/%d] %s already done, skipping",
                        i, len(jobs), name)
            continue

        cmd = build_command(*job)
        if args.dry_run:
            logger.info("[%d/%d] %s", i, len(jobs), " ".join(cmd))
            continue

        logger.info("[%d/%d] %s", i, len(jobs), name)
        t0 = time.time()
        log_path = Path(RESULTS_DIR) / f"{name}.log"
        with open(log_path, "w") as lf:
            rc = subprocess.run(cmd, env=env, stdout=lf,
                                stderr=subprocess.STDOUT).returncode
        if rc != 0:
            logger.error("  FAILED (rc=%d), see %s", rc, log_path)
            continue

        with open(out) as f:
            res = json.load(f)
        logger.info("  done in %.0fs | acc=%.4f macro_f1=%.4f",
                    time.time() - t0,
                    res["test"]["accuracy"], res["test"]["macro_f1"])


if __name__ == "__main__":
    main()
