#!/usr/bin/env bash
# run_315k_experiments.sh - Full pipeline for 315K dataset (34 PCAP subtypes)
#
# Dataset: entropy_flows_v2.h5 (315,256 flows, 55MB)
#
# Steps:
#   0 = Extract PCAP -> HDF5 (skip if exists)
#   1 = Data-plane CNN (tabular, 50 epochs)
#   2 = Seq-CNN flow-isolated baseline
#   3 = BiMamba no-ent flow-isolated (ablation)
#   4 = BiMamba+ent flow-isolated [PROPOSED]
#   5 = Seq-CNN random (leakage demo)
#   6 = BiMamba+ent random (leakage demo)
#
# Hyperparameter configs for BiMamba (steps 3,4,6):
#   a = Wider+deeper (d=96, L=6) - RECOMMENDED
#   b = Deep narrow  (d=64, L=8) - fallback if a overfits
#   c = Wide+expand  (d=96, L=4, expand=2) - needs 8+ GB VRAM
#
# Usage:
#   bash scripts/run_315k_experiments.sh           # run ALL steps with config a (sequential)
#   bash scripts/run_315k_experiments.sh all        # same as above
#   bash scripts/run_315k_experiments.sh all a p    # run ALL with config a, PARALLEL where possible
#   bash scripts/run_315k_experiments.sh 4          # run only step 4 with config a
#   bash scripts/run_315k_experiments.sh 3 b        # run only step 3 with config b
#   bash scripts/run_315k_experiments.sh 3,4,6      # run steps 3, 4, and 6
#   bash scripts/run_315k_experiments.sh 3,4,6 c p  # run steps 3, 4, 6 with config c, parallel
#
# Parallel mode (3rd arg = "p"):
#   Steps 3+4 run simultaneously (~4.5 GB each, needs >= 10 GB VRAM)
#   Steps 5+6 run simultaneously
#   Saves ~40% wall time on T4 or better GPUs
#
# Prerequisites:
#   pip install -r requirements.txt

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATA="data/sequences/entropy_flows_v2.h5"
PCAP_DIR="dataset_cic_iot_2023/PCAP"
DATA_TAB="data"
STEP="${1:-all}"
CFG="${2:-a}"
PARALLEL="${3:-}"

# ── Hyperparameter configs ───────────────────────
case "${CFG}" in
    a)
        D_MODEL=96; N_LAYERS=6; EXPAND=1
        DROPOUT=0.25; LR=2e-4; WEIGHT_DECAY=0.02
        FOCAL_GAMMA=2.5; WARMUP=8
        EPOCHS=150; PATIENCE=25
        CFG_DESC="Config A: d=96, L=6, expand=1, gamma=2.5 (recommended)"
        ;;
    b)
        D_MODEL=64; N_LAYERS=8; EXPAND=1
        DROPOUT=0.3; LR=2e-4; WEIGHT_DECAY=0.03
        FOCAL_GAMMA=3.0; WARMUP=10
        EPOCHS=150; PATIENCE=25
        CFG_DESC="Config B: d=64, L=8, expand=1, gamma=3.0 (deep narrow)"
        ;;
    c)
        D_MODEL=96; N_LAYERS=4; EXPAND=2
        DROPOUT=0.2; LR=3e-4; WEIGHT_DECAY=0.01
        FOCAL_GAMMA=2.0; WARMUP=5
        EPOCHS=120; PATIENCE=20
        CFG_DESC="Config C: d=96, L=4, expand=2, gamma=2.0 (wide, needs 8+ GB)"
        ;;
    *)
        echo "Unknown config: ${CFG}. Use a, b, or c."
        exit 1
        ;;
esac

BS_CNN=256
BS_MAMBA=64
CKPT_EVERY=25
SUFFIX="_315k_${CFG}"
CKPT_DIR="models/checkpoints"

# Ensure output directories exist
mkdir -p results models "${CKPT_DIR}"

# ── Resume helper ────────────────────────────────
# Find the latest periodic checkpoint for a given model name.
# Usage: RESUME_ARG=$(find_latest_ckpt "bimamba_ent_315k_a")
# Returns "--resume <path>" if a checkpoint exists, empty string otherwise.
find_latest_ckpt() {
    local name="$1"
    local latest
    latest=$(ls -1v "${CKPT_DIR}/${name}_epoch"*.pt 2>/dev/null | tail -n1)
    if [ -n "$latest" ] && [ -f "$latest" ]; then
        local ep
        ep=$(echo "$latest" | grep -oP 'epoch\K[0-9]+')
        echo "--resume ${latest}"
        echo "  [AUTO-RESUME] Found checkpoint: ${latest} (epoch ${ep})" >&2
    fi
}

echo "============================================"
echo " 315K Pipeline - $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU-only')"
echo " VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo " Step: ${STEP} | ${CFG_DESC}"
echo " Epochs: ${EPOCHS}, Patience: ${PATIENCE}"
echo "============================================"

# ── Step functions ───────────────────────────────

step0() {
    if [ -f "${DATA}" ]; then
        echo ""
        echo ">>> [0/6] Data extraction - ALREADY DONE, skipping"
        echo "  $(ls -lh ${DATA} | awk '{print $5, $9}')"
    else
        echo ""
        echo ">>> [0/6] Extracting PCAPs -> ${DATA}"
        PYTHONPATH=. python scripts/pcap_to_entropy_fast.py \
            --pcap-dir "${PCAP_DIR}" \
            --output "${DATA}" \
            --workers 3 --window 10.0 \
            2>&1 | tee results/log_extraction_315k.txt
    fi
}

step1() {
    echo ""
    if [ -f "results/cnn_test_metrics.json" ]; then
        echo ">>> [1/6] Data-plane CNN (tabular) - ALREADY DONE, skipping"
    elif [ ! -f "${DATA_TAB}/train.npz" ]; then
        echo ">>> [1/6] Data-plane CNN (tabular) - SKIPPED (${DATA_TAB}/train.npz not found)"
        echo "    Run scripts/prepare_datasets.py first to create tabular NPZ files"
    else
        echo ">>> [1/6] Data-plane CNN (tabular, 50 epochs)"
        PYTHONPATH=. python scripts/train_cnn.py \
            --data-dir "${DATA_TAB}" \
            --epochs 50 --patience 15 \
            --batch-size 256 --lr 1e-3 \
            --focal-gamma 2.0 \
            2>&1 | tee results/log_cnn.txt
    fi
}

step2() {
    echo ""
    echo ">>> [2/6] Seq-CNN (flow-isolated, ${EPOCHS} epochs)"
    local resume_arg
    resume_arg=$(find_latest_ckpt "seq_cnn${SUFFIX}")
    PYTHONPATH=. python scripts/train_seq_cnn.py \
        --data "${DATA}" \
        --split flow-isolated \
        --name seq_cnn${SUFFIX} \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_CNN} --lr 1e-3 \
        --checkpoint-every ${CKPT_EVERY} \
        ${resume_arg} \
        2>&1 | tee results/log_seq_cnn${SUFFIX}.txt
}

step3() {
    echo ""
    echo ">>> [3/6] BiMamba d=${D_MODEL} L=${N_LAYERS} NO entropy (flow-isolated)"
    local resume_arg
    resume_arg=$(find_latest_ckpt "bimamba_noent${SUFFIX}")
    PYTHONPATH=. python scripts/train_bimamba.py \
        --data "${DATA}" \
        --split flow-isolated \
        --name bimamba_noent${SUFFIX} \
        --d-model ${D_MODEL} --n-layers ${N_LAYERS} --expand ${EXPAND} \
        --d-state 16 --d-conv 4 \
        --dropout ${DROPOUT} \
        --no-entropy --no-amp \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_MAMBA} --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --focal-gamma ${FOCAL_GAMMA} \
        --warmup-epochs ${WARMUP} \
        --checkpoint-every ${CKPT_EVERY} \
        ${resume_arg} \
        2>&1 | tee results/log_bimamba_noent${SUFFIX}.txt
}

step4() {
    echo ""
    echo ">>> [4/6] BiMamba d=${D_MODEL} L=${N_LAYERS} + ENTROPY (flow-isolated) [PROPOSED]"
    local resume_arg
    resume_arg=$(find_latest_ckpt "bimamba_ent${SUFFIX}")
    PYTHONPATH=. python scripts/train_bimamba.py \
        --data "${DATA}" \
        --split flow-isolated \
        --name bimamba_ent${SUFFIX} \
        --d-model ${D_MODEL} --n-layers ${N_LAYERS} --expand ${EXPAND} \
        --d-state 16 --d-conv 4 \
        --dropout ${DROPOUT} \
        --augment --no-amp \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_MAMBA} --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --focal-gamma ${FOCAL_GAMMA} \
        --warmup-epochs ${WARMUP} \
        --checkpoint-every ${CKPT_EVERY} \
        ${resume_arg} \
        2>&1 | tee results/log_bimamba_ent${SUFFIX}.txt
}

step5() {
    echo ""
    echo ">>> [5/6] Seq-CNN RANDOM split (leakage demo)"
    local resume_arg
    resume_arg=$(find_latest_ckpt "seq_cnn_random${SUFFIX}")
    PYTHONPATH=. python scripts/train_seq_cnn.py \
        --data "${DATA}" \
        --split random \
        --name seq_cnn_random${SUFFIX} \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_CNN} --lr 1e-3 \
        --checkpoint-every ${CKPT_EVERY} \
        ${resume_arg} \
        2>&1 | tee results/log_seq_cnn_random${SUFFIX}.txt
}

step6() {
    echo ""
    echo ">>> [6/6] BiMamba+Entropy RANDOM split (leakage demo)"
    local resume_arg
    resume_arg=$(find_latest_ckpt "bimamba_ent_random${SUFFIX}")
    PYTHONPATH=. python scripts/train_bimamba.py \
        --data "${DATA}" \
        --split random \
        --name bimamba_ent_random${SUFFIX} \
        --d-model ${D_MODEL} --n-layers ${N_LAYERS} --expand ${EXPAND} \
        --d-state 16 --d-conv 4 \
        --dropout ${DROPOUT} \
        --augment --no-amp \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_MAMBA} --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --focal-gamma ${FOCAL_GAMMA} \
        --warmup-epochs ${WARMUP} \
        --checkpoint-every ${CKPT_EVERY} \
        ${resume_arg} \
        2>&1 | tee results/log_bimamba_ent_random${SUFFIX}.txt
}

# ── Dispatch ─────────────────────────────────────
wait_pids() {
    # Wait for background PIDs passed as args; exit on any failure
    local pids=("$@")
    local failed=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            echo "ERROR: Background process $pid failed!"
            failed=1
        fi
    done
    if [ "$failed" -eq 1 ]; then
        echo "One or more parallel jobs failed. Aborting."
        exit 1
    fi
}

run_steps() {
    local steps="$1"
    IFS=',' read -ra STEP_LIST <<< "$steps"
    for s in "${STEP_LIST[@]}"; do
        case "$s" in
            0) step0 ;;
            1) step1 ;;
            2) step2 ;;
            3) step3 ;;
            4) step4 ;;
            5) step5 ;;
            6) step6 ;;
            *) echo "Unknown step: $s (use 0-6)"; exit 1 ;;
        esac
    done
}

run_all_sequential() {
    step0; step1; step2; step3; step4; step5; step6
}

run_all_parallel() {
    # Steps 0, 1, 2: sequential (extraction, CNN, Seq-CNN)
    step0
    step1
    step2

    # Steps 3+4: parallel (BiMamba noent + BiMamba ent, ~4.5 GB each)
    echo ""
    echo ">>> PARALLEL: Running steps 3 and 4 simultaneously"
    step3 &
    local pid3=$!
    step4 &
    local pid4=$!
    wait_pids "$pid3" "$pid4"

    # Steps 5+6: parallel (leakage demos)
    echo ""
    echo ">>> PARALLEL: Running steps 5 and 6 simultaneously"
    step5 &
    local pid5=$!
    step6 &
    local pid6=$!
    wait_pids "$pid5" "$pid6"
}

case "${STEP}" in
    all)
        if [ "${PARALLEL}" = "p" ]; then
            echo ">>> MODE: Parallel (steps 3+4 and 5+6 will run concurrently)"
            run_all_parallel
        else
            run_all_sequential
        fi
        ;;
    *)
        if [ "${PARALLEL}" = "p" ]; then
            # Run comma-separated steps in parallel
            echo ">>> MODE: Parallel (all specified steps run concurrently)"
            IFS=',' read -ra STEP_LIST <<< "${STEP}"
            PIDS=()
            for s in "${STEP_LIST[@]}"; do
                case "$s" in
                    0) step0 & PIDS+=($!) ;;
                    1) step1 & PIDS+=($!) ;;
                    2) step2 & PIDS+=($!) ;;
                    3) step3 & PIDS+=($!) ;;
                    4) step4 & PIDS+=($!) ;;
                    5) step5 & PIDS+=($!) ;;
                    6) step6 & PIDS+=($!) ;;
                    *) echo "Unknown step: $s"; exit 1 ;;
                esac
            done
            wait_pids "${PIDS[@]}"
        else
            run_steps "${STEP}"
        fi
        ;;
esac

# ── Summary ──────────────────────────────────────
echo ""
echo "============================================"
echo " 315K Training Complete - $(date)"
echo "============================================"
echo ""
for f in results/*315k*_results.json results/cnn_test_metrics.json; do
    if [ -f "$f" ]; then
        model=$(basename "$f" _results.json)
        python3 -c "
import json
d = json.load(open('$f'))
t = d.get('test', d)
split = d.get('training',{}).get('split', d.get('split','?'))
print(f'  {\"$model\":35s} | split={split:15s} | acc={t[\"accuracy\"]:.4f} | F1={t[\"macro_f1\"]:.4f}')
" 2>/dev/null || true
    fi
done
