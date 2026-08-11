#!/usr/bin/env bash
# run_175k_experiments.sh - Train all models on 175K dataset (11 subtypes)
#
# Dataset: entropy_flows_v2_old_11subtypes.h5 (175,331 flows)
# Original hyperparameters tuned for this dataset size.
#
# Usage:
#   bash scripts/run_175k_experiments.sh [step]
#
# Arguments:
#   step: 1-6 or all (default: all)
#     1 = Data-plane CNN (tabular)
#     2 = Seq-CNN flow-isolated
#     3 = BiMamba no-ent flow-isolated
#     4 = BiMamba+ent flow-isolated [PROPOSED]
#     5 = Seq-CNN random (leakage demo)
#     6 = BiMamba+ent random (leakage demo)

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

DATA="data/sequences/entropy_flows_v2_old_11subtypes.h5"
DATA_TAB="data"
STEP="${1:-all}"

# ── 175K hyperparameters ─────────────────────────
EPOCHS=80
PATIENCE=15
BS_CNN=256
BS_MAMBA=64
D_MODEL=64
N_LAYERS=4
EXPAND=1
DROPOUT=0.2
LR=3e-4
WEIGHT_DECAY=0.01
FOCAL_GAMMA=2.0
WARMUP=5

if [ ! -f "${DATA}" ]; then
    echo "ERROR: ${DATA} not found!"
    exit 1
fi

# Ensure output directories exist
mkdir -p results models

echo "============================================"
echo " 175K Pipeline - $(date)"
echo " GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'CPU-only')"
echo " Step: ${STEP}"
echo " Model: d=${D_MODEL}, L=${N_LAYERS}, expand=${EXPAND}"
echo " Epochs: ${EPOCHS}, Patience: ${PATIENCE}"
echo "============================================"

step1() {
    echo ""
    echo ">>> [1/6] Data-plane CNN (tabular, 50 epochs)"
    PYTHONPATH=. python scripts/train_cnn.py \
        --data-dir "${DATA_TAB}" \
        --epochs 50 --patience 15 \
        --batch-size 256 --lr 1e-3 \
        --focal-gamma 2.0 \
        2>&1 | tee results/log_cnn.txt
}

step2() {
    echo ""
    echo ">>> [2/6] Seq-CNN (flow-isolated, ${EPOCHS} epochs)"
    PYTHONPATH=. python scripts/train_seq_cnn.py \
        --data "${DATA}" \
        --split flow-isolated \
        --name seq_cnn \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_CNN} --lr 1e-3 \
        2>&1 | tee results/log_seq_cnn.txt
}

step3() {
    echo ""
    echo ">>> [3/6] BiMamba d=${D_MODEL} L=${N_LAYERS} NO entropy (flow-isolated)"
    PYTHONPATH=. python scripts/train_bimamba.py \
        --data "${DATA}" \
        --split flow-isolated \
        --name bimamba_noent \
        --d-model ${D_MODEL} --n-layers ${N_LAYERS} --expand ${EXPAND} \
        --dropout ${DROPOUT} \
        --no-entropy --no-amp \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_MAMBA} --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --focal-gamma ${FOCAL_GAMMA} \
        --warmup-epochs ${WARMUP} \
        2>&1 | tee results/log_bimamba_noent.txt
}

step4() {
    echo ""
    echo ">>> [4/6] BiMamba d=${D_MODEL} L=${N_LAYERS} + ENTROPY (flow-isolated) [PROPOSED]"
    PYTHONPATH=. python scripts/train_bimamba.py \
        --data "${DATA}" \
        --split flow-isolated \
        --name bimamba_ent \
        --d-model ${D_MODEL} --n-layers ${N_LAYERS} --expand ${EXPAND} \
        --dropout ${DROPOUT} \
        --augment --no-amp \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_MAMBA} --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --focal-gamma ${FOCAL_GAMMA} \
        --warmup-epochs ${WARMUP} \
        2>&1 | tee results/log_bimamba_ent.txt
}

step5() {
    echo ""
    echo ">>> [5/6] Seq-CNN RANDOM split (leakage demo)"
    PYTHONPATH=. python scripts/train_seq_cnn.py \
        --data "${DATA}" \
        --split random \
        --name seq_cnn_random \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_CNN} --lr 1e-3 \
        2>&1 | tee results/log_seq_cnn_random.txt
}

step6() {
    echo ""
    echo ">>> [6/6] BiMamba+Entropy RANDOM split (leakage demo)"
    PYTHONPATH=. python scripts/train_bimamba.py \
        --data "${DATA}" \
        --split random \
        --name bimamba_ent_random \
        --d-model ${D_MODEL} --n-layers ${N_LAYERS} --expand ${EXPAND} \
        --dropout ${DROPOUT} \
        --augment --no-amp \
        --epochs ${EPOCHS} --patience ${PATIENCE} \
        --batch-size ${BS_MAMBA} --lr ${LR} \
        --weight-decay ${WEIGHT_DECAY} \
        --focal-gamma ${FOCAL_GAMMA} \
        --warmup-epochs ${WARMUP} \
        2>&1 | tee results/log_bimamba_ent_random.txt
}

case "${STEP}" in
    1)   step1 ;;
    2)   step2 ;;
    3)   step3 ;;
    4)   step4 ;;
    5)   step5 ;;
    6)   step6 ;;
    all)
        step1; step2; step3; step4; step5; step6
        ;;
    *)
        echo "Unknown step: ${STEP}"
        echo "Usage: $0 [1|2|3|4|5|6|all]"
        exit 1
        ;;
esac

echo ""
echo "============================================"
echo " 175K Training Complete - $(date)"
echo "============================================"
echo ""
for f in results/cnn_test_metrics.json results/seq_cnn_results.json results/bimamba_noent_results.json results/bimamba_ent_results.json results/seq_cnn_random_results.json results/bimamba_ent_random_results.json; do
    if [ -f "$f" ]; then
        model=$(basename "$f" _results.json)
        python3 -c "
import json
d = json.load(open('$f'))
split = d.get('training',{}).get('split','?')
print(f'  {\"$model\":25s} | split={split:15s} | acc={d[\"test\"][\"accuracy\"]:.4f} | F1={d[\"test\"][\"macro_f1\"]:.4f}')
"
    fi
done
