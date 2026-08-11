#!/usr/bin/env bash
# ==========================================================
# pack_for_mininet.sh - Create a minimal tarball with only
# the files needed for Mininet benchmarking.
#
# Usage:
#   bash scripts/pack_for_mininet.sh
#   # Transfer enn_ddos_mininet.tar.gz to the Mininet machine
#   # Then on that machine:
#   #   tar xzf enn_ddos_mininet.tar.gz
#   #   cd enn_ddos
#   #   bash scripts/setup_mininet_env.sh
# ==========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

OUTFILE="enn_ddos_mininet.tar.gz"

echo "Packaging files for Mininet deployment..."

tar czf "$OUTFILE" \
    --transform='s,^,enn_ddos/,' \
    \
    controller/__init__.py \
    controller/onnx_inference.py \
    controller/sdn_controller.py \
    controller/p4runtime_client.py \
    controller/telemetry_parser.py \
    controller/mamba_inference.py \
    \
    scripts/benchmark_onnx.py \
    scripts/setup_mininet_env.sh \
    scripts/replay_attack.py \
    \
    topology/mininet_topo.py \
    \
    p4/feature_extraction.p4 \
    p4/headers.p4 \
    p4/parsers.p4 \
    \
    models/onnx/bimamba_ent.onnx \
    \
    config/experiment_config.yaml \
    \
    2>/dev/null || true

# Check if controller/__init__.py exists, create if not
if ! tar tzf "$OUTFILE" | grep -q "controller/__init__.py"; then
    # Recreate with all files that exist
    FILES=""
    for f in \
        controller/onnx_inference.py \
        controller/sdn_controller.py \
        controller/p4runtime_client.py \
        controller/telemetry_parser.py \
        controller/mamba_inference.py \
        scripts/benchmark_onnx.py \
        scripts/setup_mininet_env.sh \
        scripts/replay_attack.py \
        topology/mininet_topo.py \
        p4/feature_extraction.p4 \
        p4/headers.p4 \
        p4/parsers.p4 \
        models/onnx/bimamba_ent.onnx \
        config/experiment_config.yaml \
    ; do
        [ -f "$f" ] && FILES="$FILES $f"
    done

    tar czf "$OUTFILE" --transform='s,^,enn_ddos/,' $FILES
fi

SIZE=$(du -h "$OUTFILE" | cut -f1)
echo ""
echo "Created: $OUTFILE ($SIZE)"
echo ""
echo "Contents:"
tar tzf "$OUTFILE" | head -30
echo ""
echo "Transfer to Mininet machine:"
echo "  scp $OUTFILE user@mininet-host:~/"
echo ""
echo "On the Mininet machine:"
echo "  tar xzf $OUTFILE"
echo "  cd enn_ddos"
echo "  bash scripts/setup_mininet_env.sh"
echo "  source venv/bin/activate"
echo "  python scripts/benchmark_onnx.py --full-pipeline"
