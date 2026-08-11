#!/usr/bin/env bash
# ==========================================================
# setup_mininet_env.sh - Set up lightweight Python venv for
# Split-Mamba Mininet benchmarking.
#
# No PyTorch, no conda. Only numpy + onnxruntime + scapy.
# Total install size: ~80MB
#
# Usage:
#   cd enn_ddos
#   bash scripts/setup_mininet_env.sh
#   source venv/bin/activate
# ==========================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  Split-Mamba: Mininet Environment Setup"
echo "============================================"

# Check Python 3.10+
PYTHON=""
for py in python3.12 python3.11 python3.10 python3; do
    if command -v "$py" &>/dev/null; then
        ver=$("$py" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ]; then
            PYTHON="$py"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required. Install with:"
    echo "  sudo apt install python3.11 python3.11-venv"
    exit 1
fi

echo "[1/4] Using $PYTHON ($($PYTHON --version 2>&1))"

# Create venv
if [ -d "venv" ]; then
    echo "[2/4] venv already exists, skipping creation"
else
    echo "[2/4] Creating virtual environment..."
    $PYTHON -m venv venv
fi

# Activate
source venv/bin/activate

# Install packages
echo "[3/4] Installing packages (numpy, onnxruntime, scapy)..."
pip install --upgrade pip setuptools wheel -q
pip install \
    numpy \
    onnxruntime \
    scapy \
    grpcio \
    protobuf \
    -q

# Verify
echo "[4/4] Verifying installation..."
python -c "
import numpy as np
import onnxruntime as ort
print(f'  numpy:       {np.__version__}')
print(f'  onnxruntime: {ort.__version__}')
try:
    import scapy
    print(f'  scapy:       {scapy.__version__}' if hasattr(scapy, '__version__') else '  scapy:       OK')
except: print('  scapy:       MISSING')
try:
    import grpc
    print(f'  grpcio:      OK')
except: print('  grpcio:      MISSING')
"

# Verify ONNX model loads
if [ -f "models/onnx/bimamba_ent.onnx" ]; then
    python -c "
import onnxruntime as ort
import numpy as np
sess = ort.InferenceSession('models/onnx/bimamba_ent.onnx', providers=['CPUExecutionProvider'])
seq = np.zeros((1, 100, 3), dtype=np.float32)
ent = np.zeros((1, 6), dtype=np.float32)
out = sess.run(None, {sess.get_inputs()[0].name: seq, sess.get_inputs()[1].name: ent})
print(f'  ONNX model:  OK (output shape {out[0].shape})')
"
else
    echo "  WARNING: models/onnx/bimamba_ent.onnx not found"
fi

echo ""
echo "============================================"
echo "  Setup complete!"
echo ""
echo "  Activate:  source venv/bin/activate"
echo ""
echo "  Run benchmark:"
echo "    python scripts/benchmark_onnx.py --full-pipeline"
echo ""
echo "  Run controller:"
echo "    python controller/sdn_controller.py \\"
echo "      --model models/onnx/bimamba_ent.onnx"
echo ""
echo "  Run Mininet:"
echo "    sudo -E env PATH=\$PATH python topology/mininet_topo.py \\"
echo "      --p4-json p4/build/feature_extraction.json"
echo "============================================"
