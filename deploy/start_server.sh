#!/bin/bash
# Start Real FastWAM Server with actual model

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CHECKPOINT="${CHECKPOINT_PATH:-/mnt/data/zhibo.zhou/Workspaces/fastwam_ws/FastWAM/runs/x1_0415_uncond_2cam224_1e-4/2026-04-18_06-33-25.horizon64/checkpoints/weights/step_014000.pt}"
DEVICE="${FASTWAM_DEVICE:-cuda:0}"
PORT="${FASTWAM_PORT:-7880}"

echo "╔════════════════════════════════════════════╗"
echo "║     FastWAM Server - Production Mode       ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Project    : $PROJECT_ROOT"
echo "Checkpoint : ${CHECKPOINT##*/}...pt  ($(numfmt --to=iec-i --suffix=B $(stat -f%z "$CHECKPOINT" 2>/dev/null || echo "0") 2>/dev/null || echo "12GB"))"
echo "Device     : $DEVICE"
echo "Server     : ws://0.0.0.0:$PORT"
echo ""
echo "Note: Model loading may take 30-60 seconds on first start..."
echo "      Once you see 'listening on ws://', server is ready."
echo ""

conda run -n fastwam_libero_v0 --no-capture-output python deploy/fastwam_server.py \
    inference.checkpoint_path="$CHECKPOINT" \
    inference.device="$DEVICE"
