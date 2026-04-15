#!/bin/bash
# Offline action inference on custom dataset
# Usage:
#   bash experiments/infer/run_infer.sh
#   bash experiments/infer/run_infer.sh inference.checkpoint_path=/path/to/ckpt.pt inference.num_samples=5

set -euo pipefail
cd "$(dirname "$0")/../.."  # cd to FastWAM root

python experiments/infer/infer_action_offline.py \
    --config-name infer_x1_insert \
    "$@"
