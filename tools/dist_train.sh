#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: bash tools/dist_train.sh CONFIG GPUS_PER_NODE [train.py arguments...]" >&2
    exit 2
fi

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"
GPUS="$2"
shift 2

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-${PORT:-28509}}"

if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ || ! "$NNODES" =~ ^[1-9][0-9]*$ ||
      ! "$NODE_RANK" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "GPUS_PER_NODE and NNODES must be positive integers; NODE_RANK must be nonnegative." >&2
    exit 2
fi
if (( NODE_RANK >= NNODES )); then
    echo "NODE_RANK must be smaller than NNODES." >&2
    exit 2
fi
if (( NNODES > 1 )) && [[ "$MASTER_ADDR" == "127.0.0.1" || "$MASTER_ADDR" == "localhost" ]]; then
    echo "Set MASTER_ADDR to the rank-0 host address for multi-node training." >&2
    exit 2
fi

# Resolve relative config, data, checkpoint and output paths from this checkout.
cd -- "$REPO_ROOT"
if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found (relative paths use $REPO_ROOT): $CONFIG" >&2
    exit 2
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export BEVFORMER_SELECTED_CONFIG="${BEVFORMER_SELECTED_CONFIG:-1}"

# Rank-0 NuScenes evaluation may outlast the default kernel/memcpy timeout
# while other ranks wait at the MCCL barrier. Allow an explicit override.
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3600000}"

echo "[launch] repo=$REPO_ROOT config=$CONFIG nodes=$NNODES rank=$NODE_RANK gpus_per_node=$GPUS master=$MASTER_ADDR:$MASTER_PORT"
torchrun \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    --nproc_per_node="$GPUS" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "$REPO_ROOT/tools/train.py" \
    "$CONFIG" \
    --launcher pytorch "$@"
