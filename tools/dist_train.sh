#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-28509}

# Default MUSA kernel/memcpy timeout is 200000ms (~3.3min). Rank0 NuScenes
# metric after eval can exceed that while other ranks sit in MCCL barrier;
# raise so the post-eval dist.barrier() can wait. Override: MUSA_EXECUTION_TIMEOUT=...
export MUSA_EXECUTION_TIMEOUT="${MUSA_EXECUTION_TIMEOUT:-3600000}"

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
torchrun \
    --nproc_per_node=$GPUS \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}


