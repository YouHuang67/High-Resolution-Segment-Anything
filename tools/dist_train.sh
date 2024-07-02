#!/bin/bash

if [[ $1 =~ ^[0-9]+$ ]]; then
    PORT=$1
    CONFIG=$2
    GPUS=$3
    shift 3
else
    PORT=29500
    CONFIG=$1
    GPUS=$2
    shift 2
fi

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch "$@"
