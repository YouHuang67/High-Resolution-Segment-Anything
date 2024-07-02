#!/bin/bash
SCRIPT=$1
shift
CONFIG=$1
shift
GPUS=$1
shift
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

if [[ $1 != --* ]]; then
    PORT=$1
    shift
else
    PORT=29500
fi

PYTHONPATH="$(dirname "$0")/../../":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $SCRIPT \
    $CONFIG \
    --launcher pytorch "$@"
