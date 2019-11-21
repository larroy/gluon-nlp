# python -c 'import os; print(os.environ); import socket; print(socket.gethostname()); import mxnet as mx; import horovod.mxnet as hvd; print(mx); hvd.init(); print(hvd.rank())'
set -e 
set -x
#pkill python
export DATA_HOME=~/mxnet-data/bert-pretraining/datasets

export DEBUG="${DEBUG:-1}"
export HOST="${HOST:-hosts}"
export NP="${NP:-8}"
export CKPTDIR="${CKPTDIR:-./test-ckpt}"
#export OPTIMIZER="${OPTIMIZER:-lamb3}"
export OPTIMIZER="${OPTIMIZER:-lamb2}"
export COMPLETE_TRAIN="${COMPLETE_TRAIN:-0}"
export DATA="${DATA:-$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-$DATA_HOME/book-corpus/book-corpus-large-split/*.dev,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.dev}"
export NO_SHARD="${NO_SHARD:-0}"
export RAW="${RAW:-1}"
export EVALRAW="${EVALRAW:-0}"
export MXNET_SAFE_ACCUMULATION=1


# only used in a docker container
export USE_DOCKER=0
export OTHER_HOST=hosts_31
export DOCKER_IMAGE=haibinlin/worker_mxnet:c5fd6fc-1.5-cu90-79e6e8-79e6e8
export CLUSHUSER=ec2-user
export COMMIT=58435d04

export NCCLMINNRINGS=1
export TRUNCATE_NORM=1
export LAMB_BULK=60
export EPS_AFTER_SQRT=1
export NO_SHARD=0
export SKIP_GLOBAL_CLIP=1
export PT_DECAY=1
export SKIP_STATE_LOADING=1
export REPEAT_SAMPLER=1
export SCALE_NORM=1
export FORCE_WD=0
export USE_PROJ=0
export DTYPE=float16
export MODEL=bert_24_1024_16
export CKPTINTERVAL=300000000
export HIERARCHICAL=0
export EVALINTERVAL=100000000
export NO_DROPOUT=0
export USE_BOUND=0
export ADJUST_BOUND=0
export WINDOW_SIZE=2000

export OPTIONS='--verbose'
if [ "$DEBUG" = "1" ]; then
    export OPTIONS="$OPTIONS --synthetic_data"
    #export NUMSTEPS=5000000000
    export LOGINTERVAL=10
    export NUMSTEPS=200
else
    #export NUMSTEPS=7038
    export NUMSTEPS=200
    export LOGINTERVAL=10
fi
if [ "$RAW" = "1" ]; then
    export OPTIONS="$OPTIONS --raw"
fi
if [ "$EVALRAW" = "0" ]; then
    export OPTIONS="$OPTIONS --eval_use_npz"
fi
BS=64
#BS=32
ACC=1
MAX_SEQ_LENGTH=128
MAX_PREDICTIONS_PER_SEQ=20 
LR=0.0001
WARMUP_RATIO=0.2

	    #--num_data_workers 8 \

#python3 -m cProfile -s cumtime run_pretraining.py \
    python3 run_pretraining.py \
	    --data="$DATA_HOME/book-corpus/book-corpus-large-split/*.train,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.train" \
	    --data_eval="$DATA_HOME/book-corpus/book-corpus-large-split/*.test,$DATA_HOME/enwiki/enwiki-feb-doc-split/*.test" \
	    --optimizer $OPTIMIZER \
	    --warmup_ratio $WARMUP_RATIO \
	    --num_steps $NUMSTEPS \
	    --ckpt_interval $CKPTINTERVAL \
	    --dtype $DTYPE \
	    --ckpt_dir $CKPTDIR \
	    --lr $LR \
	    --total_batch_size $BS \
	    --total_batch_size_eval $BS \
	    --accumulate $ACC \
	    --model $MODEL \
	    --max_seq_length $MAX_SEQ_LENGTH \
	    --max_predictions_per_seq $MAX_PREDICTIONS_PER_SEQ \
	    --num_data_workers 1 \
        --eval_interval $EVALINTERVAL \
        --verbose \
	    --no_compute_acc --raw \
	    --comm_backend horovod --log_interval $LOGINTERVAL $OPTIONS 
