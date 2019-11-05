export DEBUG="${DEBUG:-1}"
export HOST="${HOST:-hosts_32}"
export NP="${NP:-8}"
export CKPTDIR="${CKPTDIR:-/fsx/test}"
export OPTIMIZER="${OPTIMIZER:-lamb2}"

export USE_DOCKER=0
export OTHER_HOST=hosts_31
export DOCKER_IMAGE=haibinlin/worker_mxnet:c5fd6fc-1.5-cu90-79e6e8-79e6e8
export NCCLMINNRINGS=1
export TRUNCATE_NORM=1
export LAMB_BULK=60
export EPS_AFTER_SQRT=1
export NO_SHARD=0
export FORCE_WD=0
export USE_PROJ=0
export DTYPE=float16
export MODEL=bert_24_1024_16
export DATA_HOME=/data
export DATA_HOME=~/mxnet-data/bert-pretraining/datasets
export CKPTINTERVAL=300000000
export CLUSHUSER=ec2-user
export HIERARCHICAL=0
export EVALINTERVAL=100000000
export COMMIT=58435d04
export NO_DROPOUT=0
export USE_BOUND=0
export ADJUST_BOUND=0
export WINDOW_SIZE=2000

if [ "$USE_DOCKER" = "1" ]; then
    export PORT=12451
    bash clush-hvd.sh
else
    export PORT=22
fi

sleep 5
if [ "$DEBUG" = "1" ]; then
    export LOGINTERVAL=1
    export NUMSTEPS=5
    export OPTIONS='--synthetic_data --verbose --eval_use_npz'
    export NUMSTEPS=5000000000
    export LOGINTERVAL=5
else
    export LOGINTERVAL=50
    #export NUMSTEPS=7038
    export NUMSTEPS=14063
    #export OPTIONS='--start_step $NUMSTEPS'
fi
#BS=65536 ACC=4 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.006 WARMUP_RATIO=0.2843 bash mul-hvd.sh
#BS=32768 ACC=2 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
if [ "$NP" = "1" ]; then
    BS=64 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.0001 WARMUP_RATIO=0.2 bash mul-hvd.sh
elif [ "$NP" = "8" ]; then
    BS=512 ACC=1 MAX_SEQ_LENGTH=128 MAX_PREDICTIONS_PER_SEQ=20 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh
fi

if [ "$USE_DOCKER" = "1" ]; then
    export PORT=12452
    bash clush-hvd.sh
else
    export PORT=22
fi

sleep 5
if [ "$DEBUG" = "1" ]; then
    export LOGINTERVAL=1
    export OPTIONS="--synthetic_data --verbose --phase2 --phase1_num_steps=$NUMSTEPS --start_step=$NUMSTEPS"
    export NUMSTEPS=3
else
    export LOGINTERVAL=50
    export OPTIONS="--phase2 --phase1_num_steps=$NUMSTEPS --start_step=$NUMSTEPS"
    export NUMSTEPS=1563
fi
BS=32768 ACC=16 MAX_SEQ_LENGTH=512 MAX_PREDICTIONS_PER_SEQ=80 LR=0.005 WARMUP_RATIO=0.2 bash mul-hvd.sh

STEP_FORMATTED=$(printf "%07d" $NUMSTEPS)
python3 finetune_squad.py --bert_model bert_24_1024_16 --pretrained_bert_parameters $CKPTDIR/$STEP_FORMATTED.params --output_dir $CKPTDIR --optimizer adam --accumulate 3 --batch_size 8 --lr 3e-5 --epochs 2 --gpu 0
