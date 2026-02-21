#!/usr/bin/env bash
export PROJ_ROOT=$(pwd)
export EXP_NAME="eval-memocr_train_qwen2d5_7b_vl_triple"
export MODEL_ROOT="$PROJ_ROOT/results/memocr/memocr_train_qwen2d5_7b_vl_triple/global_step_1000/actor"
export MODEL_TP=4

cd $PROJ_ROOT

export PATH=$PATH:~/.local/bin
pip3 install --upgrade pip
pip3 install --upgrade ray[serve,default]
pip3 install --upgrade transformers
pip3 install pathlib hydra-core wandb codetiming torch_memory_saver beautifulsoup4

export DATAROOT="$PROJ_ROOT/data/test"
export PYTHONPATH=$PROJ_ROOT:$PYTHONPATH

export MODEL_NAME="$MODEL_ROOT/hf_ckpt"
if [ ! -d "$MODEL_NAME" ] || [ ! -f "$MODEL_NAME/vocab.json" ]; then
    bash $PROJ_ROOT/scripts/merge_ckpt.sh $MODEL_ROOT
    echo "Merged checkpoint to $MODEL_NAME successfully"
else
    echo "$MODEL_NAME already exist"
fi

mkdir -p $PROJ_ROOT/log/eval

export VANILLA_EXP_NAME=$EXP_NAME
for mem_budget in 1024 256 64 16 512 128 32 8 4 2 1; do
    export MAX_PIXELS_28_28=$mem_budget
    export EXP_NAME=${VANILLA_EXP_NAME}_MemBudget${mem_budget}
    python3 taskutils/memory_eval/run_custom.py 2>&1 | tee $PROJ_ROOT/log/eval/$EXP_NAME.log
done