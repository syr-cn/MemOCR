#!/usr/bin/env bash
set -euxo pipefail

ulimit -n 65535
# export NCCL_DEBUG=INFO

#################### 参数区 ####################
EXP_LOG_NAME="memocr_train_qwen2d5_7b_vl_triple"
# 训练代码会根据实验名称中的一些字段来切换设定
## triple: 使用multiple training objectives
## vl: 使用VL模型

N_NODE=8
N_GPU=8

# Training Setting
MAXLEN=12288
MAX_NEW_TOKEN=2048
RECURRENT_MAXLEN=2048
LR=1.0e-6
ROLLOUT_N=16
ROLLOUT_VAL_N=4

TRAIN_BS=64
PPO_MINI_BS=16
METRIC_NAME=em
export MAX_PIXELS_28_28=512
export MIN_PIXELS_28_28=8

export SUBSAMPLED_QA_REWARD_WEIGHT=0.7
export GAP_FILL_REWARD_WEIGHT=0.3

export RENDER_SERVER_IP="localhost"
export RENDER_SERVER_PORT=9000

export PROJ_ROOT=$(pwd)
STORAGE_ROOT=$PROJ_ROOT
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"
DATASET_ROOT="${STORAGE_ROOT}/data/hotpotqa"

export EXP=memocr/$EXP_LOG_NAME
export PROJ_DIR="$STORAGE_ROOT/results/${EXP}"
export LOG_PATH="$PROJ_ROOT/log/$EXP_LOG_NAME.log"
export ROLLOUT_DIR="${PROJ_DIR}/log/rollout_trajectory/${EXP_LOG_NAME}/"
export VAL_PATH="${DATASET_ROOT}/hotpotqa_dev.parquet"
export TRAIN_PATH="${DATASET_ROOT}/hotpotqa_train_32k.parquet"

cd $PROJ_ROOT
export PATH=$PATH:~/.local/bin

pip3 install --upgrade pip
pip3 install --upgrade ray[serve,default]
pip3 install --upgrade transformers
pip3 install pathlib hydra-core wandb codetiming torch_memory_saver beautifulsoup4

wandb online

export PYTHONPATH=$PYTHONPATH:$PROJ_ROOT
# Wandb
export WANDB_API_KEY="YOUR_WANDB_TOKEN"
export WANDB_PROJECT="YOUR_WANDB_PROJECT"

mkdir -p $PROJ_DIR
mkdir -p $PROJ_DIR/log/
mkdir -p $ROLLOUT_DIR

export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

# 检查模型路径
if [ -d "$MODEL_PATH" ]; then
    echo "[INFO] MODEL_PATH exists: $MODEL_PATH"
    echo "[INFO] Contents of MODEL_PATH:"
    ls -lh "$MODEL_PATH"
else
    echo "[ERROR] MODEL_PATH does not exist: $MODEL_PATH"
    exit 1
fi

export PYTHONUNBUFFERED=1
export TORCH_NCCL_AVOID_RECORD_STREAMS="1"


RAY_HEAD_ADDR="localhost"
RAY_HEAD_PORT=8278

export RAY_ADDRESS="${RAY_HEAD_ADDR}:${RAY_HEAD_PORT}"
echo "🎯 使用 Ray Cluster: $RAY_ADDRESS"

ray status --address="$RAY_ADDRESS" || {
  echo "[ERROR] 无法连接 Ray cluster: $RAY_ADDRESS"
  echo "请先在 head 节点启动: ray start --head --port=$RAY_HEAD_PORT ..."
  echo "并在 worker 节点启动: ray start --address=${RAY_HEAD_ADDR}:$RAY_HEAD_PORT"
  exit 1
}

python3 -m verl.trainer.main_ppo \
    recurrent.enable=memory \
    recurrent.memory.config.chunk_size=5000 \
    recurrent.memory.config.max_final_response_length=$RECURRENT_MAXLEN \
    recurrent.memory.config.max_memorization_length=$RECURRENT_MAXLEN \
    recurrent.memory.config.max_prompt_length=1024 \
    recurrent.memory.path="recurrent/impls/memory_img_final_only_triple.py" \
    +actor_rollout_ref.processor.use_fast=False \
    algorithm.adv_estimator=grpo \
    +algorithm.subsampled_qa_reward_weight=$SUBSAMPLED_QA_REWARD_WEIGHT \
    +algorithm.gap_fill_reward_weight=$GAP_FILL_REWARD_WEIGHT \
    algorithm.norm_adv_by_std_in_grpo=False \
    trainer.save_freq=20 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.val_kwargs.n=$ROLLOUT_VAL_N \
    trainer.logger=['console','wandb'] \
    actor_rollout_ref.actor.optim.lr_warmup_steps=20 \
    actor_rollout_ref.actor.clip_ratio_high=0.20 \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$VAL_PATH \
    data.shuffle=False \
    data.filter_overlong_prompts=True \
    data.train_batch_size=$TRAIN_BS \
    data.truncation='middle' \
    +data.context_key='context' \
    data.max_prompt_length=$MAXLEN \
    data.max_response_length=$MAX_NEW_TOKEN \
    reward_model.reward_manager='thread' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BS \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=0.999 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.ref.fsdp_config.param_offload=false \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.project_name=$WANDB_PROJECT \
    trainer.experiment_name=$EXP_LOG_NAME \
    trainer.val_before_train=false \
    trainer.n_gpus_per_node=$N_GPU \
    trainer.nnodes=$N_NODE \
    trainer.test_freq=10 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=$PROJ_DIR \
    trainer.total_epochs=30 \
    2>&1 | tee $LOG_PATH
