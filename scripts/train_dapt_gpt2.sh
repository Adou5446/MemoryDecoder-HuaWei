#!/bin/bash

# DAPT: Domain-Adaptive Pre-Training on wikitext for GPT-2 small
# Baseline comparison against Memory Decoder

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_ROCE_HF=1
export HCCL_INTER_LINK_PORT=2
export HCCL_SOCKET_IFNAME=eth0
export HCCL_OP_BASE_ACK_TIMEOUT=300
export HCCL_NIC_POLICY=RDMA
export HCCL_DYNAMIC_OPTIMIZE=1

MODEL="/data/model/gpt2-small"
DATASET="/data/processed_data/wikitext-gpt2"
OUTPUT_DIR="./results/dapt-gpt2-small-wikitext"
ACCELERATE_CONFIG="./accelerate_config/gpt2.yaml"

WANDB_PROJECT="MemoryDecoder" WANDB_RUN_NAME="dapt-gpt2-small-wikitext" \
/root/anaconda3/envs/memtest/bin/accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m train_base \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --do_train \
    --do_eval \
    --eval_subset test \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 10 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.05 \
    --weight_decay 0.01 \
    --lr_scheduler_type cosine \
    --save_strategy epoch \
    --eval_strategy epoch \
    --logging_steps 5 \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir \
    --report_to wandb
