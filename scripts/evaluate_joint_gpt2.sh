#!/bin/bash

# ========================================
# Joint Evaluation for GPT-2 small + MemDec (NPU)
# ========================================

# Model Configuration
MODEL_FAMILY="gpt2"
MODEL_SIZE="small"
DATASET_NAME="wikitext"

# Path Configuration
DATASET="/data/processed_data/wikitext-gpt2"
MODEL="/data/model/gpt2-small"

# KNN generator: last training epoch of GPT-2 small MemDec
KNN_PATH="./results/memdec-${MODEL_FAMILY}-${MODEL_SIZE}-${DATASET_NAME}/epoch_69"

OUTPUT_DIR="./results/tmp/gpt2-small-eval"

ACCELERATE_CONFIG="./accelerate_config/${MODEL_FAMILY}.yaml"

# NPU Environment Setup
export ACCELERATE_USE_ASCEND=true
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# HCCL Optimization
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_ROCE_HF=1
export HCCL_INTER_LINK_PORT=2
export HCCL_SOCKET_IFNAME=eth0
export HCCL_OP_BASE_ACK_TIMEOUT=300
export HCCL_NIC_POLICY=RDMA
export HCCL_DYNAMIC_OPTIMIZE=1

export PYTHONPATH=$PYTHONPATH:.
export PYTHONUNBUFFERED=1

mkdir -p ${OUTPUT_DIR}

echo "=========================================="
echo "Joint Evaluation: GPT-2 small + MemDec"
echo "=========================================="
echo "Base model: ${MODEL}"
echo "KNN generator: ${KNN_PATH}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"
echo "=========================================="

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    --num_processes 1 \
    -m \
    evaluate_joint \
    --do_test \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --dataset_split_name test \
    --per_device_eval_batch_size 4 \
    --output_dir ${OUTPUT_DIR} \
    --knn_temp 1.0 \
    --lmbda 0.80 \
    --knn_generator_path ${KNN_PATH} \
    --report_to none
