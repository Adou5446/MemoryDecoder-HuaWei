#!/bin/bash

# ========================================
# Memory Decoder Training Data Preparation Pipeline Configuration for NPU
# ========================================

# 重新加载华为环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# NPU distributed training optimization
# Only keep necessary variables for performance
export HCCL_INTRA_ROCE_ENABLE=1
export HCCL_INTRA_ROCE_HF=1
export HCCL_INTER_LINK_PORT=2
export HCCL_SOCKET_IFNAME=eth0
export HCCL_OP_BASE_ACK_TIMEOUT=300
export HCCL_NIC_POLICY=RDMA
export HCCL_DYNAMIC_OPTIMIZE=1

# Model Configuration
# Options: gpt2, qwen2, qwen2.5
MODEL_FAMILY="gpt2"
MODEL_SIZE="xl"

# Auto-detect dimension based on model family
if [ "$MODEL_FAMILY" == "gpt2" ]; then
    case "$MODEL_SIZE" in
        "small") DIMENSION=768 ;;
        "medium") DIMENSION=1024 ;;
        "large") DIMENSION=1280 ;;
        "xl") DIMENSION=1600 ;;
        *) DIMENSION=768 ;;
    esac
elif [ "$MODEL_FAMILY" == "qwen2" ]; then
    case "$MODEL_SIZE" in
        "0.5B") DIMENSION=896 ;;
        "1.8B") DIMENSION=2048 ;;
        "7B") DIMENSION=4096 ;;
        "14B") DIMENSION=5120 ;;
        "72B") DIMENSION=8192 ;;
        *) DIMENSION=4096 ;;
    esac
elif [ "$MODEL_FAMILY" == "qwen2.5" ]; then
    case "$MODEL_SIZE" in
        "0.5B") DIMENSION=896 ;;
        "1.5B") DIMENSION=1536 ;;
        "3B") DIMENSION=2048 ;;
        "7B") DIMENSION=3584 ;;
        "14B") DIMENSION=5120 ;;
        "32B") DIMENSION=6144 ;;
        "72B") DIMENSION=8192 ;;
        *) DIMENSION=3584 ;;
    esac
else
    echo "Unknown model family: $MODEL_FAMILY"
    exit 1
fi

# Dataset Configuration
DATASET_NAME="wikitext"
SUBSET="train"

# Path Configuration
DATASET="/data/processed_data/wikitext-gpt2"
ACCELERATE_CONFIG="./accelerate_config/${MODEL_FAMILY}.yaml"
DSTORE_DIR="./dstore/${MODEL_FAMILY}-${MODEL_SIZE}/${DATASET_NAME}"
OUTPUT_DIR="./results/tmp/${MODEL_FAMILY}-${MODEL_SIZE}-${DATASET_NAME}-ppl"
# Update model path for your environment
MODEL_TO_SAVE="/data/model/gpt2-xl"

# Milvus Configuration
MILVUS_URI="http://10.140.158.149:8766"
COLLECTION_NAME="memdec_${MODEL_FAMILY}_${SUBSET}_${DIMENSION}"

# Training Configuration
# Change batch size based on the memory of your NPU
BATCH_SIZE_EVAL=1
BATCH_SIZE_KNN=1000   # Per-process batch size for Milvus search queries

# KNN Configuration
K=1024
KNN_TEMP=16.0
PROBE=32
NLIST=4096
PQ_M=64    # Must divide DIMENSION: 768/64=12 ✓  3584/64=56 ✓
NBITS=8

# Derived paths
DSTORE_PATH="${DSTORE_DIR}/dstore_${MODEL_FAMILY}_${SUBSET}_${DIMENSION}_rank0.arrow"
VAL_PATH="${DSTORE_DIR}/${SUBSET}_vals.pkl"
INDEX_PATH="${DSTORE_DIR}/${SUBSET}_${DIMENSION}.index"
OUTPUT_PATH="${DSTORE_DIR}/knn_${MODEL_FAMILY}_${SUBSET}_${DIMENSION}.arrow"

# ========================================
# Pipeline Execution
# ========================================

echo "=========================================="
echo "NeuralKNN Pipeline"
echo "=========================================="
echo "Model: ${MODEL_FAMILY}-${MODEL_SIZE}"
echo "Dataset: ${DATASET_NAME}"
echo "Subset: ${SUBSET}"
echo "=========================================="
echo ""

# Step 1: Generate Datastore
echo "[Step 1/3] Generating datastore..."
echo "Output directory: ${OUTPUT_DIR}"
echo "Datastore directory: ${DSTORE_DIR}"
echo ""

WANDB_PROJECT="neuralKNN" accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m \
    train_base \
    --model_name_or_path ${MODEL_TO_SAVE} \
    --dataset_name ${DATASET} \
    --do_eval --eval_subset ${SUBSET} \
    --per_device_eval_batch_size ${BATCH_SIZE_EVAL} \
    --output_dir ${OUTPUT_DIR} \
    --dstore_dir ${DSTORE_DIR} \
    --save_knnlm_dstore \
    --report_to none

if [ $? -ne 0 ]; then
    echo "Error: Datastore generation failed!"
    exit 1
fi

echo ""
echo "[Step 1/3] ✓ Datastore generation completed"
echo ""

# Step 2: Build FAISS index
echo "[Step 2/3] Building FAISS IVF_PQ index..."
echo "ncentroids=${NLIST}, code_size=64, probe=${PROBE}"
echo ""

python -m knn_utils.build_index \
    --dstore_path ${DSTORE_PATH} \
    --ncentroids ${NLIST} \
    --code_size 64 \
    --probe ${PROBE}

if [ $? -ne 0 ]; then
    echo "Error: FAISS index building failed!"
    exit 1
fi

echo ""
echo "[Step 2/3] ✓ FAISS index building completed"
echo ""

# Step 3: Save KNN Results via FAISS
echo "[Step 3/3] Saving KNN results via FAISS..."
echo "K neighbors: ${K}"
echo "KNN temperature: ${KNN_TEMP}"
echo "Output path: ${OUTPUT_PATH}"
echo ""

accelerate launch \
    --config_file ${ACCELERATE_CONFIG} \
    -m knn_utils.saveKNNMulti \
    --dstore_path ${DSTORE_PATH} \
    --index_path ${INDEX_PATH} \
    --output_path ${OUTPUT_PATH} \
    --model_path ${MODEL_TO_SAVE} \
    --k ${K} \
    --knn_temp ${KNN_TEMP} \
    --probe ${PROBE} \
    --batch_size 1000

if [ $? -ne 0 ]; then
    echo "Error: KNN saving failed!"
    exit 1
fi

echo ""
echo "[Step 3/3] ✓ KNN results saved"
echo ""

echo "=========================================="
echo "Pipeline completed successfully!"
echo "=========================================="
echo "Final outputs:"
echo "  - Datastore: ${DSTORE_PATH}"
echo "  - Index: ${INDEX_PATH}"
echo "  - KNN results: ${OUTPUT_PATH}"
echo "=========================================="