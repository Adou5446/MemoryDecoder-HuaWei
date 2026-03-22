# 1. 设置 Tokenizer 路径 (GPT-2 small)
TOKENIZER="/data/model/gpt2-small"

# 2. 设置原始数据集路径 (你之前下载的位置)
DATASET_PATH="/data/datasets/wikitext"

# 3. 设置预处理后的输出路径 (gpt2 tokenizer 单独一份，和 wikitext-qwen 区分开)
OUTPUT_DIR="/data/processed_data/wikitext-gpt2"

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 4. 执行 Python 预处理脚本
python utils/preprocess_dataset.py \
    --dataset_name ${DATASET_PATH} \
    --dataset_config_name "wikitext-103-v1" \
    --tokenizer_path ${TOKENIZER} \
    --output_dir ${OUTPUT_DIR} \
    --num_proc 32