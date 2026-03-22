# Update paths for your environment
DATASET="/data/processed_data/wikitext-qwen"
MODEL=model/Qwen2.5-7B
OUTPUT_DIR=tmp/

# Use NPU environment instead of CUDA
python \
    -m \
    train_base \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET} \
    --per_device_eval_batch_size 16 \
    --do_eval \
    --eval_subset test \
    --output_dir ${OUTPUT_DIR} \
    --report_to none