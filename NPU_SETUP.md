# MemoryDecoder NPU 环境配置指南

## 环境要求

### 硬件要求
- 华为 910B NPU 卡（8张）
- 足够的内存（建议 256GB+）

### 软件要求
- Ubuntu 18.04/20.04/22.04
- Python 3.8+
- CANN 6.0+
- PyTorch 2.0+ (适配 NPU 版本)

## 安装步骤

### 1. 安装华为 CANN 工具包

参考华为官方文档安装 CANN 工具包：
```bash
# 下载并安装 CANN 工具包
# 版本建议：CANN 6.0+
```

### 2. 安装适配 NPU 的 PyTorch

```bash
# 华为官方提供的适配 NPU 的 PyTorch
pip install torch==2.0.0+ascend torchvision==0.11.0+ascend -f https://ascend-pytorch.obs.cn-east-2.myhuaweicloud.com/ascend/torch-2.0.0.html
```

### 3. 安装项目依赖

```bash
# 安装基础依赖
pip install transformers==4.55.4 datasets==4.0.0 accelerate

# 安装 FAISS CPU 版本（NPU 不支持 FAISS-GPU）
pip install faiss-cpu==1.12.0

# 安装其他依赖
pip install loguru wandb tqdm pickle pyarrow
```

### 4. 配置环境变量

```bash
# 设置 CANN 环境变量
export ASCEND_HOME=/usr/local/Ascend
export PATH=$ASCEND_HOME/bin:$ASCEND_HOME/compiler/ccec_compiler/bin:$PATH
export LD_LIBRARY_PATH=$ASCEND_HOME/lib64:$LD_LIBRARY_PATH

# 设置 PyTorch 环境变量
export TORCH_HOME=/path/to/torch/home

# 设置 NPU 可见设备（8张卡）
export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

## 模型配置

### 模型路径
- **基础模型**: `model/Qwen2.5-7B`
- **KNN 生成器**: `model/Qwen2.5-1.5B`

确保这些模型已经下载到指定的相对路径。

## 运行配置

### 1. 数据预处理
```bash
bash scripts/preprocess_dataset.sh
```

### 2. 构建 KNN 索引
```bash
python -m knn_utils.build_index \
    --dstore_path /path/to/dstore \
    --num_keys_to_add_at_a_time 1000000 \
    --ncentroids 4096 \
    --code_size 32 \
    --probe 8
```

### 3. 评估基础模型
```bash
bash scripts/evaluate_base_gpt.sh
```

### 4. 评估联合模型
```bash
bash scripts/evaluate_joint_gpt2.sh
```

### 5. 生成示例
```bash
cd demo
python generation_example.py
```

## 训练配置

### 1. 基础模型训练
```bash
# 使用 accelerate 进行多卡训练
accelerate launch --config_file accelerate_config/qwen2.yaml -m train_base \
    --model_name_or_path model/Qwen2.5-7B \
    --dataset_name /path/to/dataset \
    --do_train \
    --do_eval \
    --eval_subset validation \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --output_dir ./output/train_base \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --logging_dir ./logs/train_base \
    --logging_steps 100 \
    --report_to none
```

### 2. 构建 KNN 索引
```bash
python -m knn_utils.build_index \
    --dstore_path ./dstore/qwen2-7B/wikitext/dstore_qwen2_train_4096.arrow \
    --num_keys_to_add_at_a_time 1000000 \
    --ncentroids 4096 \
    --code_size 64 \
    --probe 32
```

### 3. 保存 KNN 结果
```bash
accelerate launch --config_file accelerate_config/qwen2.yaml -m knn_utils.saveKNNMulti \
    --model_path model/Qwen2.5-7B \
    --dstore_path ./dstore/qwen2-7B/wikitext/dstore_qwen2_train_4096.arrow \
    --val_path ./dstore/qwen2-7B/wikitext/train_vals.pkl \
    --index_path ./dstore/qwen2-7B/wikitext/train_4096.index \
    --output_path ./dstore/qwen2-7B/wikitext/knn_qwen2_train_4096.arrow \
    --k 1024 \
    --knn_temp 16.0 \
    --probe 32 \
    --batch_size 16000 \
    --ignore_first True
```

### 4. MemoryDecoder 训练
```bash
accelerate launch --config_file accelerate_config/qwen2.yaml -m train_memdec \
    --model_name_or_path model/Qwen2.5-1.5B \
    --dataset_name /path/to/dataset \
    --do_train \
    --do_eval \
    --eval_subset validation \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --output_dir ./output/train_memdec \
    --save_strategy steps \
    --save_steps 1000 \
    --evaluation_strategy steps \
    --eval_steps 500 \
    --logging_dir ./logs/train_memdec \
    --logging_steps 100 \
    --report_to none
```

## 多卡并行配置

### Accelerate 配置
项目使用 `accelerate` 进行多卡并行训练和推理。确保配置文件正确：

```yaml
# accelerate_config/qwen2.yaml
compute_environment: LOCAL_MACHINE
distributed_type: NO
use_cpu: False
num_processes: 8
gpu_ids: all
```

### 运行多卡命令
```bash
# 使用 8 张 NPU 卡
accelerate launch --config_file accelerate_config/qwen2.yaml -m train_memdec \
    --model_name_or_path model/Qwen2.5-1.5B \
    --dataset_name /path/to/dataset \
    --per_device_train_batch_size 8 \
    --num_train_epochs 3 \
    --output_dir ./output
```

## 性能优化建议

### 1. 批次大小调整
- 根据内存情况调整 `per_device_batch_size`
- NPU 内存通常比 GPU 小，建议从小批次开始测试

### 2. FAISS 索引优化
- 使用 CPU 版本的 FAISS，速度会比 GPU 版本慢
- 可以调整 `ncentroids` 和 `probe` 参数来平衡速度和精度

### 3. 内存管理
- 监控 NPU 内存使用情况
- 定期清理缓存：`torch.npu.empty_cache()`

## 常见问题

### Q1: 模型加载失败
**A**: 检查模型路径是否正确，确保模型文件完整。

### Q2: NPU 内存不足
**A**: 减小批次大小，或者使用梯度累积。

### Q3: FAISS 搜索速度慢
**A**: 这是正常的，因为使用 CPU 版本的 FAISS。可以尝试调整索引参数。

### Q4: 多卡并行不工作
**A**: 检查 accelerate 配置文件，确保 `num_processes` 设置为 8。

## 验证步骤

1. **检查 NPU 可用性**:
   ```python
   import torch
   print(f"NPU available: {torch.npu.is_available()}")
   print(f"NPU count: {torch.npu.device_count()}")
   ```

2. **测试模型加载**:
   ```python
   from transformers import AutoModelForCausalLM
   model = AutoModelForCausalLM.from_pretrained("model/Qwen2.5-7B")
   print("Model loaded successfully")
   ```

3. **运行完整评估**:
   ```bash
   bash scripts/evaluate_joint_gpt2.sh
   ```

## 技术支持

如有问题，请参考：
- 华为昇腾社区：https://www.hiascend.com/
- CANN 文档：https://www.hiascend.com/document
- PyTorch NPU 文档：https://pytorch.org/