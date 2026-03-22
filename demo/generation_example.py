from memDec import MemoryDecoder

import transformers
from transformers import AutoModelForCausalLM
from loguru import logger

# Import NPU device management utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.npu_device import get_device

# Update model paths for your environment
base_lm_path = "model/Qwen2.5-7B"
knn_generator_path = "model/Qwen2.5-1.5B"

tokenizer = transformers.AutoTokenizer.from_pretrained(base_lm_path)
base_lm = AutoModelForCausalLM.from_pretrained(base_lm_path)
knn_generator = AutoModelForCausalLM.from_pretrained(knn_generator_path)

base_lm.resize_token_embeddings(len(tokenizer))
knn_generator.resize_token_embeddings(len(tokenizer))
base_lm.eval()
knn_generator.eval()

# Create joint model with NPU support
joint = MemoryDecoder(base_lm, knn_generator, lmbda=0.55, knn_temp=1.0)

# Get NPU device
device = get_device()
joint = joint.to(device)

prompt = f"As with previous Valkyira Chronicles games , Valkyria Chronicles III is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

out_ids = joint.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"Memory Decoder output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")

out_ids = base_lm.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=False
)
logger.info(f"Base Model output: {tokenizer.decode(out_ids[0], skip_special_tokens=True)}")