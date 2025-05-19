#!/usr/bin/env python3
# -----------------------------------------------------------------------
#  Optimized distillation script with FPGA-compatible dimensions
# -----------------------------------------------------------------------
from pathlib import Path
import torch
import time
import numpy as np
import gc
import os
import atexit
import tempfile
import shutil
import signal
import sys

from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import (
    MarianTokenizer, MarianMTModel,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments,
    MarianConfig
)

# --- Initialize temp directory with proper cleanup ---
temp_dir = tempfile.mkdtemp()
print(f"Using temp directory: {temp_dir}")

def cleanup():
    """Robust cleanup function for temp files"""
    if os.path.exists(temp_dir):
        print("\nPerforming cleanup...")
        try:
            shutil.rmtree(temp_dir)
            print(f"Deleted temp directory: {temp_dir}")
        except Exception as e:
            print(f"Cleanup error: {e}")

# Register cleanup handlers
atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda *_: (cleanup(), sys.exit(1)))

# ---------- paths ------------------------------------------------------
DATA_PARQUET = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/data/jw300/jw300.en-ar.parquet")
TEACHER_DIR = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/opus-mt-en-ar")
STUDENT_DIR = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/student_ckpt/sliced_init")
OUT_DIR = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/student_distilled")
DATASET_SAVE_PATH = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/processed_dataset")

# ---------- FPGA-compatible hyper-params -------------------------------
BATCH_SIZE_PREPROCESS =16
MODEL_MAX_LENGTH = 4  # Can be increased to 64 if needed
BATCH_SIZE_TRAIN =16
LOGIT_COLS = 128
EPOCHS = 2
LR = 2e-4
T = 2.0
ALPHA = 0.5
NUM_EXAMPLES = 50_000

# FPGA-optimized dimensions
D_MODEL = 256
FFN_DIM = 128
QKV_DIM = 64  # Critical for FPGA compatibility

device = torch.device("cpu")
print("Running on:", device)

# ---------- Create FPGA-compatible student config ----------------------
print("Creating FPGA-compatible student config...")
config = MarianConfig.from_pretrained(TEACHER_DIR)
config.update({
    'd_model': D_MODEL,
    'encoder_ffn_dim': FFN_DIM,
    'decoder_ffn_dim': FFN_DIM,
    'encoder_attention_heads': 4,  # D_MODEL (256) / head_dim (64) = 4 heads
    'decoder_attention_heads': 4,
    'attention_head_dim': QKV_DIM  # Critical for FPGA compatibility
})

# ---------- tokenizer --------------------------------------------------
tok = MarianTokenizer.from_pretrained(TEACHER_DIR)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---------- load examples stream ---------------------------------------
print(f"Loading {NUM_EXAMPLES} examples...")
ds_stream = (
    load_dataset("parquet",
                data_files={"train": str(DATA_PARQUET)},
                split="train",
                streaming=True)
    .shuffle(buffer_size=10_000, seed=42)
    .take(NUM_EXAMPLES)
)

# ---------- teacher model ----------------------------------------------
print("Loading teacher model...")
teacher = MarianMTModel.from_pretrained(TEACHER_DIR).to(device).eval()
for p in teacher.parameters():
    p.requires_grad_(False)

# ---------- optimized preprocessing ------------------------------------
def preprocess(batch_data):
    start_time = time.time()
    
    src = tok(batch_data["english"], 
             truncation=True, 
             padding="max_length", 
             max_length=MODEL_MAX_LENGTH, 
             return_tensors="pt").to(device)
    tgt = tok(batch_data["non_english"], 
             truncation=True, 
             padding="max_length", 
             max_length=MODEL_MAX_LENGTH, 
             return_tensors="pt").to(device)
    
    with torch.inference_mode():
        logits = teacher(
            input_ids=src.input_ids,
            attention_mask=src.attention_mask,
            decoder_input_ids=tgt.input_ids
        ).logits[:, :, :LOGIT_COLS]
    
    if current_idx % 1000 == 0:
        print(f"Batch processed in {time.time()-start_time:.2f}s")
    
    return {
        "input_ids": src.input_ids.cpu().numpy(),
        "attention_mask": src.attention_mask.cpu().numpy(),
        "labels": tgt.input_ids.cpu().numpy(),
        "teacher_logits": logits.cpu().numpy(),
    }

    # Load model and check quantization
def create_fpga_student(original_dir, output_dir):
    """Create FPGA-compatible Marian student model"""
    # 1. Load original config and modify
    config = MarianConfig.from_pretrained(original_dir)
    config.update({
        'd_model': 256,
        'encoder_ffn_dim': 128,
        'decoder_ffn_dim': 128,
        'encoder_layers': 6,  # Match original
        'decoder_layers': 6,
        'encoder_attention_heads': 4,  # 256/64 = 4
        'decoder_attention_heads': 4
    })
    
    # 2. Create new model
    fpga_model = MarianMTModel(config)
    
    # 3. Marian-specific weight adaptation
    orig_model = MarianMTModel.from_pretrained(original_dir)
    state_dict = orig_model.state_dict()
    
    # Adapt encoder layers
    for i in range(config.encoder_layers):
        # QKV projections (original: 256x256 → new: 256x64)
        prefix = f"model.encoder.layers.{i}."
        for proj in ['q_proj', 'k_proj', 'v_proj']:
            weight = state_dict[f"{prefix}self_attn.{proj}.weight"]
            fpga_model.model.encoder.layers[i].self_attn.__dict__[proj].weight.data = weight[:,:64]
            
        # FFN layers
        fpga_model.model.encoder.layers[i].fc1.weight.data = state_dict[f"{prefix}fc1.weight"][:128,:64]
        fpga_model.model.encoder.layers[i].fc2.weight.data = state_dict[f"{prefix}fc2.weight"][:256,:128]
    
    # 4. Save FPGA-compatible model
    fpga_model.save_pretrained(output_dir)
    return output_dir


# -------- Materialize dataset with memory-mapped arrays --------
try:
    print("Initializing memory-mapped arrays...")
    memmap_shapes = {
        "input_ids": (NUM_EXAMPLES, MODEL_MAX_LENGTH),
        "attention_mask": (NUM_EXAMPLES, MODEL_MAX_LENGTH),
        "labels": (NUM_EXAMPLES, MODEL_MAX_LENGTH),
        "teacher_logits": (NUM_EXAMPLES, MODEL_MAX_LENGTH, LOGIT_COLS)
    }
    
    memmaps = {}
    for name, shape in memmap_shapes.items():
        dtype = np.int32 if name != "teacher_logits" else np.float32
        memmaps[name] = np.memmap(
            os.path.join(temp_dir, f"{name}.npy"),
            dtype=dtype,
            mode="w+",
            shape=shape
        )
    
    print("Processing dataset...")
    current_idx = 0
    pbar = tqdm(total=NUM_EXAMPLES, desc="Processing examples")
    
    for batch in ds_stream.batch(batch_size=BATCH_SIZE_PREPROCESS):
        if current_idx >= NUM_EXAMPLES:
            break
            
        processed = preprocess(batch)
        batch_size = len(batch["english"])
        
        for name, data in processed.items():
            memmaps[name][current_idx:current_idx+batch_size] = data
        
        current_idx += batch_size
        pbar.update(batch_size)
        
        if current_idx % 1000 == 0:
            gc.collect()
    
    pbar.close()
    
    # Create final Dataset
    print("Creating final Dataset...")
    ds = Dataset.from_dict({name: np.array(memmap) for name, memmap in memmaps.items()})
    print(f"Dataset created with {len(ds)} examples")
    print(f"Saving dataset to {DATASET_SAVE_PATH}...")
    ds.save_to_disk(DATASET_SAVE_PATH)
    
    # Clean up teacher model
    print("Deleting teacher model...")
    del teacher
    gc.collect()

except Exception as e:
    print(f"Error occurred: {e}")
    cleanup()
    sys.exit(1)