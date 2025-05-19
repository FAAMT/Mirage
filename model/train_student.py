#!/usr/bin/env python3
from pathlib import Path
import torch
from transformers import (
    MarianTokenizer, MarianMTModel,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)

# Paths
STUDENT_DIR = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/student_ckpt/sliced_init")
OUT_DIR = Path("/Users/faamt/Desktop/Nomadic Inc/Helsinki/student_distilled")
DATASET_PATH = Path("/path/to/your/saved/dataset")  # Update this path!

# Load existing components
print("Loading pre-trained student model...")
student = MarianMTModel.from_pretrained(STUDENT_DIR).to("cpu")
tok = MarianTokenizer.from_pretrained(STUDENT_DIR)

# Load preprocessed dataset
print("Loading preprocessed dataset...")
ds = Dataset.load_from_disk(DATASET_PATH)  # Load your saved dataset

# Fixed KDMarian class (removes decoder_input_ids issue)
class KDMarian(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.kl = torch.nn.KLDivLoss(reduction="batchmean")
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=tok.pad_token_id)

    def forward(self, input_ids, attention_mask, labels, teacher_logits):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        ce_loss = outputs.loss
        s_logp = torch.log_softmax(outputs.logits / T, dim=-1)
        t_prob = torch.softmax(teacher_logits / T, dim=-1)
        kl_loss = self.kl(s_logp, t_prob) * (T * T)
        return ALPHA * kl_loss + (1 - ALPHA) * ce_loss

# Training setup
student_kd = KDMarian(student)
collator = DataCollatorForSeq2Seq(tok, model=student)

args = TrainingArguments(
    output_dir="runs",
    per_device_train_batch_size=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    save_strategy="no",
    report_to="none",
    dataloader_num_workers=0,
)

trainer = Trainer(
    model=student_kd,
    args=args,
    train_dataset=ds,
    data_collator=collator,
)

# Start training
print("Resuming training...")
trainer.train()
student.save_pretrained(OUT_DIR)
print(f"✅ Training completed. Model saved to {OUT_DIR}")