from transformers import MarianMTModel, MarianTokenizer
import torch
import numpy as np

# Load model/tokenizer
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ar")
model.eval()

# Encode sentence
sentence = "My name is Fahad"
inputs = tokenizer(sentence, return_tensors="pt")

# Store embedding input to encoder layer 0
captured_input = {}

def hook_fn(module, input, output):
    captured_input['with_pos'] = input[0].detach().cpu().numpy()  # (1, seq, 512)

# Hook into first encoder block
hook = model.model.encoder.layers[0].register_forward_hook(hook_fn)

# Run encoder to trigger the hook
with torch.no_grad():
    _ = model.model.encoder(input_ids=inputs["input_ids"],
                            attention_mask=inputs["attention_mask"])

# Remove hook
hook.remove()

# Get token-only embeddings
tok_only = model.model.shared(inputs['input_ids']).detach().cpu().numpy()  # (1, seq, 512)

# True positional encoding = difference
true_pos_enc = captured_input['with_pos'] - tok_only

print("True PE shape:", true_pos_enc.shape)
print("First token pos enc:", true_pos_enc[0, 0, :16])
