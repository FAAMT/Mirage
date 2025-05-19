import onnxruntime as ort
import numpy as np
from transformers import MarianTokenizer
from pathlib import Path

class FPGATranslator:
    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self.sessions = self._load_sessions()
        self.tokenizer = MarianTokenizer.from_pretrained(model_dir)
        
    def _load_sessions(self):
        return {
            "encoder": ort.InferenceSession(str(self.model_dir / "encoder_model.int8.onnx")),
            "decoder": ort.InferenceSession(str(self.model_dir / "decoder_model.int8.onnx"))
        }
    
    def translate(self, text, max_length=16):
        # Tokenize with FPGA-compatible length
        inputs = self.tokenizer(
            text, 
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        
        # FPGA-optimized inference
        encoder_outputs = self.sessions["encoder"].run(None, {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        })
        
        # Decode with FPGA constraints
        decoder_input = np.array([[self.tokenizer.pad_token_id]])
        for _ in range(max_length):
            decoder_outputs = self.sessions["decoder"].run(None, {
                "input_ids": decoder_input,
                "encoder_hidden_states": encoder_outputs[0],
                "encoder_attention_mask": inputs["attention_mask"]
            })
            next_token = decoder_outputs[0][0, -1].argmax()
            if next_token == self.tokenizer.eos_token_id:
                break
            decoder_input = np.hstack([decoder_input, [[next_token]]])
        
        return self.tokenizer.decode(decoder_input[0], skip_special_tokens=True)

if __name__ == "__main__":
    translator = FPGATranslator("/Users/faamt/Desktop/Nomadic Inc/Helsinki/student_ckpt/student_fpga")
    while True:
        text = input("Enter text (or 'quit'): ")
        if text.lower() == 'quit':
            break
        print(f"Translation: {translator.translate(text)}")