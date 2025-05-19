from transformers import MarianMTModel, MarianTokenizer
from optimum.exporters.onnx import main_export
from pathlib import Path

def convert_to_onnx(model_name: str, output_dir: str):
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Export using optimum
    main_export(
        model_name_or_path=model_name,
        task="translation",
        output=output_dir,
        opset=13
    )

# Arabic to English
convert_to_onnx("Helsinki-NLP/opus-mt-ar-en", "./onnx/ar-en")

# English to Arabic
convert_to_onnx("Helsinki-NLP/opus-mt-en-ar", "./onnx/en-ar")
