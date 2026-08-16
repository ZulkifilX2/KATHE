import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

# NOTE: We will replace this with your actual HF username
HF_REPO_ID = "YOUR_HF_USERNAME/kathe-2026-nllb-1.3b"
BASE_MODEL_NAME = "facebook/nllb-200-distilled-1.3B"

def load_translator_model():
    """Loads base NLLB 1.3B and attaches the fine-tuned LoRA adapter."""
    print(f"Loading tokenizer from {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, 
        src_lang="eng_Latn", 
        tgt_lang="kas_Arab"
    )
    
    print("Loading base model in FP16...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Downloading and attaching LoRA weights from {HF_REPO_ID}...")
    model = PeftModel.from_pretrained(base_model, HF_REPO_ID)
    model.eval()
    
    return model, tokenizer

if __name__ == "__main__":
    print("Testing model loading script...")
    model, tokenizer = load_translator_model()
    print("✅ Model and tokenizer loaded successfully!")
