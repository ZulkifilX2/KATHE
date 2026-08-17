import os

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from peft import PeftModel


# ============================================================
# CONFIGURATION
# ============================================================

# Replace this with your actual Hugging Face repository.
HF_REPO_ID = "NyxT-T/kathe-2026-kashmiri"

BASE_MODEL_NAME = (
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_translator_model():
    """
    Load the base Qwen model and attach the trained
    Kashmiri LoRA adapter from Hugging Face.
    """

    print(
        f"Loading tokenizer from {BASE_MODEL_NAME}..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        trust_remote_code=True,
    )

    print(
        "Loading base Qwen model..."
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(
        f"Loading LoRA adapter from {HF_REPO_ID}..."
    )

    model = PeftModel.from_pretrained(
        base_model,
        HF_REPO_ID,
        is_trainable=False,
    )

    model.eval()

    print(
        "✅ Model and adapter loaded successfully"
    )

    return model, tokenizer


if __name__ == "__main__":

    model, tokenizer = (
        load_translator_model()
    )

    print(
        "✅ Model loading test passed"
    )
