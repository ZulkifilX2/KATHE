# ============================================================
# KATHE 2026 - MODEL LOADER
# Supports both:
#   - Qwen causal-LM LoRA adapters
#   - NLLB seq2seq LoRA adapters
# ============================================================

import os

import torch

from huggingface_hub import HfApi
from peft import PeftConfig, PeftModel
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


# ============================================================
# CONFIGURATION
# ============================================================

HF_REPO_ID = os.environ.get(
    "KATHE_HF_REPO",
    "NyxT-T/kathe-2026-kashmiri",
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_translator_model():

    print(
        f"Loading adapter configuration from:\n"
        f"{HF_REPO_ID}"
    )

    # --------------------------------------------------------
    # Read PEFT configuration directly from HF
    # --------------------------------------------------------

    peft_config = PeftConfig.from_pretrained(
        HF_REPO_ID
    )

    base_model_name = (
        peft_config.base_model_name_or_path
    )

    task_type = str(
        peft_config.task_type
    )

    print(
        "\nBase model:",
        base_model_name
    )

    print(
        "PEFT task type:",
        task_type
    )

    # --------------------------------------------------------
    # Inspect base architecture
    # --------------------------------------------------------

    base_config = AutoConfig.from_pretrained(
        base_model_name
    )

    is_encoder_decoder = (
        getattr(
            base_config,
            "is_encoder_decoder",
            False,
        )
    )

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    print(
        "\nLoading tokenizer..."
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    # --------------------------------------------------------
    # Load correct model architecture
    # --------------------------------------------------------

    if is_encoder_decoder:

        print(
            "Architecture: encoder-decoder"
        )

        base_model = (
            AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        )

    else:

        print(
            "Architecture: causal language model"
        )

        base_model = (
            AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        )

    # --------------------------------------------------------
    # Attach LoRA
    # --------------------------------------------------------

    print(
        "\nLoading LoRA adapter..."
    )

    model = PeftModel.from_pretrained(
        base_model,
        HF_REPO_ID,
        is_trainable=False,
    )

    model.eval()

    print(
        "\n✅ Model and adapter loaded successfully"
    )

    print(
        "Base:",
        base_model_name
    )

    print(
        "Encoder-decoder:",
        is_encoder_decoder
    )

    return (
        model,
        tokenizer,
        is_encoder_decoder,
    )


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    (
        model,
        tokenizer,
        is_encoder_decoder,
    ) = load_translator_model()

    print(
        "\n✅ Model loading test passed"
    )
