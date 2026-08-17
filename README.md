# KATHE 2026 — English to Kashmiri Machine Translation

This repository contains the inference code for the KATHE 2026
English → Kashmiri machine translation submission.

## Model

Current submission model:

`unsloth/Qwen2.5-7B-Instruct-bnb-4bit`

with a PEFT LoRA adapter trained for English → Kashmiri translation.

The trained adapter is hosted separately on Hugging Face.

Hugging Face model:

`YOUR_HF_USERNAME/kashmiri-qwen-2527-lora`

## Dataset

The model was trained using the AI4Bharat BPCC dataset.

Configuration:

`bpcc-seed-latest`

Language split:

`kas_Arab`

Training data:

98,929 English/Kashmiri examples.

## Methodology

The translation pipeline uses the Qwen instruction-tuned language model
with a trained LoRA adapter.

Inference uses deterministic beam search with an adaptive decoding
ladder:

1. repetition penalty 1.05
2. repetition penalty 1.10
3. repetition penalty 1.15
4. repetition penalty 1.10 + no-repeat 3-gram constraint
5. repetition penalty 1.15 + no-repeat 3-gram constraint

Outputs are rejected when they exhibit:

- long repeated word cycles
- repeated multi-word loops
- empty or very short output
- excessively long output
- the 256-token generation ceiling

Accepted outputs receive conservative Kashmiri orthographic
normalization.

Normalization includes:

- `ے` → `ی`
- `ك` → `ک`
- ASCII comma → Kashmiri/Arabic comma `،`

## Repository structure

```text
.
├── batch_inference.py
├── single_inference.py
├── load_model.py
├── requirements.txt
├── README.md
└── LICENSE
