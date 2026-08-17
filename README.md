# KATHE 2026 — English to Kashmiri Machine Translation

Inference code for the KATHE 2026 English → Kashmiri machine
translation challenge.

## Model

The repository loads the current submission LoRA adapter from Hugging Face.

The inference code automatically detects whether the uploaded adapter
uses:

- a causal language model architecture, such as Qwen, or
- an encoder-decoder translation architecture, such as NLLB.

The Hugging Face adapter repository contains the actual model-specific
weights and configuration.

## Dataset

Training data:

AI4Bharat BPCC

Configuration:

`bpcc-seed-latest`

Language split:

`kas_Arab`

Training examples:

98,929

## Methodology

### Causal-LM inference

For the Qwen-style causal language model, inference uses an adaptive
decoding ladder:

1. repetition penalty 1.05
2. repetition penalty 1.10
3. repetition penalty 1.15
4. repetition penalty 1.10 + no-repeat 3-gram constraint
5. repetition penalty 1.15 + no-repeat 3-gram constraint

Outputs are rejected when they contain:

- repeated word cycles
- repeated multi-word loops
- extremely short output
- excessively long output
- generation reaching the 256-token limit

Accepted outputs receive conservative Kashmiri orthographic normalization.

### Encoder-decoder inference

For NLLB-style sequence-to-sequence models, the inference pipeline uses:

- 6-beam search
- length penalty 1.2
- repetition penalty 1.2
- no-repeat 3-gram constraint
- Kashmiri `kas_Arab` target language forcing

## Repository structure

```text
.
├── README.md
├── LICENSE
├── load_model.py
├── single_inference.py
├── batch_inference.py
├── requirements.txt
└── .gitignore
