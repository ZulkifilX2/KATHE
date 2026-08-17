import torch
import re


# ============================================================
# PROMPT
# ============================================================

SYSTEM_PROMPT = (
    "You are an expert native Kashmiri translator. "
    "Translate the given English text into natural, accurate "
    "Kashmiri using the Perso-Arabic script."
)


def build_prompt(
    source_text: str,
) -> str:

    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"English: {source_text}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        "Kashmiri: "
    )


# ============================================================
# STRONG REPETITION DETECTOR
# ============================================================

def detect_repetition_loop(
    text: str,
    min_run: int = 3,
    max_ngram: int = 8,
    tail_tokens: int = 80,
):

    words = text.split()

    if len(words) < min_run:
        return False, None, 0

    # --------------------------------------------------------
    # Consecutive repeated blocks
    # --------------------------------------------------------

    for n in range(
        1,
        min(
            max_ngram,
            len(words)
        ) + 1,
    ):

        for i in range(
            0,
            len(words) - n * min_run + 1,
        ):

            block = words[
                i:i+n
            ]

            repeated = True

            for r in range(
                1,
                min_run,
            ):

                start = (
                    i + r * n
                )

                end = (
                    start + n
                )

                if words[
                    start:end
                ] != block:

                    repeated = False
                    break

            if repeated:

                return (
                    True,
                    " ".join(block),
                    min_run,
                )

    # --------------------------------------------------------
    # Tail-cycle detection
    # --------------------------------------------------------

    tail = words[
        max(
            0,
            len(words) - tail_tokens,
        ):
    ]

    for n in range(
        2,
        min(
            max_ngram,
            len(tail) // 3,
        ) + 1,
    ):

        for start in range(
            max(
                0,
                len(tail) - 40,
            ),
            len(tail) - 2 * n,
        ):

            block = tail[
                start:start+n
            ]

            repetitions = 1
            pos = start + n

            while (
                pos + n <= len(tail)
                and tail[
                    pos:pos+n
                ] == block
            ):

                repetitions += 1
                pos += n

            if repetitions >= min_run:

                return (
                    True,
                    " ".join(block),
                    repetitions,
                )

    return False, None, 0


# ============================================================
# ORTHOGRAPHIC NORMALIZATION
# ============================================================

def normalize_kashmiri(
    text: str,
) -> str:

    text = str(text)

    text = text.replace(
        "ے",
        "ی",
    )

    text = text.replace(
        "ك",
        "ک",
    )

    text = text.replace(
        ",",
        "،",
    )

    return text.strip()


# ============================================================
# CLEANING
# ============================================================

def clean_kashmiri_output(
    text: str,
) -> str:

    if not isinstance(
        text,
        str,
    ):

        return ""

    text = text.strip()

    prefixes = [
        r"^Kashmiri\s*:\s*",
        r"^Translation\s*:\s*",
        r"^Here is the translation\s*:\s*",
    ]

    for pattern in prefixes:

        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE,
        )

    for tag in [
        "<|im_start|>",
        "<|im_end|>",
        "<|assistant|>",
        "<|user|>",
    ]:

        text = text.replace(
            tag,
            "",
        )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# ============================================================
# BAD OUTPUT CHECK
# ============================================================

def is_bad_output(
    text: str,
    generated_tokens: int,
) -> bool:

    if not isinstance(
        text,
        str,
    ):

        return True

    text = text.strip()

    if len(text) < 8:
        return True

    if len(text) > 300:
        return True

    if generated_tokens >= 256:
        return True

    loop, _, _ = detect_repetition_loop(
        text,
        min_run=3,
        max_ngram=8,
        tail_tokens=80,
    )

    return loop


# ============================================================
# DECODING LADDER
# ============================================================

DECODING_STEPS = [

    {
        "name": "rep1.05",
        "repetition_penalty": 1.05,
        "no_repeat_ngram_size": 0,
    },

    {
        "name": "rep1.10",
        "repetition_penalty": 1.10,
        "no_repeat_ngram_size": 0,
    },

    {
        "name": "rep1.15",
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 0,
    },

    {
        "name": "rep1.10_ngram3",
        "repetition_penalty": 1.10,
        "no_repeat_ngram_size": 3,
    },

    {
        "name": "rep1.15_ngram3",
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    },
]


# ============================================================
# ONE GENERATION ATTEMPT
# ============================================================

def _generate_attempt(
    source_text: str,
    model,
    tokenizer,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
):

    prompt = build_prompt(
        source_text
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(
        model.device
    )

    prompt_length = (
        inputs["input_ids"].shape[-1]
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=3,
            do_sample=False,
            repetition_penalty=(
                repetition_penalty
            ),
            no_repeat_ngram_size=(
                no_repeat_ngram_size
            ),
            pad_token_id=(
                tokenizer.eos_token_id
            ),
        )

    generated_tokens = (
        outputs[0][prompt_length:]
    )

    text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    loop, pattern, run_length = (
        detect_repetition_loop(
            text,
            min_run=3,
            max_ngram=8,
            tail_tokens=80,
        )
    )

    return {
        "text": text,
        "tokens": len(
            generated_tokens
        ),
        "is_loop": loop,
        "pattern": pattern,
        "run_length": run_length,
    }


# ============================================================
# PUBLIC TRANSLATION FUNCTION
# ============================================================

def translate_sentence(
    text: str,
    model,
    tokenizer,
) -> str:

    last_text = ""

    for config in DECODING_STEPS:

        result = _generate_attempt(
            text,
            model,
            tokenizer,
            repetition_penalty=(
                config[
                    "repetition_penalty"
                ]
            ),
            no_repeat_ngram_size=(
                config[
                    "no_repeat_ngram_size"
                ]
            ),
        )

        last_text = result["text"]

        bad = is_bad_output(
            result["text"],
            result["tokens"],
        )

        if not bad:

            cleaned = (
                clean_kashmiri_output(
                    result["text"]
                )
            )

            return normalize_kashmiri(
                cleaned
            )

    # --------------------------------------------------------
    # Final fallback
    # --------------------------------------------------------

    return normalize_kashmiri(
        clean_kashmiri_output(
            last_text
        )
    )


if __name__ == "__main__":

    from load_model import (
        load_translator_model
    )

    model, tokenizer = (
        load_translator_model()
    )

    test_sentence = (
        "She was a true visionary."
    )

    translation = translate_sentence(
        test_sentence,
        model,
        tokenizer,
    )

    print(
        "Source:",
        test_sentence,
    )

    print(
        "Translation:",
        translation,
    )
