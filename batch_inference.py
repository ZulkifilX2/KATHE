# ============================================================
# KATHE 2026 - BATCH INFERENCE
# ============================================================

import sys

import pandas as pd

from load_model import (
    load_translator_model,
)

from single_inference import (
    translate_sentence,
)


def batch_translate(
    input_csv_path: str,
    output_csv_path: str,
):

    print(
        f"Loading:\n{input_csv_path}"
    )

    df = pd.read_csv(
        input_csv_path
    )

    # --------------------------------------------------------
    # Validate competition input
    # --------------------------------------------------------

    if "ID" not in df.columns:
        raise ValueError(
            "Input CSV must contain an 'ID' column."
        )

    if "sentence" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'sentence' column."
        )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        df.columns.tolist()
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    (
        model,
        tokenizer,
        is_encoder_decoder,
    ) = load_translator_model()

    # --------------------------------------------------------
    # Translate
    # --------------------------------------------------------

    translations = []

    total = len(df)

    print(
        f"\nTranslating {total} sentences..."
    )

    for i, source_text in enumerate(
        df["sentence"],
        start=1,
    ):

        translation = (
            translate_sentence(
                str(source_text),
                model,
                tokenizer,
                is_encoder_decoder,
            )
        )

        translations.append(
            translation
        )

        if i % 25 == 0:

            print(
                f"Processed {i}/{total}"
            )

    # --------------------------------------------------------
    # Submission
    # --------------------------------------------------------

    submission = pd.DataFrame({
        "ID": df["ID"],
        "kashmiri_text": translations,
    })

    submission.to_csv(
        output_csv_path,
        index=False,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    assert len(submission) == total

    assert list(
        submission.columns
    ) == [
        "ID",
        "kashmiri_text",
    ]

    empty = (
        submission[
            "kashmiri_text"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
        .sum()
    )

    if empty > 0:

        raise RuntimeError(
            f"❌ {empty} empty translations found."
        )

    print(
        "\n✅ Batch inference complete"
    )

    print(
        "Output:",
        output_csv_path
    )

    print(
        "Rows:",
        len(submission)
    )


if __name__ == "__main__":

    input_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "englishdev.csv"
    )

    output_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "submission.csv"
    )

    batch_translate(
        input_path,
        output_path,
    )
