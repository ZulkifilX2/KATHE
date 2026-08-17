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
        f"Loading data from {input_csv_path}..."
    )

    df = pd.read_csv(
        input_csv_path
    )

    if "sentence" not in df.columns:
        raise ValueError(
            "Input CSV must contain a 'sentence' column."
        )

    if "ID" not in df.columns:
        raise ValueError(
            "Input CSV must contain an 'ID' column."
        )

    model, tokenizer = (
        load_translator_model()
    )

    translations = []

    total = len(df)

    print(
        f"Translating {total} sentences..."
    )

    for i, text in enumerate(
        df["sentence"],
        start=1,
    ):

        translation = (
            translate_sentence(
                str(text),
                model,
                tokenizer,
            )
        )

        translations.append(
            translation
        )

        if i % 25 == 0:

            print(
                f"Processed {i}/{total}"
            )

    submission = pd.DataFrame({
        "ID": df["ID"],
        "kashmiri_text": translations,
    })

    submission.to_csv(
        output_csv_path,
        index=False,
    )

    print(
        f"\n✅ Batch inference complete."
    )

    print(
        f"Saved to {output_csv_path}"
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
