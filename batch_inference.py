import sys
import pandas as pd
from load_model import load_translator_model
from single_inference import translate_sentence

def batch_translate(input_csv_path: str, output_csv_path: str):
    print(f"Loading data from {input_csv_path}...")
    df = pd.read_csv(input_csv_path)
    
    col = "sentence" if "sentence" in df.columns else "english"
    model, tokenizer = load_translator_model()
    
    translations = []
    total = len(df)
    print(f"Translating {total} sentences...")
    
    for i, text in enumerate(df[col]):
        tr = translate_sentence(str(text), model, tokenizer)
        translations.append(tr)
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{total} rows")
            
    df["kashmiri_text"] = translations
    
    if "ID" in df.columns:
        sub = df[["ID", "kashmiri_text"]]
    else:
        sub = df
        
    sub.to_csv(output_csv_path, index=False)
    print(f"✅ Batch inference complete. Saved to {output_csv_path}")

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "input.csv"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "submission.csv"
    batch_translate(in_path, out_path)
