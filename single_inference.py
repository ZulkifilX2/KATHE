import torch
import unicodedata
from load_model import load_translator_model

def normalize_kashmiri(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKC", text.strip())

def translate_sentence(text: str, model, tokenizer) -> str:
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=256
    ).to(model.device)
    
    kashmiri_id = tokenizer.convert_tokens_to_ids("kas_Arab")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=kashmiri_id,
            max_length=128,
            num_beams=6,
            length_penalty=1.2,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3
        )
        
    decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    return normalize_kashmiri(decoded_text)

if __name__ == "__main__":
    model, tokenizer = load_translator_model()
    test_sentence = "She was a true visionary."
    translation = translate_sentence(test_sentence, model, tokenizer)
    print(f"Source: {test_sentence}")
    print(f"Translation: {translation}")
