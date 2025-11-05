# Tokenizer

Tokenizers and scripts for preparing tokenizers for ModernBERT.

## Files

- `prepare_tokenizer.py` - Script to prepare RoBERTa-style tokenizer from base model (default: Qwen2.5-7B)
- `check_tokenizer.py` - Script to verify the prepared tokenizer

## Usage

### Prepare a new tokenizer:
```bash
python tokenizer/prepare_tokenizer.py --base Viet-Mistral/Vistral-7B-Chat --out tokenizer/HUIT-BERT
```

### Check tokenizer:
```bash
python tokenizer/check_tokenizer.py
```

