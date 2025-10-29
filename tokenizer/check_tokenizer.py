from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("tokenizer/huit-bert", use_fast=True)

print("single:", tok("Xin chào").input_ids[:5], tok.convert_ids_to_tokens(tok("Xin chào").input_ids[:5]))

enc = tok("câu A", "câu B")

print("pair head/tail tokens:", tok.convert_ids_to_tokens(enc.input_ids[:6]), tok.convert_ids_to_tokens(enc.input_ids[-6:]))

print("mask/pad/cls/sep ids:", tok.mask_token_id, tok.pad_token_id, tok.cls_token_id, tok.sep_token_id)