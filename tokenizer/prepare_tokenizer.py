import argparse
import os

from transformers import AutoTokenizer


def build_roberta_style_tokenizer(base_model: str, out_dir: str, add_prefix_space: bool = True) -> None:
    """
    Build a RoBERTa-style tokenizer from a base model.
    
    Args:
        base_model: Base tokenizer/model repo id or local path
        out_dir: Output directory to save the prepared tokenizer
        add_prefix_space: Whether to add prefix space.
    """
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, trust_remote_code=True)

    special_tokens = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>",
        "mask_token": "<mask>",
        "cls_token": "<s>",
        "sep_token": "</s>",
    }
    tokenizer.add_special_tokens(special_tokens)

    if tokenizer.cls_token != "<s>":
        tokenizer.cls_token = "<s>"
    if tokenizer.sep_token != "</s>":
        tokenizer.sep_token = "</s>"
    if tokenizer.unk_token != "<unk>":
        tokenizer.unk_token = "<unk>"

    try:
        from tokenizers.processors import TemplateProcessing 

        if getattr(tokenizer, "is_fast", False) and hasattr(tokenizer, "_tokenizer"):
            bos_id = tokenizer.convert_tokens_to_ids("<s>")
            eos_id = tokenizer.convert_tokens_to_ids("</s>")
            tokenizer._tokenizer.post_processor = TemplateProcessing(
                single="<s> $A </s>",
                pair="<s> $A </s> </s> $B </s>",
                special_tokens=[
                    ("<s>", bos_id),
                    ("</s>", eos_id),
                ],
            )
    except Exception:
        pass

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    
    try:
        tokenizer.add_prefix_space = add_prefix_space
    except Exception:
        pass
    
    try:
        tokenizer.model_input_names = ["input_ids", "attention_mask"]
    except Exception:
        pass

    os.makedirs(out_dir, exist_ok=True)
    tokenizer.save_pretrained(out_dir)

    try:
        import json
        cfg_path = os.path.join(out_dir, "tokenizer_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        else:
            cfg = {}
        
        cfg["model_input_names"] = ["input_ids", "attention_mask"]
        cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
        cfg["add_prefix_space"] = add_prefix_space
        
        cfg.pop("chat_template", None)
        cfg.pop("use_default_system_prompt", None)
        
        chat_tokens = ["<<SYS>>", "<</SYS>>", "[INST]", "[/INST]"]
        
        if "additional_special_tokens" in cfg:
            cfg["additional_special_tokens"] = [
                token for token in cfg["additional_special_tokens"]
                if token not in chat_tokens
            ]
        
        if "added_tokens_decoder" in cfg:
            keys_to_remove = []
            for key, token_info in cfg["added_tokens_decoder"].items():
                if isinstance(token_info, dict) and token_info.get("content") in chat_tokens:
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                cfg["added_tokens_decoder"].pop(key)
        
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
        
        special_tokens_path = os.path.join(out_dir, "special_tokens_map.json")
        if os.path.exists(special_tokens_path):
            with open(special_tokens_path, "r", encoding="utf-8") as f:
                special_tokens_map = json.load(f)
            
            if "additional_special_tokens" in special_tokens_map:
                if isinstance(special_tokens_map["additional_special_tokens"], list):
                    special_tokens_map["additional_special_tokens"] = [
                        token for token in special_tokens_map["additional_special_tokens"]
                        if (isinstance(token, str) and token not in chat_tokens) or
                           (isinstance(token, dict) and token.get("content") not in chat_tokens)
                    ]
            
            with open(special_tokens_path, "w", encoding="utf-8") as f:
                json.dump(special_tokens_map, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    print(
        {
            "saved_to": os.path.abspath(out_dir),
            "unk_token": tokenizer.unk_token,
            "unk_token_id": tokenizer.unk_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "mask_token": tokenizer.mask_token,
            "mask_token_id": tokenizer.mask_token_id,
            "cls_token": tokenizer.cls_token,
            "cls_token_id": tokenizer.cls_token_id,
            "sep_token": tokenizer.sep_token,
            "sep_token_id": tokenizer.sep_token_id,
            "add_prefix_space": add_prefix_space,
            "model_input_names": ["input_ids", "attention_mask"],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a RoBERTa-style tokenizer from Qwen2.5 base")
    parser.add_argument(
        "--base",
        type=str,
        default="Qwen/Qwen2.5-7B",
        help="Base tokenizer/model repo id or local path (default: Qwen/Qwen2.5-7B)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=os.path.join("tokenizer", "qwen2.5-roberta"),
        help="Output directory to save the prepared tokenizer",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.getenv("HF_TOKEN", ""),
        help="Hugging Face token to login (defaults to env HF_TOKEN if set)",
    )
    parser.add_argument(
        "--add-prefix-space",
        action="store_true",
        default=True,
        help="Add prefix space to tokenizer (default: True). ",
    )
    parser.add_argument(
        "--no-add-prefix-space",
        dest="add_prefix_space",
        action="store_false",
        help="Disable add_prefix_space",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if getattr(args, "hf_token", None):
        try:
            from huggingface_hub import login

            login(token=args.hf_token)
        except Exception:
            pass
    build_roberta_style_tokenizer(args.base, args.out, args.add_prefix_space)
