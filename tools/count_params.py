import argparse
import os
import sys
from pathlib import Path

# When counting parameters we don't need to construct the real rotary embedding /
# FlashAttention kernels. Tell the model code to skip those hard dependencies so
# that parameter counting works even if flash_attn isn't installed (e.g. on
# Windows or CPU-only environments).
os.environ.setdefault("MODERNBERT_SKIP_ROPE", "1")

import importlib
import importlib.util
from importlib.machinery import ModuleSpec
import types

import torch
from omegaconf import OmegaConf as om
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))


def _new_spec(name: str):
    return ModuleSpec(name=name, loader=None, is_package=True)


def _install_flash_attn_stub():
    """Install lightweight flash_attn stubs when the real package is unavailable."""

    if importlib.util.find_spec("flash_attn") is not None:
        # Real package is discoverable; leave it alone.
        return

    if "flash_attn" in sys.modules and getattr(sys.modules["flash_attn"], "__dict__", None):
        # Already stubbed.
        return

    flash_attn_mod = types.ModuleType("flash_attn")
    flash_attn_mod.__spec__ = _new_spec("flash_attn")
    flash_attn_mod.__path__ = []

    layers_mod = types.ModuleType("flash_attn.layers")
    layers_mod.__spec__ = _new_spec("flash_attn.layers")
    layers_mod.__path__ = []

    class _IdentityRotaryEmbedding(torch.nn.Module):
        def __init__(self, dim, base=10000.0, scale_base=None, interleaved=False):
            super().__init__()
            self.dim = dim
            self.base = base
            self.scale_base = scale_base
            self.interleaved = interleaved

        def forward(self, qkv, **kwargs):
            return qkv

    rotary_mod = types.ModuleType("flash_attn.layers.rotary")
    rotary_mod.__spec__ = _new_spec("flash_attn.layers.rotary")
    rotary_mod.RotaryEmbedding = _IdentityRotaryEmbedding

    layers_mod.rotary = rotary_mod
    flash_attn_mod.layers = layers_mod

    ops_mod = types.ModuleType("flash_attn.ops")
    ops_mod.__spec__ = _new_spec("flash_attn.ops")
    ops_mod.__path__ = []

    triton_mod = types.ModuleType("flash_attn.ops.triton")
    triton_mod.__spec__ = _new_spec("flash_attn.ops.triton")
    triton_mod.__path__ = []

    triton_rotary_mod = types.ModuleType("flash_attn.ops.triton.rotary")
    triton_rotary_mod.__spec__ = _new_spec("flash_attn.ops.triton.rotary")

    def _apply_rotary(qkv, cos, sin, **kwargs):
        return qkv

    triton_rotary_mod.apply_rotary = _apply_rotary

    triton_mod.rotary = triton_rotary_mod
    ops_mod.triton = triton_mod
    flash_attn_mod.ops = ops_mod

    sys.modules["flash_attn"] = flash_attn_mod
    sys.modules["flash_attn.layers"] = layers_mod
    sys.modules["flash_attn.layers.rotary"] = rotary_mod
    sys.modules["flash_attn.ops"] = ops_mod
    sys.modules["flash_attn.ops.triton"] = triton_mod
    sys.modules["flash_attn.ops.triton.rotary"] = triton_rotary_mod


if os.environ.get("MODERNBERT_SKIP_ROPE") == "1":
    _install_flash_attn_stub()

from src.bert_layers.configuration_bert import FlexBertConfig
from main import build_model 

def count_params(model: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_cfg(defaults_path: Path, yaml_path: Path):
    with open(defaults_path, "r", encoding="utf-8") as f:
        defaults = om.load(f)
    with open(yaml_path, "r", encoding="utf-8") as f:
        specific = om.load(f)
    cfg = om.merge(defaults, specific)
    return cfg


def format_count(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.3f}M"
    if n >= 1_000:
        return f"{n/1_000:.3f}K"
    return str(n)


def main():
    parser = argparse.ArgumentParser(description="Count parameters for ModernBERT YAML configs")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(ROOT / "yamls" / "main"),
        help="Directory of YAML configs to scan",
    )
    parser.add_argument(
        "--defaults",
        type=str,
        default=str(ROOT / "yamls" / "defaults.yaml"),
        help="Path to defaults.yaml merged before each config",
    )
    parser.add_argument("--pattern", type=str, default="*.yaml", help="Filename pattern to include")
    parser.add_argument("--trainable", action="store_true", help="Report trainable parameters only")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path or HF repo to align vocab")
    parser.add_argument(
        "--pad-multiple",
        type=int,
        default=8,
        help="Pad vocab size to this multiple (use 1 to disable padding)",
    )
    parser.add_argument(
        "--respect_tokenizer",
        action="store_true",
        default=True,
        help="Inject tokenizer vocab_size into model config before building",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a compact model summary table (Model Size, Hidden Dim, Num Layers, Vocab, Context, Task)",
    )
    args = parser.parse_args()

    target_dir = Path(args.dir)
    defaults_path = Path(args.defaults)
    yaml_files = sorted(target_dir.glob(args.pattern))

    if len(yaml_files) == 0:
        print(f"No YAML files found in {target_dir} with pattern {args.pattern}")
        return

    print(f"Scanning {len(yaml_files)} configs in {target_dir} (trainable_only={args.trainable})")

    rows = []
    summary_rows = []
    for yaml_path in yaml_files:
        try:
            cfg = load_cfg(defaults_path, yaml_path)

            target_vocab = None
            if args.tokenizer and args.respect_tokenizer:
                try:
                    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True, trust_remote_code=True)
                    target_vocab = len(tok)
                    if args.pad_multiple and args.pad_multiple > 1:
                        pm = args.pad_multiple
                        target_vocab = target_vocab + ((-target_vocab) % pm)
                    om.set_struct(cfg, False)
                    cfg.model.tokenizer_name = args.tokenizer
                    if getattr(cfg.model, "name", None) == "flex_bert":
                        if "model_config" not in cfg.model:
                            cfg.model.model_config = {}
                        cfg.model.model_config["allow_embedding_resizing"] = True
                        cfg.model.model_config["vocab_size"] = int(target_vocab)
                except Exception as e:
                    print(f"[WARN] Failed to load tokenizer '{args.tokenizer}': {e}")
                    target_vocab = None

            model = build_model(cfg.model)

            used_vocab = None
            try:
                emb = model.model.get_input_embeddings()
                used_vocab = emb.num_embeddings
                if target_vocab is not None and emb.num_embeddings != target_vocab:
                    model.model.resize_token_embeddings(int(target_vocab))
                    emb = model.model.get_input_embeddings()
                    used_vocab = emb.num_embeddings
            except Exception:
                pass

            n_total = count_params(model, trainable_only=False)
            n_train = count_params(model, trainable_only=True)

            token_emb_params = None
            head_vocab_params = None
            non_embed_params = None
            try:
                core = getattr(model, "model", model)
                tok_emb = core.bert.embeddings.tok_embeddings
                token_emb_params = tok_emb.weight.numel()

                tie = getattr(core.config, "tie_word_embeddings", True)
                dec = core.get_output_embeddings() if hasattr(core, "get_output_embeddings") else None
                if dec is not None:
                    if not tie:
                        head_vocab_params = dec.weight.numel() + (dec.bias.numel() if dec.bias is not None else 0)
                    else:
                        head_vocab_params = 0

                if token_emb_params is not None:
                    non_embed_params = n_total - token_emb_params - (head_vocab_params or 0)
            except Exception:
                pass

            name = yaml_path.stem
            rows.append(
                (
                    name,
                    n_total,
                    n_train,
                    used_vocab if used_vocab is not None else target_vocab,
                    token_emb_params,
                    head_vocab_params,
                    non_embed_params,
                )
            )

            try:
                core = getattr(model, "model", model)
                hidden_dim = getattr(core.config, "hidden_size", None)
                num_layers = getattr(core.config, "num_hidden_layers", None)
                vocab_eff = (used_vocab if used_vocab is not None else getattr(core.config, "vocab_size", None))
                ctx_len = getattr(core.config, "max_position_embeddings", None)
                cls_name = core.__class__.__name__
                if "MaskedLM" in cls_name:
                    task = "Masked LM"
                elif "SequenceClassification" in cls_name:
                    task = "Sequence Classification"
                elif "MultipleChoice" in cls_name:
                    task = "Multiple Choice"
                else:
                    task = "N/A"

                summary_rows.append((name, n_total, hidden_dim, num_layers, vocab_eff, ctx_len, task))
            except Exception:
                pass
        except Exception as e:
            print(f"[ERROR] {yaml_path.name}: {e}")

    if len(summary_rows) > 0 and args.summary:
        name_w = max((len(r[0]) for r in summary_rows), default=10)

        def fmt_model_size(n: int) -> str:
            if n >= 1_000_000:
                return f"{int(round(n/1_000_000))}M"
            if n >= 1_000:
                return f"{int(round(n/1_000))}K"
            return str(n)

        header = (
            f"{'Model'.ljust(name_w)}  {'Model Size':>10}  {'Hidden Dim':>10}  {'Num Layers':>10}  {'Vocab Size':>10}  {'Context':>7}  Task"
        )
        print(header)
        print("-" * len(header))
        for name, n_total, hidden_dim, num_layers, vocab_eff, ctx_len, task in summary_rows:
            print(
                f"{name.ljust(name_w)}  {fmt_model_size(n_total):>10}  {str(hidden_dim):>10}  {str(num_layers):>10}  {str(vocab_eff):>10}  {str(ctx_len):>7}  {task}"
            )

    name_w = max((len(r[0]) for r in rows), default=10)
    header = f"{'config'.ljust(name_w)}  total_params  trainable_params  vocab_used  token_emb  head_vocab  non_embed"
    print(header)
    print("-" * len(header))
    for name, n_total, n_train, vocab_used, token_emb_params, head_vocab_params, non_embed_params in rows:
        vocab_str = str(vocab_used) if vocab_used is not None else "-"
        tok_str = format_count(token_emb_params) if token_emb_params is not None else "-"
        head_str = format_count(head_vocab_params) if head_vocab_params is not None else "-"
        non_emb_str = format_count(non_embed_params) if non_embed_params is not None else "-"
        print(
            f"{name.ljust(name_w)}  {format_count(n_total):>12}  {format_count(n_train):>16}  {vocab_str:>10}  {tok_str:>9}  {head_str:>9}  {non_emb_str:>9}"
        )


if __name__ == "__main__":
    main()

# python .\tools\count_params.py --dir yamls\main --pattern flex-bert-vi-base.yaml --tokenizer tokenizer\huit-bert