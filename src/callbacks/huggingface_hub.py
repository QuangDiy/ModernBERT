# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Upload checkpoints to Hugging Face Hub during training."""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist, reproducibility
from huggingface_hub import HfApi, create_repo
from omegaconf import OmegaConf

__all__ = ["HuggingFaceHubUploader"]

log = logging.getLogger(__name__)


class HuggingFaceHubUploader(Callback):
    """Uploads model checkpoints to Hugging Face Hub at specified intervals.

    This callback saves checkpoints to a temporary directory and uploads them
    to a Hugging Face Hub repository. It automatically includes model weights,
    configuration, tokenizer files, and a model card with training metrics.

    Args:
        repo_id (str): The Hugging Face Hub repository ID (e.g., 'username/model-name').
            Can also be set via HF_REPO_ID environment variable.
        upload_interval (str): How often to upload checkpoints (e.g., '3500ba' for every 3500 batches,
            '1ep' for every epoch). Defaults to '3500ba'.
        token (Optional[str]): Hugging Face API token. If not provided, will use HF_TOKEN
            environment variable or the token from huggingface-cli login.
        private (bool): Whether to create a private repository. Defaults to False.
        create_repo_if_missing (bool): Whether to create the repository if it doesn't exist.
            Defaults to True.
        upload_latest_only (bool): If True, only keeps the latest checkpoint in the repo.
            If False, keeps all uploaded checkpoints. Defaults to False.
        rank_zero_only (bool): Only upload from rank 0 process. Defaults to True.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        upload_interval: str = "3500ba",
        token: Optional[str] = None,
        private: bool = False,
        create_repo_if_missing: bool = True,
        upload_latest_only: bool = False,
        rank_zero_only: bool = True,
    ):
        self.repo_id = repo_id or os.environ.get("HF_REPO_ID")
        if not self.repo_id:
            raise ValueError(
                "repo_id must be provided either as an argument or via HF_REPO_ID environment variable"
            )

        self.upload_interval = upload_interval
        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self.create_repo_if_missing = create_repo_if_missing
        self.upload_latest_only = upload_latest_only
        self.rank_zero_only = rank_zero_only

        self.api = HfApi(token=self.token)
        self.last_upload_timestamp = None

        # Create repo if needed (only on rank 0)
        if self.create_repo_if_missing and (not rank_zero_only or dist.get_global_rank() == 0):
            try:
                create_repo(
                    repo_id=self.repo_id,
                    token=self.token,
                    private=self.private,
                    exist_ok=True,
                )
                log.info(f"Hugging Face Hub repository ready: {self.repo_id}")
            except Exception as e:
                log.warning(f"Could not create repository {self.repo_id}: {e}")

    def _should_upload(self, state: State) -> bool:
        """Check if we should upload based on the interval."""
        if self.rank_zero_only and dist.get_global_rank() != 0:
            return False

        # Parse interval (e.g., '3500ba', '1ep', '1000sp')
        from composer.core import Time, TimeUnit

        interval_value = int(''.join(filter(str.isdigit, self.upload_interval)))
        interval_unit = ''.join(filter(str.isalpha, self.upload_interval))

        # Map interval unit to TimeUnit
        unit_map = {
            'ba': TimeUnit.BATCH,
            'ep': TimeUnit.EPOCH,
            'sp': TimeUnit.SAMPLE,
            'tok': TimeUnit.TOKEN,
            'dur': TimeUnit.DURATION,
        }

        time_unit = unit_map.get(interval_unit)
        if time_unit is None:
            log.warning(f"Unknown time unit {interval_unit}, defaulting to batch")
            time_unit = TimeUnit.BATCH

        current_timestamp = state.timestamp.get(time_unit)

        # Check if we've reached the interval
        if self.last_upload_timestamp is None:
            return False  # Don't upload on first call

        return current_timestamp.value - self.last_upload_timestamp >= interval_value

    def _create_model_card(self, state: State, metrics: dict) -> str:
        """Create a model card with training information."""
        card = f"""---
language: multilingual
license: apache-2.0
tags:
- bert
- modernbert
- flexbert
- masked-language-modeling
---

# {self.repo_id.split('/')[-1]}

This model is a ModernBERT/FlexBERT checkpoint uploaded during training.

## Training Information

- **Step**: {state.timestamp.batch.value}
- **Epoch**: {state.timestamp.epoch.value}
- **Samples Seen**: {state.timestamp.sample.value}

## Metrics

"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                card += f"- **{key}**: {value:.4f}\n"

        card += """

## Model Architecture

This model uses the FlexBERT architecture with modern improvements over traditional BERT.

## Usage

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("{}")
tokenizer = AutoTokenizer.from_pretrained("{}")
```

## Citation

If you use this model, please cite the ModernBERT paper.
""".format(self.repo_id, self.repo_id)

        return card

    def batch_checkpoint(self, state: State, logger: Logger):
        """Called after checkpoint is saved. Upload to HF Hub if interval is reached."""
        if not self._should_upload(state):
            # Update timestamp for first call
            if self.last_upload_timestamp is None:
                interval_unit = ''.join(filter(str.isalpha, self.upload_interval))
                unit_map = {
                    'ba': 'batch',
                    'ep': 'epoch',
                    'sp': 'sample',
                    'tok': 'token',
                    'dur': 'duration',
                }
                time_attr = unit_map.get(interval_unit, 'batch')
                self.last_upload_timestamp = getattr(state.timestamp, time_attr).value
            return

        try:
            # Get current metrics
            metrics = {}
            if hasattr(state, 'eval_metrics') and state.eval_metrics:
                metrics.update(state.eval_metrics)
            if hasattr(state, 'train_metrics') and state.train_metrics:
                metrics.update(state.train_metrics)

            log.info(f"Uploading checkpoint to Hugging Face Hub: {self.repo_id}")

            # Save model in HF format to temporary directory
            with tempfile.TemporaryDirectory() as tmpdir:
                # Save the model
                if hasattr(state.model, 'module'):
                    model = state.model.module  # Unwrap DDP/FSDP
                else:
                    model = state.model

                # Save using HF's save_pretrained if available
                if hasattr(model, 'save_pretrained'):
                    model.save_pretrained(tmpdir)
                else:
                    # Fallback: save state dict
                    torch.save(model.state_dict(), os.path.join(tmpdir, "pytorch_model.bin"))
                    if hasattr(model, 'config'):
                        model.config.save_pretrained(tmpdir)

                # Save tokenizer if available
                if hasattr(state, 'tokenizer') and state.tokenizer is not None:
                    if hasattr(state.tokenizer, 'save_pretrained'):
                        state.tokenizer.save_pretrained(tmpdir)

                # Create and save model card
                model_card = self._create_model_card(state, metrics)
                with open(os.path.join(tmpdir, "README.md"), "w") as f:
                    f.write(model_card)

                # Upload to HF Hub
                commit_message = f"Upload checkpoint at step {state.timestamp.batch.value}"

                self.api.upload_folder(
                    folder_path=tmpdir,
                    repo_id=self.repo_id,
                    commit_message=commit_message,
                    token=self.token,
                )

                log.info(f"Successfully uploaded checkpoint to {self.repo_id}")

            # Update last upload timestamp
            interval_unit = ''.join(filter(str.isalpha, self.upload_interval))
            unit_map = {
                'ba': 'batch',
                'ep': 'epoch',
                'sp': 'sample',
                'tok': 'token',
                'dur': 'duration',
            }
            time_attr = unit_map.get(interval_unit, 'batch')
            self.last_upload_timestamp = getattr(state.timestamp, time_attr).value

        except Exception as e:
            log.error(f"Failed to upload checkpoint to Hugging Face Hub: {e}", exc_info=True)

    def fit_end(self, state: State, logger: Logger):
        """Upload final checkpoint at end of training."""
        log.info("Training complete. Uploading final checkpoint to Hugging Face Hub...")

        # Force upload regardless of interval
        old_timestamp = self.last_upload_timestamp
        self.last_upload_timestamp = -999999  # Ensure upload happens

        self.batch_checkpoint(state, logger)

        self.last_upload_timestamp = old_timestamp
