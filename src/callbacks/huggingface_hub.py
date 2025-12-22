# Copyright 2024 onwards Answer.AI, LightOn, and contributors
# License: Apache-2.0

"""Upload checkpoints to Hugging Face Hub during training."""

import logging
import os
from pathlib import Path
from typing import Optional

from composer.core import Callback, State
from composer.loggers import Logger
from composer.utils import dist
from huggingface_hub import HfApi, create_repo

__all__ = ["HuggingFaceHubUploader"]

log = logging.getLogger(__name__)


class HuggingFaceHubUploader(Callback):
    """Uploads Composer .pt checkpoints to Hugging Face Hub at specified intervals.

    This callback uploads Composer checkpoint files (.pt) to a Hugging Face Hub repository
    with a flat structure using the run name from the training config.

    Args:
        repo_id (str): The Hugging Face Hub repository ID (e.g., 'username/model-name').
            Can also be set via HF_REPO_ID environment variable.
        run_name (str): The run name from training config (used in checkpoint filename).
        upload_interval (str): How often to upload checkpoints (e.g., '3500ba' for every 3500 batches,
            '1ep' for every epoch). Defaults to '3500ba'.
        token (Optional[str]): Hugging Face API token. If not provided, will use HF_TOKEN
            environment variable or the token from huggingface-cli login.
        private (bool): Whether to create a private repository. Defaults to False.
        create_repo_if_missing (bool): Whether to create the repository if it doesn't exist.
            Defaults to True.
        rank_zero_only (bool): Only upload from rank 0 process. Defaults to True.
        save_folder (Optional[str]): Path to the folder where Composer saves checkpoints.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        run_name: Optional[str] = None,
        upload_interval: str = "3500ba",
        token: Optional[str] = None,
        private: bool = False,
        create_repo_if_missing: bool = True,
        rank_zero_only: bool = True,
        save_folder: Optional[str] = None,
    ):
        self.repo_id = repo_id or os.environ.get("HF_REPO_ID")
        if not self.repo_id:
            raise ValueError(
                "repo_id must be provided either as an argument or via HF_REPO_ID environment variable"
            )

        self.run_name = run_name
        if not self.run_name:
            raise ValueError("run_name must be provided")

        self.upload_interval = upload_interval
        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self.create_repo_if_missing = create_repo_if_missing
        self.rank_zero_only = rank_zero_only
        self.save_folder = save_folder

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

    def batch_checkpoint(self, state: State, logger: Logger):
        """Called after checkpoint is saved. Upload .pt checkpoint to HF Hub if interval is reached."""
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

        if not self.save_folder:
            log.warning("save_folder not provided, cannot upload checkpoint")
            return

        try:
            batch_num = state.timestamp.batch.value
            log.info(f"Uploading checkpoint to Hugging Face Hub: {self.repo_id}")

            # If save_folder is relative, make it absolute
            save_path = Path(self.save_folder)
            if not save_path.is_absolute():
                save_path = Path(os.getcwd()) / save_path

            # Try multiple checkpoint file patterns
            pt_files_to_try = [
                save_path / "latest-rank0.pt",  # Standard latest checkpoint (may be symlink)
                save_path / f"ba{batch_num}-rank0.pt",  # Batch-specific checkpoint
            ]

            # Also look for any checkpoint files in the directory (excluding symlinks)
            if save_path.exists():
                for f in save_path.glob("*-rank0.pt"):
                    if not f.is_symlink():  # Skip symlinks
                        pt_files_to_try.append(f)

            pt_file = None
            for candidate in pt_files_to_try:
                if candidate.exists():
                    # Resolve symlink if it is one
                    if candidate.is_symlink():
                        pt_file = candidate.resolve()
                    else:
                        pt_file = candidate

                    # Verify the resolved file exists
                    if pt_file.exists():
                        break
                    else:
                        pt_file = None

            if pt_file:
                # Use Composer's original checkpoint filename (e.g., ep0-ba200-rank0.pt)
                filename = pt_file.name
                
                log.info(f"Uploading Composer checkpoint: {pt_file} as {filename}")
                self.api.upload_file(
                    path_or_fileobj=str(pt_file),
                    path_in_repo=filename,
                    repo_id=self.repo_id,
                    token=self.token,
                    commit_message=f"Upload checkpoint at step {batch_num}"
                )
                log.info(f"Successfully uploaded checkpoint to {self.repo_id}/{filename}")
            else:
                log.warning(f"Composer checkpoint not found in {save_path}")

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
