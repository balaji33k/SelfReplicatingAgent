"""
persist.py — Upload generation artifacts to a HuggingFace Dataset repo.

Called automatically after each generation completes so all results
survive HuggingFace Space restarts and are browsable at:
  https://huggingface.co/datasets/Balaji33k/self-replicating-agent-runs

Requires: HF_TOKEN set as a Space secret (Settings → Variables and secrets).
Non-fatal: if upload fails, evolution continues uninterrupted.
"""
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATASET_REPO = "Balaji33k/self-replicating-agent-runs"

# Files to skip when uploading (sandbox outputs, caches, etc.)
SKIP_PATTERNS = {"*.pyc", "__pycache__", "sandbox", "*.log"}


def _should_skip(path: Path) -> bool:
    for part in path.parts:
        if part in ("__pycache__", "sandbox"):
            return True
    return path.suffix in (".pyc",)


def upload_generation(gen_num: int, gen_dir: Path) -> bool:
    """
    Upload all meaningful files from gen_dir to the HF Dataset repo.

    Uploads:
      - All .py agent source files (the evolved pipeline code)
      - All .json result files (task solutions + summaries)

    Returns True on success, False on any failure (non-fatal).
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.warning(
            "[persist] HF_TOKEN not set — skipping upload. "
            "Add it under Space Settings → Variables and secrets."
        )
        return False

    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("[persist] huggingface_hub not installed — skipping upload.")
        return False

    try:
        api = HfApi(token=token)

        # Create dataset repo if it doesn't exist yet
        api.create_repo(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            exist_ok=True,
            private=False,
        )
        logger.info(f"[persist] Dataset repo ready: {DATASET_REPO}")

        # Collect files to upload
        files_to_upload = []
        for fp in sorted(gen_dir.rglob("*")):
            if fp.is_file() and not _should_skip(fp):
                rel = fp.relative_to(gen_dir)
                files_to_upload.append((fp, str(rel)))

        if not files_to_upload:
            logger.warning(f"[persist] No files found in {gen_dir} to upload.")
            return False

        # Upload each file
        uploaded = 0
        for fp, rel_path in files_to_upload:
            try:
                api.upload_file(
                    path_or_fileobj=str(fp),
                    path_in_repo=f"gen_{gen_num}/{rel_path}",
                    repo_id=DATASET_REPO,
                    repo_type="dataset",
                    commit_message=f"gen_{gen_num}: {rel_path}",
                )
                uploaded += 1
            except Exception as e:
                logger.warning(f"[persist] Could not upload {rel_path}: {e}")

        logger.info(
            f"[persist] ✅ Gen {gen_num} — {uploaded}/{len(files_to_upload)} files "
            f"uploaded to huggingface.co/datasets/{DATASET_REPO}/tree/main/gen_{gen_num}/"
        )
        return True

    except Exception as e:
        logger.warning(f"[persist] Upload failed (non-fatal): {e}")
        return False
