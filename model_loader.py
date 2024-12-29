"""Handle model downloading and loading."""

import os
import gdown
from pathlib import Path

def download_from_drive(url: str, output_path: str) -> None:
    """Download file from Google Drive."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

def ensure_models_available(config: dict) -> None:
    """Ensure all required models are downloaded."""
    for model_name, model_info in config.items():
        download_from_drive(
            model_info['drive_url'],
            model_info['local_path']
        )