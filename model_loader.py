"""Handle model downloading and loading."""

import os
from pathlib import Path
import gdown


def download_from_drive(url: str, output_path: str) -> None:
    """
    Download a file from Google Drive using the gdown library.
    
    Args:
        url (str): The Google Drive URL or direct download link.
        output_path (str): The local path where the file will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Create directories if they don't exist
    if not os.path.exists(output_path):  # Check if the file already exists
        print(f"Downloading file to: {output_path}")
        gdown.download(url, output_path, quiet=False)  # Download with progress bar
    else:
        print(f"File already exists at: {output_path}")


def ensure_models_available(config: dict) -> None:
    """
    Ensure all required models specified in the config are downloaded.
    
    Args:
        config (dict): A dictionary containing model names and their download information.
    """
    for model_name, model_info in config.items():
        print(f"Checking model: {model_name}")
        download_from_drive(
            model_info['drive_url'],
            model_info['local_path']
        )
