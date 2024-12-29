from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from ultralytics import YOLO
from config import MODEL_CONFIGS
from model_loader import ensure_models_available


def load_models(device=None):
    """
    Initialize and load all required models.

    Args:
        device (str, optional): The device to load the models onto. Defaults to 'cuda' if available, otherwise 'cpu'.

    Returns:
        dict: A dictionary containing loaded YOLO, processor, and captioning models.
    """
    # Determine device
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure models are downloaded
    ensure_models_available(MODEL_CONFIGS)

    # Set default dtype for torch
    torch.set_default_dtype(torch.float32)

    # Load YOLO model
    try:
        yolo_model = YOLO(MODEL_CONFIGS['yolo']['local_path']).to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading YOLO model: {e}")

    # Load Florence-2 processor
    try:
        processor = AutoProcessor.from_pretrained(
            "microsoft/Florence-2-base", 
            trust_remote_code=True
        )
    except Exception as e:
        raise RuntimeError(f"Error loading processor: {e}")

    # Load captioning model
    try:
        caption_model = AutoModelForCausalLM.from_pretrained(
            MODEL_CONFIGS['florence']['local_path'],
            torch_dtype=torch.float32,
            trust_remote_code=True,
            local_files_only=True
        ).to(device)
    except Exception as e:
        raise RuntimeError(f"Error loading caption model: {e}")

    # Return loaded models
    return {
        'yolo_model': yolo_model,
        'processor': processor,
        'caption_model': caption_model
    }
