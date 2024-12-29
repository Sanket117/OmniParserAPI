from transformers import AutoProcessor, AutoModelForCausalLM
import torch
from ultralytics import YOLO
from config import MODEL_CONFIGS
from model_loader import ensure_models_available

def load_models(device='cpu'):
    """Initialize and load all required models."""
    # Ensure models are downloaded
    ensure_models_available(MODEL_CONFIGS)
    
    # Set default dtype for torch
    torch.set_default_dtype(torch.float32)
    
    yolo_model = YOLO(MODEL_CONFIGS['yolo']['local_path']).to(device)
    
    processor = AutoProcessor.from_pretrained(
        "microsoft/Florence-2-base", 
        trust_remote_code=True,
        token=False
    )
    
    caption_model = AutoModelForCausalLM.from_pretrained(
        "weights/icon_caption_florence",
        torch_dtype=torch.float32,
        trust_remote_code=True,
        token=False,
        local_files_only=True,
        pretrained_model_name_or_path=MODEL_CONFIGS['florence']['local_path']
    ).to(device)
    
    return {
        'yolo_model': yolo_model,
        'processor': processor,
        'caption_model': caption_model
    }