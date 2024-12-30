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
    
    # Load YOLO model
    yolo_model = YOLO(MODEL_CONFIGS['yolo']['local_path']).to(device)
    
    # Load the processor with the trust_remote_code argument
    processor = AutoProcessor.from_pretrained(
        MODEL_CONFIGS['florence']['local_path'],
        trust_remote_code=True  # Allow custom code to run
    )
    
    # Load the caption model with the trust_remote_code argument
    caption_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS['florence']['local_path'],  # Path to directory
        torch_dtype=torch.float32,
        local_files_only=True,  # Ensures only local files are loaded
        trust_remote_code=True  # Allow custom code to run
    ).to(device)
    
    return {
        'yolo_model': yolo_model,
        'processor': processor,
        'caption_model': caption_model
    }
