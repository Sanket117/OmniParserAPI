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
    
    # Load the processor (Florence2Processor) from a local directory or Hugging Face model
    processor = AutoProcessor.from_pretrained(MODEL_CONFIGS['florence']['local_path'])  # Correct path for Florence model
    
    # Load the caption model (AutoModelForCausalLM)
    caption_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS['florence']['local_path'],  # Path for local Florence model
        torch_dtype=torch.float32,
        local_files_only=True  # Ensures only local files are loaded
    ).to(device)
    
    return {
        'yolo_model': yolo_model,
        'processor': processor,
        'caption_model': caption_model
    }
