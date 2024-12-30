import gdown
import os

# Define the MODEL_CONFIGS dictionary
MODEL_CONFIGS = {
    'yolo': {
        'drive_url': 'https://drive.google.com/uc?id=1p-Y7rd0FfjNnv_jewCi7ZjXH3T-qtyAa',
        'local_path': 'weights/best.pt'
    },
    'florence': {
        'drive_url': 'https://drive.google.com/uc?id=1hUCqZ3X8mcM-KcwWFjcsFg7PA0hUvE3k',
        'local_path': 'weights/icon_caption_florence/model.safetensors',
        
        # Define the URL for the PyTorch model file
        'pytorch_model_url': 'https://drive.google.com/file/d/1PiNJSTORmAbaOMpOv_wa6djuVLbN3gh9',
        'pytorch_model_path': 'weights/pytorch_model.bin'  # Local path to save pytorch_model.bin
    }
}

# Function to download the model
def download_model_from_google_drive(url, output_path):
    # Extract the file ID from the URL
    file_id = url.split('/d/')[1].split('/')[0]
    download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
    
    # Download the model using gdown
    gdown.download(download_url, output_path, quiet=False)

# Check if the PyTorch model file exists, if not, download it
florence_model_url = MODEL_CONFIGS['florence']['pytorch_model_url']
florence_model_path = MODEL_CONFIGS['florence']['pytorch_model_path']

# Download the model if it doesn't exist
if not os.path.exists(florence_model_path):
    print(f"Downloading the model from {florence_model_url}...")
    download_model_from_google_drive(florence_model_url, florence_model_path)
else:
    print(f"Model already exists at {florence_model_path}")
