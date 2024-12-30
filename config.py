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
import time

def download_model_from_google_drive(url, output_path, retries=3, delay=5):
    try:
        # Extract the file ID from the URL
        file_id = url.split('/d/')[1].split('/')[0]
        download_url = f'https://drive.google.com/uc?export=download&id={file_id}'
        
        # Retry logic for downloading the model
        for attempt in range(retries):
            try:
                print(f"Downloading {output_path} (Attempt {attempt + 1}/{retries})...")
                gdown.download(download_url, output_path, quiet=False)
                print(f"Model downloaded to {output_path}")
                return  # Success, exit the function
            except Exception as e:
                print(f"Error downloading model (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached. Download failed.")
    except Exception as e:
        print(f"Error downloading model: {e}")

