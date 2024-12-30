import gdown
import os

# Define the MODEL_CONFIGS dictionary
MODEL_CONFIGS = {
    'yolo': {
        'drive_url': 'https://drive.google.com/file/d/1-84XgKFTiM17IRKfEluDGko0gVlSvVGY',
        'local_path': 'weights/best.pt'
    },
    'florence': {
        'drive_url': 'https://drive.google.com/file/d/1ZubRSK_Y34_M8Mx8E1pcxbtouokD3IMG',
        'local_path': 'weights/icon_caption_florence/model.safetensors',
        
        # Define the URL for the PyTorch model file
        'pytorch_model_url': 'https://drive.google.com/file/d/1RZcaDSfM_o7qxp7wFJoo6gt-eCrK1J3h',
        'pytorch_model_path': 'weights/pytorch_model.bin'  # Local path to save pytorch_model.bin
    }
}
import gdown
import os
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

                # Check if the file exists and has content
                if os.path.getsize(output_path) > 0:
                    print(f"Model downloaded to {output_path}")
                    return  # Success, exit the function
                else:
                    print(f"Downloaded file is empty. Retrying...")
            except Exception as e:
                print(f"Error downloading model (Attempt {attempt + 1}/{retries}): {e}")
                if attempt < retries - 1:
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print(f"Max retries reached. Download failed.")
    except Exception as e:
        print(f"Error downloading model: {e}")


