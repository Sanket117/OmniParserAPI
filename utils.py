import os
from PIL import Image
import io

def validate_image(file_content: bytes) -> Image.Image:
    """Validate and return a PIL Image."""
    try:
        # Open the image from byte content
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify it's a valid image
        
        # Reopen the image after verification to allow further manipulation
        image = Image.open(io.BytesIO(file_content))  
        
        # Convert to RGB if image has alpha channel or is not RGB
        if image.mode in ('RGBA', 'LA') or (image.mode != 'RGB'):
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        # Handle any errors during image verification
        raise ValueError(f"Invalid image file: {str(e)}")

def save_temp_image(image: Image.Image, path: str) -> str:
    """Save image to temporary location."""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the image to the given path
        image.save(path)
        return path
    except Exception as e:
        # Handle file saving errors
        raise RuntimeError(f"Error saving image to {path}: {str(e)}")
