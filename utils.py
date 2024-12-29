import os
from PIL import Image
import io


def validate_image(file_content: bytes) -> Image.Image:
    """
    Validate and return a PIL Image.
    
    Args:
        file_content (bytes): The byte content of the uploaded image file.

    Returns:
        Image.Image: A PIL Image object.

    Raises:
        ValueError: If the file is not a valid image or cannot be processed.
    """
    try:
        # Open the image from byte content
        image = Image.open(io.BytesIO(file_content))
        image.verify()  # Verify it's a valid image
        
        # Reopen the image after verification for further processing
        image = Image.open(io.BytesIO(file_content))
        
        # Convert to RGB if necessary
        if image.mode in ('RGBA', 'LA') or (image.mode != 'RGB'):
            image = image.convert('RGB')
            
        return image
    except Exception as e:
        raise ValueError(f"Invalid image file: {str(e)}")


def save_temp_image(image: Image.Image, path: str, format: str = 'JPEG') -> str:
    """
    Save an image to a temporary location.
    
    Args:
        image (Image.Image): The PIL Image object to save.
        path (str): The file path where the image should be saved.
        format (str): The format to save the image in (default is 'JPEG').

    Returns:
        str: The file path where the image was saved.

    Raises:
        RuntimeError: If the image cannot be saved to the specified path.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save the image to the specified path and format
        image.save(path, format=format)
        return path
    except Exception as e:
        raise RuntimeError(f"Error saving image to {path}: {str(e)}")
