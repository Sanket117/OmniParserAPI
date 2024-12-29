import numpy as np
from PIL import Image
import base64
import io

def process_image_with_models(
    image: Image.Image,
    models: dict,
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1
) -> tuple:
    """Process image with YOLO and captioning models."""
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Run YOLO detection
    results = models['yolo_model'](img_array)
    
    # Get bounding boxes and labels
    boxes = results[0].boxes
    coordinates = boxes.xyxy.cpu().numpy().tolist()
    
    # Process with caption model
    inputs = models['processor'](images=image, return_tensors="pt")
    outputs = models['caption_model'].generate(
        **inputs,
        max_length=50,
        num_beams=5,
        early_stopping=True
    )
    
    # Decode captions
    captions = models['processor'].batch_decode(outputs, skip_special_tokens=True)
    
    # Create labeled image
    img_with_boxes = results[0].plot()
    
    # Convert numpy array to PIL Image and then to base64
    labeled_img = Image.fromarray(img_with_boxes)
    buffered = io.BytesIO()
    labeled_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    
    return img_str, coordinates, captions
