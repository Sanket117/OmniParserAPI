"""Configuration settings for the application."""

MODEL_CONFIGS = {
    'yolo': {
        # Correctly formatted Google Drive shareable link
        'drive_url': 'https://drive.google.com/uc?id=1p-Y7rd0FfjNnv_jewCi7ZjXH3T-qtyAa',
        # Ensure the path is relative to the project root or absolute
        'local_path': 'weights/best.pt'
    },
    'florence': {
        # Correctly formatted Google Drive shareable link
        'drive_url': 'https://drive.google.com/uc?id=1hUCqZ3X8mcM-KcwWFjcsFg7PA0hUvE3k',
        # Ensure the path is relative to the project root or absolute
        'local_path': 'weights/icon_caption_florence/model.safetensors'
    }
}
