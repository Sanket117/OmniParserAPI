from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
import base64

from models import load_models
from image_processing import process_image_with_models
from utils import validate_image, save_temp_image

# Initialize FastAPI  
app = FastAPI()

# Load models at startup
models = load_models(device='cpu')  # Use 'cuda' if GPU is available

@app.get("/", response_class=HTMLResponse)
async def read_root():
    return """
    <html>
        <body>
            <h1>Image Processing API</h1>
            <p>Visit <a href="/docs">/docs</a> to see the API documentation.</p>
        </body>
    </html>
    """

@app.post("/process/")
async def process_image(
    file: UploadFile = File(...),
    box_threshold: float = 0.05,
    iou_threshold: float = 0.1
):
    try:
        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload a PNG or JPEG image."
            )

        # Read and validate image
        content = await file.read()
        image = validate_image(content)
        
        # Process image
        labeled_img, coordinates, parsed_content = process_image_with_models(
            image,
            models,
            box_threshold,
            iou_threshold
        )

        return {
            "labeled_image": base64.b64encode(labeled_img).decode("utf-8"),
            "parsed_content": "\n".join(parsed_content),
            "coordinates": coordinates
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing the image: {str(e)}"
        )
