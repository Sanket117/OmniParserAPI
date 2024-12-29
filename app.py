from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import HTMLResponse
import base64

from models import load_models
from image_processing import process_image_with_models
from utils import validate_image

# Initialize FastAPI
app = FastAPI()

# Load models at startup
models = None

@app.on_event("startup")
async def load_all_models():
    global models
    models = load_models(device='cuda' if torch.cuda.is_available() else 'cpu')

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """
    Home page for the Image Processing API.
    """
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
    box_threshold: float = Query(0.05, ge=0.0, le=1.0, description="Box threshold (0.0 to 1.0)"),
    iou_threshold: float = Query(0.1, ge=0.0, le=1.0, description="IOU threshold (0.0 to 1.0)")
):
    """
    Process an uploaded image to detect objects and generate labels.

    Args:
        file (UploadFile): The uploaded image file (PNG or JPEG).
        box_threshold (float): Confidence threshold for box detection.
        iou_threshold (float): IOU threshold for overlapping boxes.

    Returns:
        dict: A dictionary containing the labeled image (Base64), detected coordinates, and parsed content.
    """
    try:
        # Validate file type
        if file.content_type not in ["image/png", "image/jpeg"]:
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

        # Convert labeled image to Base64
        labeled_img_bytes = io.BytesIO()
        labeled_img.save(labeled_img_bytes, format="JPEG")
        labeled_img_base64 = base64.b64encode(labeled_img_bytes.getvalue()).decode("utf-8")

        return {
            "labeled_image": f"data:image/jpeg;base64,{labeled_img_base64}",
            "coordinates": coordinates,
            "parsed_content": "\n".join(parsed_content)
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while processing the image."
        )
