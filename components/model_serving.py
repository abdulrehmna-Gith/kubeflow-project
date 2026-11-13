
from pathlib import Path
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
from ultralytics import YOLO


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "model_registry"
    / "YOLO26n"
    / "v1"
    / "model.onnx"
)

# Create FastAPI application
app = FastAPI(
    title="YOLO26 Model Serving API",
    description="Object detection using YOLO26 ONNX",
    version="1.0"
)

# Load ONNX model
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"ONNX model not found: {MODEL_PATH}"
    )

print("Loading YOLO26 ONNX model...")

model = YOLO(str(MODEL_PATH), task="detect")

print("YOLO26 ONNX model loaded successfully!")


@app.get("/")
def home():
    return {
        "message": "YOLO26 Model Serving API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "YOLO26n",
        "format": "ONNX"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    # Check image file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )

    try:
        # Read uploaded image
        image_bytes = await file.read()
        image = Image.open(
            BytesIO(image_bytes)
        ).convert("RGB")

        # Run object detection
        results = model.predict(
            source=image,
            imgsz=640,
            conf=0.25,
            device="cpu",
            verbose=False
        )

        result = results[0]

        detections = []

        # Extract detected objects
        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls[0].item())

                detections.append({
                    "class_id": class_id,
                    "class_name": result.names[class_id],
                    "confidence": round(
                        float(box.conf[0].item()), 4
                    ),
                    "bounding_box": [
                        round(float(value), 2)
                        for value in box.xyxy[0].tolist()
                    ]
                })

        return {
            "filename": file.filename,
            "image_width": image.width,
            "image_height": image.height,
            "total_detections": len(detections),
            "detections": detections
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
