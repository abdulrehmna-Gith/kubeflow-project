
import json
import shutil
from pathlib import Path

from ultralytics import YOLO


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_MODEL = (
    PROJECT_ROOT / "models" / "yolo26n_coco128_best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_MODEL = OUTPUT_DIR / "yolo26n_coco128_best.onnx"
INFO_FILE = OUTPUT_DIR / "optimization_info.json"


def main():
    print("Starting model optimization...")

    # 1. Check that the trained model exists
    if not SOURCE_MODEL.exists():
        raise FileNotFoundError(
            f"Trained model not found: {SOURCE_MODEL}"
        )

    print(f"Loading trained model: {SOURCE_MODEL}")

    # 2. Load the trained YOLO model
    model = YOLO(str(SOURCE_MODEL))

    # 3. Export the model to ONNX
    print("Exporting model to ONNX format...")

    exported_path = model.export(
        format="onnx",
        imgsz=640,
        dynamic=False,
        simplify=True,
        device="cpu"
    )

    exported_path = Path(exported_path)

    # 4. Verify that the exported model exists
    if not exported_path.exists():
        raise FileNotFoundError(
            f"ONNX export was not found: {exported_path}"
        )

    # 5. Copy the ONNX model to our optimization folder
    if exported_path.resolve() != OUTPUT_MODEL.resolve():
        shutil.copy2(exported_path, OUTPUT_MODEL)

    # 6. Calculate model sizes
    original_size_mb = SOURCE_MODEL.stat().st_size / (1024 * 1024)
    optimized_size_mb = OUTPUT_MODEL.stat().st_size / (1024 * 1024)

    # 7. Save optimization information
    optimization_info = {
        "source_model": str(SOURCE_MODEL),
        "optimized_model": str(OUTPUT_MODEL),
        "format": "ONNX",
        "image_size": 640,
        "dynamic_input": False,
        "simplified": True,
        "device": "CPU",
        "original_size_mb": round(original_size_mb, 2),
        "optimized_size_mb": round(optimized_size_mb, 2)
    }

    with open(INFO_FILE, "w") as file:
        json.dump(optimization_info, file, indent=4)

    print("\nModel optimization completed successfully!")
    print(f"Original model size: {original_size_mb:.2f} MB")
    print(f"ONNX model size: {optimized_size_mb:.2f} MB")
    print(f"Optimized model: {OUTPUT_MODEL}")
    print(f"Information saved: {INFO_FILE}")


if __name__ == "__main__":
    main()
