
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PT_MODEL = (
    PROJECT_ROOT / "models" / "yolo26n_coco128_best.pt"
)

ONNX_MODEL = (
    PROJECT_ROOT
    / "outputs"
    / "optimization"
    / "yolo26n_coco128_best.onnx"
)

REGISTRY_DIR = (
    PROJECT_ROOT / "model_registry" / "YOLO26n" / "v1"
)


def main():
    print("Starting Model Registry component...")

    # 1. Check that both model files exist
    if not PT_MODEL.exists():
        raise FileNotFoundError(
            f"PyTorch model not found: {PT_MODEL}"
        )

    if not ONNX_MODEL.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {ONNX_MODEL}"
        )

    # 2. Create the registry directory
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Copy both models into the registry
    registered_pt = REGISTRY_DIR / "model.pt"
    registered_onnx = REGISTRY_DIR / "model.onnx"

    shutil.copy2(PT_MODEL, registered_pt)
    shutil.copy2(ONNX_MODEL, registered_onnx)

    print("Both models copied into the registry.")

    # 4. Read training information if available
    training_info_file = (
        PROJECT_ROOT
        / "outputs"
        / "training"
        / "training_info.json"
    )

    training_info = {}

    if training_info_file.exists():
        with open(training_info_file, "r") as file:
            training_info = json.load(file)

    # 5. Read optimization information
    optimization_info_file = (
        PROJECT_ROOT
        / "outputs"
        / "optimization"
        / "optimization_info.json"
    )

    with open(optimization_info_file, "r") as file:
        optimization_info = json.load(file)

    # 6. Create registry metadata
    registry_info = {
        "model_name": "YOLO26n",
        "version": "v1",
        "task": "object_detection",
        "dataset": "COCO128",
        "status": "registered",
        "registered_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "artifacts": {
            "pytorch_model": "model.pt",
            "onnx_model": "model.onnx"
        },
        "training_information": training_info,
        "optimization_information": optimization_info
    }

    # 7. Save registry information
    info_file = REGISTRY_DIR / "registry_info.json"

    with open(info_file, "w") as file:
        json.dump(registry_info, file, indent=4)

    print("\nModel Registry completed successfully!")
    print(f"Model name: YOLO26n")
    print(f"Version: v1")
    print(f"PyTorch model: {registered_pt}")
    print(f"ONNX model: {registered_onnx}")
    print(f"Registry information: {info_file}")


if __name__ == "__main__":
    main()
