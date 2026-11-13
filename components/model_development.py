
from pathlib import Path
import json
import shutil

from ultralytics import YOLO


# Project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def develop_model(weights_path=None):
    """
    Load the pretrained YOLO26 Nano model,
    store its weights, and save model information.
    """

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    destination = models_dir / "yolo26n.pt"

    # Find pretrained weights
    if weights_path is None:
        candidates = [
            destination,
            PROJECT_ROOT / "yolo26n.pt"
        ]

        source = next(
            (path for path in candidates if path.is_file()),
            None
        )
    else:
        source = Path(weights_path)

        if not source.is_absolute():
            source = PROJECT_ROOT / source

        if not source.is_file():
            raise FileNotFoundError(
                f"Model weights not found: {source}"
            )

    if source is None:
        raise FileNotFoundError(
            "YOLO26 Nano weights not found. "
            "Expected yolo26n.pt in the project root or models folder."
        )

    # Copy pretrained weights into models directory
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)

    print("Loading pretrained YOLO26 Nano model...")

    # Load model
    model = YOLO(str(destination))

    # Get model information
    model_names = model.names

    if isinstance(model_names, dict):
        class_names = {
            str(key): value
            for key, value in model_names.items()
        }
    else:
        class_names = {
            str(index): name
            for index, name in enumerate(model_names)
        }

    model_info = {
        "model_name": "YOLO26 Nano",
        "model_file": str(destination),
        "task": model.task,
        "number_of_classes": len(class_names),
        "class_names": class_names,
        "pretrained": True
    }

    # Verify object detection task
    if model.task != "detect":
        raise ValueError(
            f"Expected object detection model, got: {model.task}"
        )

    # Save model information
    info_path = models_dir / "model_info.json"

    with info_path.open("w") as file:
        json.dump(model_info, file, indent=4)

    print("\nModel development completed successfully.")
    print(f"Model: {model_info['model_name']}")
    print(f"Task: {model_info['task']}")
    print(f"Classes: {model_info['number_of_classes']}")
    print(f"Weights saved at: {destination}")
    print(f"Model information saved at: {info_path}")

    return model_info


if __name__ == "__main__":
    develop_model()
