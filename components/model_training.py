
from pathlib import Path
import json
import shutil
import yaml

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def train_model(
    epochs=5,
    image_size=640,
    batch_size=4
):
    """Fine-tune pretrained YOLO26 Nano on COCO128."""

    models_dir = PROJECT_ROOT / "models"
    outputs_dir = PROJECT_ROOT / "outputs" / "training"

    pretrained_weights = models_dir / "yolo26n.pt"
    config_path = PROJECT_ROOT / "configs" / "data.yaml"

    # Check required files
    if not pretrained_weights.is_file():
        raise FileNotFoundError(
            f"Pretrained weights not found: {pretrained_weights}"
        )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {config_path}"
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Read dataset configuration
    with config_path.open("r") as file:
        data_config = yaml.safe_load(file)

    # Resolve dataset path
    dataset_root = (
        PROJECT_ROOT / data_config["path"]
    ).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_root}"
        )

    # Verify training and validation directories
    for split in ["train", "val"]:
        split_dir = dataset_root / data_config[split]

        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"{split} directory not found: {split_dir}"
            )

    # Create a resolved YAML file for Ultralytics
    data_config["path"] = str(dataset_root)

    resolved_config = outputs_dir / "data_resolved.yaml"

    with resolved_config.open("w") as file:
        yaml.safe_dump(data_config, file, sort_keys=False)

    # Load pretrained model
    print("Loading pretrained YOLO26 Nano...")
    model = YOLO(str(pretrained_weights))

    print("\nStarting YOLO26 training...")
    print(f"Epochs: {epochs}")
    print(f"Image size: {image_size}")
    print(f"Batch size: {batch_size}")
    print("Device: CPU")

    # Fine-tune the model
    model.train(
        data=str(resolved_config),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device="cpu",
        workers=0,
        amp=False,
        project=str(outputs_dir),
        name="yolo26n_coco128",
        exist_ok=True,
        plots=False,
        verbose=True
    )

    # Get training directory from YOLO trainer
    training_dir = Path(model.trainer.save_dir)

    best_weights = training_dir / "weights" / "best.pt"
    last_weights = training_dir / "weights" / "last.pt"

    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Best model weights not found: {best_weights}"
        )

    # Copy best model into models directory
    final_model = models_dir / "yolo26n_coco128_best.pt"
    shutil.copy2(best_weights, final_model)

    # Collect training metrics
    metrics = {}

    results = model.trainer.metrics

    if hasattr(results, "results_dict"):
        for key, value in results.results_dict.items():
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                metrics[key] = str(value)

    # Save training information
    training_info = {
        "model": "YOLO26 Nano",
        "dataset": "COCO128",
        "epochs": epochs,
        "image_size": image_size,
        "batch_size": batch_size,
        "device": "cpu",
        "best_weights": str(final_model),
        "last_weights": str(last_weights),
        "training_directory": str(training_dir),
        "metrics": metrics
    }

    info_path = outputs_dir / "training_info.json"

    with info_path.open("w") as file:
        json.dump(training_info, file, indent=4)

    print("\nTraining completed successfully!")
    print(f"Best model: {final_model}")
    print(f"Training information: {info_path}")
    print(f"Metrics: {metrics}")

    return training_info


if __name__ == "__main__":
    train_model()
