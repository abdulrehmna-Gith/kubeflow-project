
from pathlib import Path
import yaml


def validate_labels(label_dir: Path, num_classes: int) -> int:
    """Validate existing YOLO label files. Missing files mean no annotations."""

    label_files = list(label_dir.glob("*.txt"))
    annotation_count = 0

    for label_file in label_files:
        with label_file.open("r") as file:
            for line_number, line in enumerate(file, start=1):
                values = line.strip().split()

                if not values:
                    continue

                if len(values) != 5:
                    raise ValueError(
                        f"Invalid annotation in {label_file}, "
                        f"line {line_number}: expected 5 values."
                    )

                try:
                    class_id = int(values[0])
                    coordinates = [float(value) for value in values[1:]]
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid values in {label_file}, line {line_number}."
                    ) from exc

                if not 0 <= class_id < num_classes:
                    raise ValueError(
                        f"Invalid class ID {class_id} in {label_file}."
                    )

                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    raise ValueError(
                        f"Coordinates outside the 0-1 range in {label_file}."
                    )

                annotation_count += 1

    return annotation_count


def prepare_data(config_path="configs/data.yaml"):
    """Check dataset configuration, images, and YOLO annotations."""

    config_path = Path(config_path).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {config_path}")

    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    required_keys = ["path", "train", "val", "nc", "names"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing '{key}' in {config_path}")

    num_classes = int(config["nc"])
    names = config["names"]

    if num_classes <= 0 or len(names) != num_classes:
        raise ValueError("Class count does not match the names list.")

    dataset_root = (config_path.parent.parent / config["path"]).resolve()

    results = {}

    for split in ["train", "val"]:
        image_dir = (dataset_root / config[split]).resolve()

        if not image_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Convert images/... path into labels/... path.
        relative_path = Path(config[split])

        if relative_path.parts[0] != "images":
            raise ValueError(
                f"Expected '{split}' path to start with 'images/'."
            )

        label_relative_path = Path("labels").joinpath(*relative_path.parts[1:])
        label_dir = dataset_root / label_relative_path

        if not label_dir.is_dir():
            raise FileNotFoundError(f"Label directory not found: {label_dir}")

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        images = [
            path for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        ]

        if not images:
            raise ValueError(f"No images found in {image_dir}")

        missing_labels = [
            image.name for image in images
            if not (label_dir / f"{image.stem}.txt").exists()
        ]

        annotation_count = validate_labels(label_dir, num_classes)

        results[split] = {
            "images": len(images),
            "label_files": len(list(label_dir.glob("*.txt"))),
            "images_without_label_files": len(missing_labels),
            "annotations": annotation_count,
        }

    print("Data preparation completed successfully.")
    print(f"Dataset: {dataset_root}")
    print(f"Classes: {num_classes}")
    print(f"Training: {results['train']}")
    print(f"Validation: {results['val']}")

    return results


if __name__ == "__main__":
    prepare_data()
