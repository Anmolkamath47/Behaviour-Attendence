"""
ClassroomAI - Custom YOLO11 Training Pipeline
Prepares dataset YAML configuration and provides automated training scripts
for YOLO11s/YOLO11n on classroom datasets (Google Colab / Kaggle / Local).
"""

import os
import sys
import yaml
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def create_dataset_yaml(output_path: str = None) -> str:
    """Generate YOLO dataset configuration file for classroom object detection."""
    dataset_dir = PROJECT_ROOT / "datasets"
    raw_dir = dataset_dir / "raw"
    processed_dir = dataset_dir / "processed"

    config = {
        "path": str(processed_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "student",
            1: "mobile_phone",
            2: "notebook",
            3: "raised_hand",
            4: "head_down"
        }
    }

    if output_path is None:
        output_path = dataset_dir / "classroom_dataset.yaml"

    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Created YOLO dataset configuration at: {output_path}")
    return str(output_path)


def run_yolo_training(
    data_yaml: str,
    model_name: str = "yolo11s.pt",
    epochs: int = 50,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "0"
):
    """
    Train custom YOLO11 model on classroom dataset.
    Suitable for Google Colab/Kaggle GPU environments.
    """
    print(f"Loading base model: {model_name}")
    model = YOLO(model_name)

    print(f"Starting training on {data_yaml} for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project=str(PROJECT_ROOT / "models" / "yolo"),
        name="custom_classroom_yolo11"
    )

    print("Training finished. Evaluating best weights...")
    val_metrics = model.val()
    print(f"Validation mAP50-95: {val_metrics.box.map}")

    # Export to ONNX for fast CPU inference
    print("Exporting trained model to ONNX for CPU inference...")
    onnx_file = model.export(format="onnx")
    print(f"Exported ONNX model: {onnx_file}")
    return results


if __name__ == "__main__":
    yaml_path = create_dataset_yaml()
    print(f"Dataset YAML ready for Colab/Kaggle execution: {yaml_path}")
