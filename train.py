"""
Downloads and prepares a dataset from a zip file, then initiates training.
Saves the best performing weights to a specified directory.

Example usage:
python train.py \
    --dataset-zip /your/path/to/dataset \
    --output-dir ./yolov8s_finetune_run \
    --model-name yolov8s.pt \
    --epochs 50 \
    --batch-size 64 \
    --img-size 640 \
    --workers 4 \
    --device 0
"""

import os
import shutil
import argparse
import logging
import yaml
from ultralytics import YOLO
import utils


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def run_training(
    data_yaml_path: str,
    output_dir: str,
    model_name: str = 'yolov8s.pt',
    epochs: int = 50,
    batch_size: int = 64, # Lower this if you run out of memory
    img_size: int = 640,
    workers: int = 4,
    device: int = 0,
    project_name: str = 'training_runs'
) -> str:
    """
    Runs YOLOv8 training using the specified parameters.

    Args:
        data_yaml_path: Absolute path to the data.yaml file.
        output_dir: Base directory where training runs and final weights will be saved.
        model_name: Pretrained model to use (e.g., 'yolov8s.pt').
        epochs: Number of training epochs.
        batch_size: Training batch size.
        img_size: Input image size.
        workers: Number of dataloader workers.
        device: GPU device ID (0 for first GPU, 'cpu' for CPU).
        project_name: Name for the parent directory holding training runs.

    Returns:
        Path to the saved best performing weights (`best.pt`).

    Raises:
        FileNotFoundError: If the best weights file cannot be found after training.
        Exception: For errors during the training process.
    """
    logging.info(f"Starting training with model: {model_name}, epochs: {epochs}")
    weights_dir = os.path.join(output_dir, 'weights')
    runs_dir = os.path.join(output_dir, project_name) # Directory for Ultralytics output

    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(runs_dir, exist_ok=True)

    try:
        # Load the model
        model = YOLO(model_name)

        # Train the model
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            workers=workers,
            device=device,
            project=runs_dir, # Tell Ultralytics where to save runs
            name='yolov8_finetune' # Specific run name
            # Add other parameters as needed
        )
        logging.info("Training complete.")

        # Copy best weights
        run_save_dir = results.save_dir # Get the exact save directory from results
        logging.info(f"Training run saved to: {run_save_dir}")

        source_best_weights = os.path.join(run_save_dir, 'weights', 'best.pt')
        dest_best_weights = os.path.join(weights_dir, 'best.pt')

        if os.path.exists(source_best_weights):
            shutil.copy2(source_best_weights, dest_best_weights)
            logging.info(f"Copied best weights to: {dest_best_weights}")
            return dest_best_weights
        else:
            raise FileNotFoundError(f"Could not find best.pt in {os.path.join(run_save_dir, 'weights')}")

    except Exception as e:
        logging.error(f"An error occurred during training: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv8 model.")
    parser.add_argument(
        "--dataset-zip",
        type=str,
        required=True,
        help="Path to the dataset zip file (e.g., from Google Drive)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Base directory for output (weights, data, training_runs)."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="yolov8s.pt",
        help="Pretrained model name (e.g., yolov8s.pt)."
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers.")
    parser.add_argument("--device", type=str, default="0", help="Device ID (e.g., 0, 1) or 'cpu'.")

    args = parser.parse_args()

    # Define output subdirectories relative to the base output directory
    data_extract_dir = os.path.join(args.output_dir, 'data')

    if not os.path.exists(data_extract_dir) or not os.listdir(data_extract_dir):
        logging.info(f"Dataset directory '{data_extract_dir}' not found or appears empty. Unzipping...")
        # Call unzip only if needed
        utils.unzip_dataset(args.dataset_zip, data_extract_dir)
    else:
        logging.info(f"Dataset directory '{data_extract_dir}' already exists. Skipping unzip.")

    # Find .yaml path
    yaml_path = utils.find_yaml_path(data_extract_dir)

    # Convert device arg if it's a digit
    train_device = int(args.device) if args.device.isdigit() else args.device

    # Run Training
    best_weights_path = run_training(
        data_yaml_path=yaml_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        workers=args.workers,
        device=train_device
    )
    logging.info(f"Training finished successfully. Best weights at: {best_weights_path}")