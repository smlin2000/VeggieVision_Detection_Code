"""
Performs inference using a trained YOLOv8 model.
Loads weights and runs prediction on a specified image source file, directory, or pattern.
Saves prediction results (images, labels) to an output directory.
This can also run inference on videos, the output will be a .avi file with a corresponding labels folder.

Example usage:
python inference.py \
    --weights ./yolov8s_finetune_weights/weights/best.pt \
    --source /path/to/your/inference_images/ \
    --output-dir ./inference_output \
    --conf-thres 0.4 \
    --device 0
"""

import os
import argparse
import logging
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2 # Imported for optional visualization
from IPython.display import Image, display # For use in environments like Jupyter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

"""
    Runs YOLOv8 inference on the source path using specified weights.

    Args:
        weights_path: Path to the trained model weights (.pt file).
        source_path: Path to the input image, directory, or pattern.
        output_dir: Directory to save the inference results.
        conf_thres: Confidence threshold for predictions.
        device: Device ID (e.g., '0', 'cpu') for inference.
        save_results: Whether to save predicted images/videos.
        save_txt: Whether to save prediction labels as text files.
        visualize_limit: Number of prediction images to display (0 for none).

    Raises:
        FileNotFoundError: If the weights file or source path does not exist.
        Exception: For errors during model loading or prediction.
    """
def perform_inference(
    weights_path: str,
    source_path: str,
    output_dir: str,
    conf_thres: float = 0.4,
    device: str = '0',
    save_results: bool = True,
    save_txt: bool = True,
    visualize_limit: int = 0
):
    
    logging.info(f"Starting inference.")
    logging.info(f"Weights: {weights_path}")
    logging.info(f"Source: {source_path}")
    logging.info(f"Output Dir: {output_dir}")
    logging.info(f"Confidence: {conf_thres}")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not os.path.exists(source_path):
        # Allow glob patterns, YOLO handles this, so just log warning if it's not a direct file/dir
        if '*' not in source_path and '?' not in source_path:
             logging.warning(f"Source path does not exist: {source_path}. Proceeding as it might be a URL or pattern.")
        #raise FileNotFoundError(f"Source path/directory not found: {source_path}")

    # YOLO predict creates project/name, so output_dir acts as project
    os.makedirs(output_dir, exist_ok=True)
    run_name = 'predictions' # Subdirectory name within output_dir

    try:
        # Load the trained model
        model = YOLO(weights_path)
        logging.info("Model loaded successfully.")

        # Run prediction
        results = model.predict(
            source=source_path,
            conf=conf_thres,
            device=device,
            save=save_results,
            save_txt=save_txt,
            project=output_dir,
            name=run_name,
            exist_ok=True # Overwrite previous predictions in output_dir/run_name
        )
        logging.info(f"Inference complete. Results saved to {os.path.join(output_dir, run_name)}")

        # Visualization
        if visualize_limit > 0:
            logging.info(f"Displaying first {visualize_limit} prediction results...")
            prediction_dir = os.path.join(output_dir, run_name)
            count = 0
            # Check if prediction directory exists and has files
            if os.path.isdir(prediction_dir):
                 for item in sorted(os.listdir(prediction_dir)):
                      if item.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                           img_path = os.path.join(prediction_dir, item)
                           try:
                               # Attempt to display using IPython if available
                               display(Image(filename=img_path))
                               count += 1
                               if count >= visualize_limit:
                                    break
                           except NameError:
                                logging.warning("IPython display not available. Skipping visualization.")
                                break # Stop trying to visualize
                           except Exception as e:
                                logging.warning(f"Could not display {img_path}: {e}")

            else:
                 logging.warning(f"Prediction directory not found or empty: {prediction_dir}")


    except Exception as e:
        logging.error(f"An error occurred during inference: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 inference.")
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the trained model weights file (best.pt)."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to input image, directory, or glob pattern."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="inference_runs",
        help="Directory to save inference results."
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.4,
        help="Confidence threshold for predictions."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device ID (e.g., 0, 1) or 'cpu'."
    )
    parser.add_argument(
        "--visualize-limit",
        type=int,
        default=0,
        help="Number of prediction images to display (requires IPython environment)."
    )

    args = parser.parse_args()


    perform_inference(
        weights_path=args.weights,
        source_path=args.source,
        output_dir=args.output_dir,
        conf_thres=args.conf_thres,
        device=args.device,
        visualize_limit=args.visualize_limit
    )
    logging.info("Inference script finished successfully.")

