# VeggieVision_Detection_Code

https://github.com/user-attachments/assets/7fdc40d7-3341-46c6-8d8a-6a5b74b37168

## Links

* **Dataset (Zip File):** [Google Drive Link](https://drive.google.com/file/d/1gh45LaWQiIdfAwP6mkHSnf4-eVR3eRey/view?usp=sharing)
* **Pretrained Weights:** [Google Drive Link](https://drive.google.com/drive/folders/1A0hT3nWyXW8v9qHJvXDUGcWWZQl4K9uU?usp=sharing)

## Setup

Follow these steps to set up the necessary environment.

1.  **Prerequisites:**
    * Python 3.8 or higher.
    * `pip` package installer.
    * Access to an NVIDIA GPU for significantly faster training and inference.

2.  **Install Dependencies:**
    Install all required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

## Training (`train.py`)

The `train.py` script handles extracting the dataset from a provided zip file and training the YOLOv8 model.

1.  **Prepare your Dataset:** Ensure you have your dataset prepared in YOLO format and zipped (e.g., `my_dataset.zip`). This zip file should typically contain a structure with `data.yaml` and `train/`, `valid/`, `test/` subdirectories inside.

2.  **Run Training:**
    Execute the script from your project directory, providing the path to your dataset zip file. The script will unzip the dataset into the `dataset/` folder (if it doesn't exist or is empty) and save the best weights to `runs/weights/best.pt`.

    ```bash
    python train.py \
        --dataset-zip /path/to/your/dataset.zip \
        --output-dir . \
        --model-name yolov8s.pt \
        --epochs 50 \
        --batch-size 16 \
        --device 0
    ```

    * **`--dataset-zip`**: **(Required)** Path to your dataset zip file.
    * **`--output-dir`**: (Optional, defaults to `.`) Base directory for outputs (`dataset/`, `weights/`, `training_runs/`).
    * **`--model-name`**: (Optional) Base YOLOv8 model (e.g., `yolov8n.pt`, `yolov8s.pt`).
    * **`--epochs`**: (Optional) Number of training epochs.
    * **`--batch-size`**: (Optional) Training batch size (adjust based on GPU memory).
    * **`--device`**: (Optional) GPU device ID (e.g., `0`) or `cpu`.

    Training logs and intermediate files will be saved under `training_runs/`. The best performing model weights will be copied to `./weights/best.pt` (assuming `--output-dir .`).

## Inference (`inference.py`)

The `inference.py` script uses the trained weights (`best.pt`) to run object detection on new images or videos.

1.  **Ensure Weights Exist:** Make sure you have successfully run `train.py` and the `runs/weights/best.pt` file exists.

2.  **Prepare Inference Data:** Gather the images or videos you want to process in a directory or note the path to a single file.

3.  **Run Inference:**
    Execute the script, providing the path to the weights and the source data.

    * **Example for Image(s):**
        ```bash
        python inference.py \
            --weights ./weights/best.pt \
            --source /path/to/your/inference_images/ \
            --output-dir ./inference_output \
            --conf-thres 0.4
        ```

    * **Example for a Video File:**
        ```bash
        python inference.py \
            --weights ./weights/best.pt \
            --source /path/to/your/input_video.mp4 \
            --output-dir ./inference_output_video \
            --conf-thres 0.4
        ```

    * **`--weights`**: **(Required)** Path to the trained `.pt` weights file.
    * **`--source`**: **(Required)** Path to the input image, video file, directory, or pattern (e.g., `'images/*.jpg'`).
    * **`--output-dir`**: (Optional) Directory where results (annotated images/videos, label files) will be saved (inside a `predictions` subdirectory).
    * **`--conf-thres`**: (Optional) Confidence threshold for detections.

    Inference results will be saved in the specified output directory under a subdirectory named `predictions`.

## Utilities (`utils.py`)

This file contains helper functions used by `train.py` for tasks like unzipping datasets and finding configuration files like data.yaml.

## Licenses and Acknowledgements

This project utilizes the following libraries and services:

* **Ultralytics YOLOv8:** The core object detection model and library used in this project.
    * License: AGPL-3.0 ([License Text](https://www.gnu.org/licenses/agpl-3.0.en.html)). Note that use in distributed or network-accessible applications may require open-sourcing your project under AGPL-3.0 unless an Enterprise License is obtained from Ultralytics ([Ultralytics Licensing](https://www.ultralytics.com/license)).
    * Website: [https://ultralytics.com/](https://ultralytics.com/)

* **Roboflow:** Used for dataset labeling, management, and augmentation.
    * Website: [https://roboflow.com/](https://roboflow.com/)
