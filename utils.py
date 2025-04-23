"""
Utility functions for dataset setup and file management.
Includes functions for unzipping datasets and finding .yaml files.
"""

import os
import zipfile
import logging


# Configure logging for utility functions if not configured by main script
if not logging.getLogger().hasHandlers():
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Dataset Setup Functions
def unzip_dataset(dataset_zip_path: str, extract_dir: str):
    """
    Checks for and extracts the dataset from a zip file.

    Args:
        dataset_zip_path: Path to the dataset zip file.
        extract_dir: Directory where the dataset should be extracted.

    Raises:
        FileNotFoundError: If the zip file is not found.
        zipfile.BadZipFile: If the zip file is corrupted.
        Exception: For other errors during extraction.
    """
    logging.info(f"Checking dataset zip file: {dataset_zip_path}")
    if not os.path.exists(dataset_zip_path):
        raise FileNotFoundError(f"Dataset zip file not found at: {dataset_zip_path}")

    # Ensure extraction directory exists
    os.makedirs(extract_dir, exist_ok=True)
    logging.info(f"Extracting dataset to: {extract_dir}")

    try:
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        logging.info("Dataset extraction complete.")
    except zipfile.BadZipFile as e:
        logging.error(f"Failed to extract zip file: {e}")
        raise
    except Exception as e:
        logging.error(f"An error occurred during extraction: {e}")
        raise


def find_yaml_path(base_extract_dir: str) -> str:
    """
    Finds the data.yaml file within the extracted dataset directory structure.

    Searches common locations (e.g., base_extract_dir/dataset_name/data.yaml).

    Args:
        base_extract_dir: The directory where the zip file was extracted.

    Returns:
        Absolute path to the data.yaml file.

    Raises:
        FileNotFoundError: If data.yaml cannot be found.
    """
    logging.info(f"Searching for data.yaml in: {base_extract_dir}")
    data_yaml_path = None

    extracted_items = os.listdir(base_extract_dir)
    if not extracted_items:
        raise FileNotFoundError(f"No items found in extraction directory: {base_extract_dir}")

    # Find the directory created by unzipping
    dataset_base_dir_name = None
    for item in extracted_items:
        potential_dir = os.path.join(base_extract_dir, item)
        if os.path.isdir(potential_dir):
            dataset_base_dir_name = item
            logging.info(f"Identified dataset base directory: {dataset_base_dir_name}")
            break

    if not dataset_base_dir_name:
         raise FileNotFoundError(f"Could not find dataset base directory within {base_extract_dir}")

    # Check common location first
    potential_yaml_path = os.path.join(base_extract_dir, dataset_base_dir_name, 'data.yaml')
    if os.path.exists(potential_yaml_path):
        data_yaml_path = os.path.abspath(potential_yaml_path)
        logging.info(f"Found data.yaml at common location: {data_yaml_path}")
    else:
        # Alternative search if not directly in the base dir
        logging.info("data.yaml not in common location, searching subdirectories...")
        for root, _, files_in_dir in os.walk(os.path.join(base_extract_dir, dataset_base_dir_name)):
            if 'data.yaml' in files_in_dir:
                data_yaml_path = os.path.abspath(os.path.join(root, 'data.yaml'))
                logging.info(f"Found data.yaml deeper at: {data_yaml_path}")
                break

    if not data_yaml_path:
        raise FileNotFoundError(f"data.yaml not found within the extracted dataset at {base_extract_dir}")

    return data_yaml_path