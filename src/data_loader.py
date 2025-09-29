import os
import glob
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(image_dir, mask_dir, img_height=256, img_width=256):
    """
    Loads and preprocesses data from parallel 'images' and 'masks' folders
    that contain identical subfolder structures and extensionless filenames.
    """
    images = []
    masks = []

    # --- START OF FIX ---
    # Find all files recursively in the image directory, regardless of extension
    image_paths = []
    for root, _, files in os.walk(image_dir):
        for file in files:
            # Add files that don't have a common image extension (like desktop.ini)
            if not file.lower().endswith(('.ini', '.db')):
                image_paths.append(os.path.join(root, file))
    # --- END OF FIX ---
    
    for img_path in image_paths:
        try:
            # --- Path Matching Logic ---
            # Get the relative path of the image (e.g., '001/01_L')
            relative_path = os.path.relpath(img_path, image_dir)
            
            # The mask path is the same relative path, just in the mask directory
            mask_path = os.path.join(mask_dir, relative_path)
            # --- End of Logic ---

            if not os.path.exists(mask_path):
                print(f"Warning: Mask not found for image {img_path}. Looked for mask at {mask_path}. Skipping.")
                continue

            # Load image and mask in grayscale
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Check if images loaded correctly
            if image is None or mask is None:
                print(f"Warning: Failed to load image or mask for {img_path}. Skipping.")
                continue

            # Resize image and mask
            image = cv2.resize(image, (img_width, img_height))
            mask = cv2.resize(mask, (img_width, img_height))
            
            # Normalize pixel values to the [0, 1] range
            image = image / 255.0
            mask = mask / 255.0
            
            images.append(image)
            masks.append(mask)
        except Exception as e:
            print(f"Error loading file pair for {img_path}: {e}")
            
    # Convert lists to numpy arrays and add the channel dimension for the model
    images = np.array(images)[..., np.newaxis]
    masks = np.array(masks)[..., np.newaxis]
    
    if len(images) == 0:
        raise ValueError("No valid image-mask pairs were found. Please verify your folder structure and file paths.")

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, masks, test_size=0.2, random_state=42
    )

    return (X_train, y_train), (X_val, y_val)