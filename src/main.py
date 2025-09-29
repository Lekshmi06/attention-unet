import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os # <-- Import the os library

# Import our custom modules
from data_loader import load_data
from model import build_attention_unet

# --- Configuration ---
# Construct paths relative to the script's location
# This is a robust way to handle the new 'src' folder structure
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the 'src' folder path
PROJECT_ROOT = os.path.dirname(ROOT_DIR) # Goes up one level to the project root

IMAGE_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "IITD", "images")
MASK_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "IITD", "masks")

IMG_HEIGHT = 256
IMG_WIDTH = 256
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, 1)
EPOCHS = 25
BATCH_SIZE = 8

# --- Custom Loss and Metrics (No changes here) ---
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1 - dice_coefficient(y_true, y_pred)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and prepare data
    print("Loading and preprocessing data...")
    print(f"Looking for images in: {IMAGE_DIRECTORY}")
    print(f"Looking for masks in: {MASK_DIRECTORY}")
    (X_train, y_train), (X_val, y_val) = load_data(IMAGE_DIRECTORY, MASK_DIRECTORY, IMG_HEIGHT, IMG_WIDTH)
    print(f"Data loaded: {len(X_train)} training samples, {len(X_val)} validation samples.")

    # 2. Build the model
    print("Building Attention U-Net model...")
    model = build_attention_unet(input_shape=INPUT_SHAPE)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=dice_loss, 
        metrics=[dice_coefficient, 'binary_accuracy']
    )
    model.summary()
    
    # 3. Set up callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint("attention_unet_best.keras", save_best_only=True, monitor='val_dice_coefficient', mode='max'),
        tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_dice_coefficient', mode='max', restore_best_weights=True)
    ]
    
    # 4. Train the model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )
    
    # 5. Visualize a prediction
    print("\nVisualizing a sample prediction...")
    best_model = tf.keras.models.load_model(
        "attention_unet_best.keras",
        custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient}
    )
    
    idx = np.random.randint(0, len(X_val))
    sample_image = X_val[idx]
    sample_mask = y_val[idx]
    predicted_mask = best_model.predict(np.expand_dims(sample_image, axis=0))[0]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(sample_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sample_mask, cmap='gray')
    plt.title("Ground Truth Mask")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(predicted_mask > 0.5, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("prediction_sample.png")
    plt.show()

    print("\nDone! A sample prediction has been saved as 'prediction_sample.png'")