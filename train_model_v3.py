import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
# We don't need class_weight anymore
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
# This is for your 10k+ Fundus Dataset
DATA_DIR = "D:/BICS/insightweb/data/images/preprocessed_images/" 
LABEL_FILE = "D:/BICS/insightweb/data/images/full_df.csv"
IMG_SIZE = (224, 224) # B0 default size
BATCH_SIZE = 32
EPOCHS = 30 # Early stopping will handle this
AUTOTUNE = tf.data.AUTOTUNE
NEW_MODEL_NAME = "cataract_model_v3.keras" # Final model
# ---------------

def load_data(label_file):
    """
    Loads the CSV, creates binary labels, and VERIFIES files exist.
    """
    df = pd.read_csv(label_file)
    print("Parsing image-level labels...")
    
    left_df = df[['Left-Fundus', 'Left-Diagnostic Keywords']].copy()
    left_df.rename(columns={'Left-Fundus': 'filename'}, inplace=True)
    left_df['Cataract'] = left_df['Left-Diagnostic Keywords'].apply(lambda x: 1 if 'cataract' in str(x) else 0)
    
    right_df = df[['Right-Fundus', 'Right-Diagnostic Keywords']].copy()
    right_df.rename(columns={'Right-Fundus': 'filename'}, inplace=True)
    right_df['Cataract'] = right_df['Right-Diagnostic Keywords'].apply(lambda x: 1 if 'cataract' in str(x) else 0)
    
    all_images_df = pd.concat([
        left_df[['filename', 'Cataract']], 
        right_df[['filename', 'Cataract']]
    ]).dropna()
    
    all_images_df = all_images_df[all_images_df['filename'] != 'db_no_fovea.jpg'].reset_index(drop=True)
    all_images_df['filepath'] = all_images_df['filename'].apply(lambda x: os.path.join(DATA_DIR, x))
    
    print(f"Verifying {len(all_images_df)} filepaths...")
    all_images_df['file_exists'] = all_images_df['filepath'].apply(os.path.exists)
    removed_count = len(all_images_df) - all_images_df['file_exists'].sum()
    if removed_count > 0:
        print(f"Warning: Removed {removed_count} missing file references.")
        all_images_df = all_images_df[all_images_df['file_exists'] == True].copy()
        
    all_images_df = all_images_df.drop(columns=['file_exists'])
    all_images_df['Cataract'] = all_images_df['Cataract'].astype(int)
    print("Label parsing and file verification complete.")
    return all_images_df

# --- Data Augmentation Layer ---
def data_augmentation(img_size=IMG_SIZE):
    return models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

# --- Build Model (EfficientNetV2B0) ---
def build_model(img_size=IMG_SIZE):
    """
    Builds an EfficientNetV2B0 model.
    """
    inputs = layers.Input(shape=(*img_size, 3))
    
    # Augmentation
    x = data_augmentation(img_size)(inputs)
    
    # Base model
    # EfficientNetV2 *includes* the preprocessing (rescaling) by default
    base_model = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_tensor=x, # Pass augmented images directly
        include_preprocessing=True # This handles the scaling
    )
    base_model.trainable = False # Freeze base model

    # Custom head
    x = layers.GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, name="top_dropout")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x) # Binary classification

    model = models.Model(inputs, outputs)

    # --- COMPILE WITH FOCAL LOSS ---
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryFocalCrossentropy(from_logits=False, gamma=2.0), # Use the built-in Keras 3 loss CLASS 
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]
    )
    return model

# --- tf.data Helper Functions ---
def parse_image(filename, label):
    """Loads and resizes a single image file. NO SCALING."""
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label # Return image in [0, 255] range

def create_dataset(df, shuffle=True):
    ds = tf.data.Dataset.from_tensor_slices((df['filepath'].values, df['Cataract'].values))
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

# --- MAIN TRAINING SCRIPT ---
if __name__ == "__main__":
    
    all_df = load_data(LABEL_FILE)
    if all_df.empty: exit()

    print(f"\nTotal images found and verified: {len(all_df)}")
    print(all_df['Cataract'].value_counts())

    train_df, val_df = train_test_split(all_df, test_size=0.2, random_state=42, stratify=all_df['Cataract'])
    
    print(f"\nTraining images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    
    # --- NO CLASS WEIGHTS NEEDED ---
    print("\nUsing Focal Loss. Class weights are not required.")

    print("\nCreating tf.data.Dataset pipelines...")
    train_ds = create_dataset(train_df, shuffle=True)
    val_ds = create_dataset(val_df, shuffle=False)
    print("Pipelines created.")
    
    model = build_model(img_size=IMG_SIZE)
    model.summary()

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        NEW_MODEL_NAME, 
        monitor='val_auc', 
        save_best_only=True, 
        mode='max',
        verbose=1
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', 
        patience=10, 
        mode='max',
        restore_best_weights=True,
        verbose=1
    )

    print(f"\n--- STARTING MODEL TRAINING (EfficientNetV2B0 + Focal Loss) ---")
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        # --- DO NOT PASS class_weight ---
        callbacks=[checkpoint, early_stopping]
    )
    
    print("\n--- TRAINING COMPLETE ---")
    print(f"The best model has been saved as '{NEW_MODEL_NAME}'")
    
    # Plot training history
    try:
        history_df = pd.DataFrame(history.history)
        plt.figure(figsize=(12, 8))
        plt.plot(history_df['auc'], label='Train AUC')
        plt.plot(history_df['val_auc'], label='Validation AUC')
        plt.title('Model AUC Over Epochs (EfficientNetV2 + Focal Loss)')
        plt.legend()
        plt.savefig('training_history_v3.png')
        print("Saved training history plot to 'training_history_v3.png'")
    except Exception as e:
        print(f"Could not plot history: {e}")