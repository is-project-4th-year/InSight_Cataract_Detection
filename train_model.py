# train_model.py
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import os
import matplotlib.pyplot as plt

# --- CONFIG ---
DATA_DIR = "D:/BICS/insightweb/data/images/preprocessed_images/" 
LABEL_FILE = "D:/BICS/insightweb/data/images/full_df.csv"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30 # Set to 30 as you requested
AUTOTUNE = tf.data.AUTOTUNE # For tf.data pipeline
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
    
    # Create the full image filepath
    all_images_df['filepath'] = all_images_df['filename'].apply(lambda x: os.path.join(DATA_DIR, x))
    
    # --- NEW: VERIFICATION STEP ---
    print(f"Verifying {len(all_images_df)} filepaths...")
    
    # Check if files exist
    all_images_df['file_exists'] = all_images_df['filepath'].apply(lambda x: os.path.exists(x))
    
    # Filter out missing files
    original_count = len(all_images_df)
    all_images_df = all_images_df[all_images_df['file_exists'] == True].copy()
    new_count = len(all_images_df)
    
    if original_count > new_count:
        removed_count = original_count - new_count
        print(f"Warning: Removed {removed_count} file references that were missing from the images folder.")
        
    if new_count == 0:
        print("Error: No valid image files were found. Check your DATA_DIR path.")
        return pd.DataFrame() # Return empty df
        
    # We don't need the 'file_exists' column anymore
    all_images_df = all_images_df.drop(columns=['file_exists'])
    # -------------------------------
    
    # Convert label to integer (tf.data prefers ints)
    all_images_df['Cataract'] = all_images_df['Cataract'].astype(int)
    
    print("Label parsing and file verification complete.")
    return all_images_df

# --- Keras 3 Data Augmentation ---
def data_augmentation(img_size=(224, 224)):
    """
    Creates a Keras Sequential model for data augmentation.
    """
    return models.Sequential([
        layers.Input(shape=(*img_size, 3)),
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.3),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
    ], name="data_augmentation")

# --- build_model function using MobileNetV2 ---
def build_model(img_size=(224, 224)):
    """
    Builds the MobileNetV2 model using Keras 3.
    """
    inputs = layers.Input(shape=(*img_size, 3))
    
    augmentation_layers = data_augmentation(img_size)
    x = augmentation_layers(inputs)
    
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    
    base_model = tf.keras.applications.MobileNetV2(
        include_top=False, 
        weights="imagenet",
        input_shape=(*img_size, 3),
        name="base_mobilenet" 
    )
    base_model.trainable = False 

    x = base_model(x, training=False) 
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5, name="top_dropout")(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    
    model = models.Model(inputs, outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
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
    """
    Loads and preprocesses a single image file.
    """
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    return image, label

def create_dataset(df, shuffle=True):
    """
    Creates a tf.data.Dataset from a dataframe.
    """
    ds = tf.data.Dataset.from_tensor_slices((
        df['filepath'].values,
        df['Cataract'].values
    ))
    
    ds = ds.map(parse_image, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
        
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

# --- MAIN TRAINING SCRIPT ---
if __name__ == "__main__":
    
    # 1. Load Data (now with file verification)
    all_df = load_data(LABEL_FILE)
    
    if all_df.empty:
        print("Error: Dataframe is empty or no files found. Exiting.")
        exit()

    print(f"Total images found and verified: {len(all_df)}")
    print(all_df['Cataract'].value_counts())

    # 2. Split Data
    train_df, val_df = train_test_split(
        all_df, 
        test_size=0.2, 
        random_state=42, 
        stratify=all_df['Cataract']
    )
    
    print(f"\nTraining images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")

    # 3. Calculate Class Weights
    #class_labels = np.array(train_df['Cataract'].values.astype(int))
    #weights = class_weight.compute_class_weight(
    #    'balanced',
    #    classes=np.unique(class_labels),
    #    y=class_labels
    #)
    #class_weights = dict(zip(np.unique(class_labels), weights))
    # print(f"\nCalculated Class Weights: {class_weights}")

    # --- Try Manual Weights (Less Aggressive) ---
    class_weights = {
        0: 1.0,  # Weight for 'No Cataract'
        1: 5.0   # Weight for 'Cataract' (Pay 5x more attention)
    }
    print(f"\nUsing Manual Class Weights: {class_weights}")
    # ----------------------------------------

    # 4. Create tf.data.Dataset pipelines
    print("\nCreating tf.data.Dataset pipelines...")
    train_ds = create_dataset(train_df, shuffle=True)
    val_ds = create_dataset(val_df, shuffle=False)
    print("Pipelines created.")
    
    # 5. Build Model
    model = build_model(img_size=IMG_SIZE)
    model.summary()

    # 6. Callbacks
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "cataract_model_v2.keras", 
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

    # 7. Train Model
    print("\n--- STARTING MODEL TRAINING (using MobileNetV2) ---")
    
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        class_weight=class_weights,
        callbacks=[checkpoint, early_stopping]
    )
    
    print("\n--- TRAINING COMPLETE ---")
    print("The best model has been saved as 'cataract_model_v2.keras'")
    
    # 8. Plot training history
    try:
        history_df = pd.DataFrame(history.history)
        plt.figure(figsize=(12, 8))
        plt.plot(history_df['auc'], label='Train AUC')
        # --- FIXED TYPO ON THIS LINE ---
        plt.plot(history_df['val_auc'], label='Validation AUC') 
        plt.title('Model AUC Over Epochs')
        plt.legend()
        plt.savefig('training_history.png')
        print("Saved training history plot to 'training_history.png'")
    except Exception as e:
        print(f"Could not plot history: {e}")
