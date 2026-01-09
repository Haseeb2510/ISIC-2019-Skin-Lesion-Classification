import os
import sys

# Get the current directory of this script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight 
import tensorflow as tf
from tensorflow.keras import Sequential, Input, Model # pyright: ignore[reportMissingImports]
from tensorflow.keras.applications import EfficientNetB0 # pyright: ignore[reportMissingImports]
from tensorflow.keras.optimizers import Adam # pyright: ignore[reportMissingImports]
from tensorflow.keras.layers import (       # pyright: ignore[reportMissingImports]
    RandomFlip, RandomRotation, RandomZoom, RandomBrightness, RandomContrast, GlobalAveragePooling2D, Dropout, Dense, 
    Conv2D, BatchNormalization, Lambda, Activation, MaxPooling2D, Rescaling) 
from tensorflow import saved_model
from tensorflow.keras.metrics import CategoricalAccuracy, TopKCategoricalAccuracy # pyright: ignore[reportMissingImports] 
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt
import seaborn as sns
from data_cleaning.paths import DATA, TENSORFLOW_MODEL_FOLDER, METRICS_FOLDER

IMG_SIZE = 224
BATCH_SIZE = 64

# Optimize TensorFlow settings
tf.config.optimizer.set_jit(True)  # Enable XLA compilation
AUTOTUNE = tf.data.AUTOTUNE

# Load splits dataframes
def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(os.path.join(DATA, "train/split_train.csv"))
    train_df["filepath"] = train_df.apply(
        lambda row: os.path.join(DATA, "train", row["label_name"], f"{row['image']}.jpg"), 
        axis=1
    )

    val_df = pd.read_csv(os.path.join(DATA, "val/split_val.csv"))
    val_df["filepath"] = val_df.apply(
        lambda row: os.path.join(DATA, "val", row["label_name"], f"{row['image']}.jpg"), 
        axis=1
    )

    test_df = pd.read_csv(os.path.join(DATA, "test/split_test.csv"))
    test_df["filepath"] = test_df.apply(
        lambda row: os.path.join(DATA, "test", row["label_name"], f"{row['image']}.jpg"), 
        axis=1
    )
    
    # Verify files exist
    print("Checking file existence...")
    for split_name, df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        existing = df["filepath"].apply(os.path.exists).sum()
        print(f"{split_name}: {existing}/{len(df)} files found")
    
    return train_df, val_df, test_df

def get_num_classes(df: pd.DataFrame) -> int:
    return len(df["label_idx"].unique())

def load_image(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3) # type: ignore
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    return img, label

def create_data_augmentation() -> Sequential:
    return Sequential([
        # 1. Flipping (only horizontal - maintains anatomical orientation)
        RandomFlip("horizontal"),
        
        # 2. Small rotations (lesions can appear at slight angles)
        RandomRotation(0.03),  # ±11° max
        
        # 3. Zoom/crop variations (different camera distances)
        RandomZoom(0.08, fill_mode='reflect'),
        
        # 4. Brightness variations (different lighting conditions)
        Lambda(lambda x: tf.image.random_brightness(x, max_delta=0.15)),
        
        # 5. Contrast variations (different skin tones/lighting)
        Lambda(lambda x: tf.image.random_contrast(x, lower=0.85, upper=1.15)),
        
        # 6. Hue/saturation variations (different camera color profiles)
        Lambda(lambda x: tf.image.random_hue(x, max_delta=0.02)),
        Lambda(lambda x: tf.image.random_saturation(x, lower=0.9, upper=1.1)),
        
        # 7. Additive Gaussian noise (sensor noise)
        Lambda(lambda x: x + tf.random.normal(
            tf.shape(x), mean=0.0, stddev=0.02
        ))
    ])

def make_dataset(df: pd.DataFrame, data_augmentation: Sequential, augment=False, batch_size=BATCH_SIZE, shuffle_buffer=None):
    paths = df["filepath"].values
    labels = df["label_idx"].values

    # Create dataset
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    # Only shuffle training set ONCE
    if shuffle_buffer:
        ds = ds.shuffle(buffer_size=shuffle_buffer, reshuffle_each_iteration=False)
    
    # Load and decode images
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Cache after loading (before augmentation)
    ds = ds.cache()
    
    # Apply augmentation AFTER caching (if needed)
    if augment:
        ds = ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
    
    # Batch, repeat, prefetch
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds

def compute_class_weights(df: pd.DataFrame) -> dict:
    y = df["label_idx"].values
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y), # type: ignore
        y=y # type: ignore
    )
    
    # Use square root to reduce extreme values
    class_weights = np.sqrt(class_weights)
    return {i: w for i, w in enumerate(class_weights)}

def build_model(num_classes: int):
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    x = tf.keras.applications.efficientnet.preprocess_input(inputs) # type: ignore

    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )    
    
    # Freeze first 150 layers (general features)
    for layer in base_model.layers[:150]:
        layer.trainable = False
    
    # Unfreeze last 50 layers (specialized features)
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)

    x = Dense(256, activation="relu")(x)
    x = Dropout(0.5)(x)

    outputs = Dense(num_classes, activation="softmax")(x)
    
    model = Model(inputs, outputs)
    
    model.compile(
        optimizer=Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=[
            CategoricalAccuracy(name="acc"),
            TopKCategoricalAccuracy(k=2, name="top2_acc")
        ]
    )
    return model

def train_model(model, train_ds, val_ds, class_weights, epochs=100):
    # Callbacks
    callbacks = [
        # Learning rate scheduler
        ReduceLROnPlateau(
            monitor='val_accuracy',  # Monitor accuracy, not loss
            factor=0.5,
            patience=5,  # Increased patience
            min_lr=1e-6,
            mode='max',  # Maximize accuracy
            verbose=1
        ),
        # Early stopping
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,  # More patience for medical images
            restore_best_weights=True,
            mode='max',
            min_delta=0.001,  # Minimum improvement
            verbose=1
        )
    ]
    
    # Train with class weights
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    return history, model

def evaluate_model(model, test_ds):
    print("\nEvaluating on test set...")
    results = model.evaluate(test_ds, verbose=0)
    print(f"Test Loss: {results[0]:.4f}")
    print(f"Test Accuracy: {results[1]:.4f}")
    
    # Get predictions for confusion matrix
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(labels.numpy())
    
    return y_true, y_pred, results

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def save_metrics(model, history, results, y_pred, y_true, class_names, model_name="skin_condition_model"):
    # Create directory if it doesn't exist
    os.makedirs(TENSORFLOW_MODEL_FOLDER, exist_ok=True)
    
    # Save model
    model_path = os.path.join(TENSORFLOW_MODEL_FOLDER, f'{model_name}')
    saved_model.save(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(TENSORFLOW_MODEL_FOLDER, f'{model_name}_history.joblib')
    joblib.dump(history.history, history_path)

    # Save metrics
    metrics = {
        "results": results,
        "y_pred": y_pred,
        "y_true": y_true,
        "class_names": class_names
    }
    joblib.dump(os.path.join(METRICS_FOLDER, "tensorflow_model_metrics.joblib"), metrics)

    print(f"Training history, model and metrics saved successfully.")

def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def main():
    print("=" * 60)
    print("SKIN CONDITION CLASSIFICATION TRAINING")
    print("=" * 60)
    
    # Load data
    print("\n1. Loading data splits...")
    train_df, val_df, test_df = load_splits()
    
    # Dataset statistics
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f"\nDataset Statistics:")
    print(f"Total samples: {total_samples}")
    print(f"Train: {len(train_df)} ({len(train_df)/total_samples*100:.1f}%)")
    print(f"Validation: {len(val_df)} ({len(val_df)/total_samples*100:.1f}%)")
    print(f"Test: {len(test_df)} ({len(test_df)/total_samples*100:.1f}%)")
    
    # Get class information
    num_classes = get_num_classes(train_df)
    class_names = sorted(train_df["label_name"].unique())
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {class_names}")
    
    # Create datasets
    print("\n2. Creating datasets...")
    data_augmentation = create_data_augmentation()
    train_ds = make_dataset(
        train_df, 
        data_augmentation,
        augment=True, 
        batch_size=BATCH_SIZE,
        shuffle_buffer=min(1000, len(train_df))  # Smaller buffer
    )
    val_ds = make_dataset(val_df, data_augmentation, augment=False, batch_size=BATCH_SIZE)
    test_ds = make_dataset(test_df, data_augmentation, augment=False, batch_size=BATCH_SIZE)
    
    # Compute class weights
    print("\n3. Computing class weights...")
    class_weights = compute_class_weights(train_df)
    print(f"Class weights: {class_weights}")
    
    # Build model
    print("\n4. Building model...")
    model = build_model(num_classes)
    model.summary()
    
    # Train model
    print("\n5. Training model...")
    history, model = train_model(model, train_ds, val_ds, class_weights, epochs=100)
    
    # Evaluate model
    print("\n6. Evaluating model...")
    y_true, y_pred, test_results = evaluate_model(model, test_ds)
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Save model
    print("\n7. Saving model...")
    save_metrics(model, history, test_results, y_pred, y_true, class_names)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()