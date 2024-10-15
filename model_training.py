import tensorflow as tf
import numpy as np
from datetime import datetime
import os

def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=15, batch_size=128, log_dir='logs/fit', model_dir='models'):
    # Ensure y_train and y_val are NumPy arrays to prevent type issues
    y_train = np.array(y_train) if not isinstance(y_train, np.ndarray) else y_train
    y_val = np.array(y_val) if not isinstance(y_val, np.ndarray) else y_val
    
    # Check if y_train and y_val are 1D arrays
    if y_train.ndim != 1:
        y_train = y_train.ravel()
    if y_val.ndim != 1:
        y_val = y_val.ravel()
    
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    # TensorBoard callback for visualization
    log_path = os.path.join(log_dir, datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path, histogram_freq=1)

    # Early stopping to prevent overfitting
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True
    )

    # Save the best model during training
    model_path = os.path.join(model_dir, 'best_model.keras')
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=model_path, monitor='val_accuracy', save_best_only=True, mode='max'
    )

    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
        verbose=1

    )
    print(f"Returned from train_model: {type(history)}")
    return history, model
