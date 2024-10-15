import tensorflow as tf
from datetime import datetime

def train_model(model, X_train, y_train, X_val, y_val, epochs=15, batch_size=128):
    # TensorBoard callback
    log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    # Early stopping callback
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50, restore_best_weights=True)
    
    # Model checkpoint callback
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='models/best_model.keras',
                                                          monitor='val_accuracy',
                                                          save_best_only=True,
                                                          mode='max')
    
    # Train the model
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[tensorboard_callback, early_stopping, model_checkpoint],
                        verbose=1)
    
    return history