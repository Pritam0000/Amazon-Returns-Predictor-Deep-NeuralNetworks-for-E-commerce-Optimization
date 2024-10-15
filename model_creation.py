import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation

def create_baseline_model(input_shape):
    model = Sequential([
        Dense(256, activation='relu', input_shape=input_shape),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def create_regularized_model(input_shape):
    L2Reg = tf.keras.regularizers.L2(l2=1e-6)
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer=L2Reg, input_shape=input_shape),
        Dense(128, activation='relu', kernel_regularizer=L2Reg),
        Dense(64, activation='relu', kernel_regularizer=L2Reg),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def create_dropout_model(input_shape):
    L2Reg = tf.keras.regularizers.L2(l2=1e-6)
    model = Sequential([
        Dense(256, activation='relu', kernel_regularizer=L2Reg, input_shape=input_shape),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=L2Reg),
        Dropout(0.3),
        Dense(64, activation='relu', kernel_regularizer=L2Reg),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

def create_batchnorm_model(input_shape):
    L2Reg = tf.keras.regularizers.L2(l2=1e-6)
    model = Sequential([
        Dense(256, kernel_regularizer=L2Reg, input_shape=input_shape),
        BatchNormalization(),
        Activation('relu'),
        Dropout(0.2),
        Dense(128, kernel_regularizer=L2Reg),
        BatchNormalization(),
        Activation('relu'),
        Dense(64, kernel_regularizer=L2Reg),
        BatchNormalization(),
        Activation('relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model