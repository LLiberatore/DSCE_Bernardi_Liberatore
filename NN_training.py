import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd

epochs = 200
patience = int(0.2 * epochs)

# Load preprocessed data
X = np.load("X_features.npy")
Y_labels = np.load("Y_labels.npy")

# Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_labels, test_size=0.2, random_state=42)

# Scale inputs
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

# Scale outputs (trainin only)
scaler_Y = MinMaxScaler()
Y_train_scaled = scaler_Y.fit_transform(Y_train)

# Define network
net = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(Y_labels.shape[1], activation='linear'),
])

# Compile
net.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor = 'val_loss',           # monitor validation loss
    patience = patience,            # stop if no improvement after 20% of total epochs
    restore_best_weights = True     # restore weights from best epoch
)

# Train
history = net.fit(X_train_scaled, Y_train_scaled, validation_split = 0.2, epochs=epochs, batch_size=32, verbose=1, callbacks=[early_stop])

# Predict (rescale predictions)
Y_pred_scaled = net.predict(X_test_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
