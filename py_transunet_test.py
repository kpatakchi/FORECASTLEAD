# python packages and directories
from py_env_train import *
from keras_unet_collection import models
import keras
from func_train import UNET

# Function to generate synthetic data
def generate_synthetic_data(n_samples, n_lat, n_lon, n_channels):
    X = np.random.rand(n_samples, n_lat, n_lon, n_channels).astype(np.float32)
    Y = np.random.rand(n_samples, n_lat, n_lon, 1).astype(np.float32)  # Assuming the output has 1 channel
    return X, Y

n_lat=128
n_lon=256
n_channels=7
ifn=64
dropout_rate=0
n_samples = 10  # Number of samples in the synthetic dataset

X, Y = generate_synthetic_data(n_samples, n_lat, n_lon, n_channels)

model = UNET(n_lat, n_lon, n_channels, ifn, dropout_rate, "trans-unet")
model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(X, Y, epochs=5, batch_size=2)

# Evaluate the model on the synthetic data
loss, accuracy = model.evaluate(X, Y)
print(f"Loss: {loss}, Accuracy: {accuracy}")