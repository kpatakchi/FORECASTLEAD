# python packages and directories
from py_env_train import *
from keras_unet_collection import models
import keras
from func_train import UNET

# Function to generate synthetic data
def generate_synthetic_data(n_samples, n_lat, n_lon, n_channels):
    X = np.random.rand(n_samples, n_lat, n_lon, n_channels).astype(np.float32)
    Y = np.random.rand(n_samples, n_lat, n_lon, 1).astype(np.float32)
    Z = Y*0
    return X, Y, Z

n_lat=128
n_lon=256
n_channels=7
ifn=64
dropout_rate=0
n_samples = 10  # Number of samples in the synthetic dataset
BS=2

X, Y, Z = generate_synthetic_data(n_samples, n_lat, n_lon, n_channels)
train_dataset = tf.data.Dataset.from_tensor_slices((X, Y, Z)).batch(BS)
val_dataset = tf.data.Dataset.from_tensor_slices((X+1, Y+1, Z)).batch(BS)

model = UNET(n_lat, n_lon, n_channels, ifn, dropout_rate, "trans-unet")
#model.summary()

model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

model.fit(train_dataset, validation_data=val_dataset)
