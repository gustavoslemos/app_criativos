import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import os
import numpy as np

def load_data(data_folder):
    images = []
    for filename in os.listdir(data_folder):
        img_path = os.path.join(data_folder, filename)
        img = cv2.imread(img_path)
        images.append(img)
    images = np.array(images) / 255.0
    return images

def build_vae(input_shape):
    # Codificador
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2, padding='same')(x)
    # Camada Latente
    latent = layers.Flatten()(x)
    # Decodificador
    x = layers.Dense(np.prod(input_shape), activation='relu')(latent)
    x = layers.Reshape(input_shape)(x)
    x = layers.Conv2DTranspose(32, 3, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    # Modelo VAE
    vae = Model(inputs, outputs)
    vae.compile(optimizer='adam', loss='mse')
    return vae

def train_vae(vae, data):
    vae.fit(data, data, epochs=50, batch_size=16)

if __name__ == "__main__":
    data = load_data('data/processed/')
    vae = build_vae(input_shape=(256, 256, 3))
    train_vae(vae, data)
    vae.save('models/vae_model.h5')
