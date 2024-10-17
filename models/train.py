import os
import numpy as np
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model

def load_data(data_folder):
    """
    Carrega as imagens da pasta e normaliza os pixels.
    
    Args:
        data_folder (str): Caminho da pasta com as imagens.
    
    Returns:
        np.array: Array com as imagens normalizadas.
    """
    images = []
    for filename in os.listdir(data_folder):
        img_path = os.path.join(data_folder, filename)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = img / 255.0
        images.append(img)
    images = np.array(images)
    return images

def build_vae(input_shape):
    """
    Constrói um modelo de VAE simples.
    
    Args:
        input_shape (tuple): Formato da entrada (altura, largura, canais).
    
    Returns:
        Model: Modelo compilado do VAE.
    """
    # Encoder
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D(2, padding='same')(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D(2, padding='same')(x)
    # Latent space
    shape_before_flattening = x.shape[1:]
    x = layers.Flatten()(x)
    latent = layers.Dense(128, activation='relu')(x)
    # Decoder
    x = layers.Dense(np.prod(shape_before_flattening), activation='relu')(latent)
    x = layers.Reshape(shape_before_flattening)(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(3, 3, activation='sigmoid', padding='same')(x)
    # Model
    vae = Model(inputs, outputs)
    vae.compile(optimizer='adam', loss='mse')
    return vae

def train_vae(vae, data, epochs=50, batch_size=16):
    """
    Treina o modelo VAE.
    
    Args:
        vae (Model): O modelo VAE compilado.
        data (np.array): Dados de treinamento.
        epochs (int): Número de épocas para treinar.
        batch_size (int): Tamanho do lote.
    """
    vae.fit(data, data, epochs=epochs, batch_size=batch_size)
    print("Treinamento concluído.")

if __name__ == "__main__":
    data = load_data('data/processed/')
    vae = build_vae(input_shape=(256, 256, 3))
    train_vae(vae, data)
    if not os.path.exists('models/'):
        os.makedirs('models/')
    vae.save('models/vae_model.h5')
