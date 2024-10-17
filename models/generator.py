import os
import numpy as np
import cv2
import tensorflow as tf

def generate_new_creatives(vae_model_path, num_creatives, output_folder):
    """
    Gera novos criativos usando o modelo VAE treinado.
    
    Args:
        vae_model_path (str): Caminho para o modelo VAE salvo.
        num_creatives (int): Número de criativos a serem gerados.
        output_folder (str): Pasta para salvar os criativos gerados.
    """
    vae = tf.keras.models.load_model(vae_model_path)
    latent_dim = 128  # Deve corresponder ao tamanho da camada latente no treinamento
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i in range(num_creatives):
        latent_vector = np.random.normal(size=(1, latent_dim))
        generated_image = vae.decoder(latent_vector)
        generated_image = generated_image.numpy()[0]
        generated_image = (generated_image * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, f'creative_{i}.png')
        cv2.imwrite(output_path, generated_image)
    print(f"{num_creatives} criativos gerados em {output_folder}")

if __name__ == "__main__":
    generate_new_creatives('models/vae_model.h5', 10, 'data/generated/')
