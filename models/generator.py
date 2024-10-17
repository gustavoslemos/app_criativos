import tensorflow as tf
import numpy as np
import cv2

def generate_new_creatives(vae_model_path, num_creatives, output_folder):
    vae = tf.keras.models.load_model(vae_model_path)
    for i in range(num_creatives):
        latent_vector = np.random.normal(size=(1, np.prod(vae.input_shape[1:])))
        generated_image = vae.layers[-1](latent_vector).numpy()
        generated_image = (generated_image[0] * 255).astype(np.uint8)
        output_path = os.path.join(output_folder, f'creative_{i}.png')
        cv2.imwrite(output_path, generated_image)

if __name__ == "__main__":
    generate_new_creatives('models/vae_model.h5', 10, 'data/generated/')
