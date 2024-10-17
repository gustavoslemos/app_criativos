import os
import cv2

def preprocess_images(input_folder, output_folder, image_size=(256, 256)):
    """
    Redimensiona e normaliza as imagens na pasta de entrada e salva na pasta de saída.
    
    Args:
        input_folder (str): Caminho da pasta com as imagens originais.
        output_folder (str): Caminho da pasta para salvar as imagens processadas.
        image_size (tuple): Tamanho das imagens de saída (largura, altura).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, image_size)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
    print(f"Imagens processadas e salvas em {output_folder}")
