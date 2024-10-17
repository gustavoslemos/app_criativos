import os
import cv2

def preprocess_images(input_folder, output_folder, image_size=(256, 256)):
    for filename in os.listdir(input_folder):
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, img)
