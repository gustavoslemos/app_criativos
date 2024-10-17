from utils.downloader import download_from_drive
from utils.preprocessor import preprocess_images
from models.train import load_data, build_vae, train_vae
from models.generator import generate_new_creatives

def main():
    # Etapa 1: Baixar os criativos existentes
    folder_url = input("Insira o link da pasta do Google Drive com os criativos existentes: ")
    download_from_drive(folder_url, 'data/raw/')
    
    # Etapa 2: Pré-processamento dos dados
    preprocess_images('data/raw/', 'data/processed/')
    
    # Etapa 3: Treinamento do modelo
    data = load_data('data/processed/')
    vae = build_vae(input_shape=(256, 256, 3))
    train_vae(vae, data)
    
    # Etapa 4: Geração de novos criativos
    generate_new_creatives('models/vae_model.h5', 10, 'data/generated/')
    print("Processo concluído com sucesso.")

if __name__ == "__main__":
    main()
