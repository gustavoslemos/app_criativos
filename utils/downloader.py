import os
import requests
import zipfile
from io import BytesIO
from google_drive_downloader import GoogleDriveDownloader as gdd

def download_from_drive(folder_url, destination):
    """
    Baixa todos os arquivos de uma pasta compartilhada do Google Drive.
    
    Args:
        folder_url (str): URL da pasta compartilhada do Google Drive.
        destination (str): Caminho local onde os arquivos serão salvos.
    """
    # Extrair o ID da pasta a partir da URL
    try:
        folder_id = folder_url.split('/')[-1]
        if 'folders' not in folder_url:
            raise ValueError("URL inválida. Certifique-se de fornecer o link de uma pasta do Google Drive.")
    except Exception as e:
        print(f"Erro ao extrair o ID da pasta: {e}")
        return
    
    # Criar o diretório de destino se não existir
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    # Baixar a pasta como um arquivo zip usando a API do Google Drive
    zip_url = f"https://drive.google.com/uc?id={folder_id}&export=download"
    
    session = requests.Session()
    response = session.get(zip_url, stream=True)
    token = None
    
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break
    
    if token:
        params = {'id': folder_id, 'export': 'download', 'confirm': token}
        response = session.get(zip_url, params=params, stream=True)
    
    # Extrair o conteúdo do zip
    try:
        with zipfile.ZipFile(BytesIO(response.content)) as z:
            z.extractall(destination)
        print(f"Arquivos baixados e extraídos em {destination}")
    except zipfile.BadZipFile:
        print("Falha ao baixar ou extrair a pasta. Verifique se o link está correto e se a pasta é pública.")
