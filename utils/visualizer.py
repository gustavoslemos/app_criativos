import matplotlib.pyplot as plt

def show_images(images, titles=None):
    """
    Exibe uma lista de imagens lado a lado.
    
    Args:
        images (list): Lista de arrays de imagens.
        titles (list): Lista de títulos para as imagens.
    """
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()
