import matplotlib.pyplot as plt

def show_images(images, titles=None):
    n = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(images[i])
        if titles:
            plt.title(titles[i])
        plt.axis('off')
    plt.show()
