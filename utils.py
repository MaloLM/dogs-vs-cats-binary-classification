import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import random

def get_random_samples(dataset, n):
    """
    Get n random samples from the dataset.

    Parameters:
    dataset (torch.utils.data.Dataset): The dataset to sample from.
    n (int): The number of samples to return.

    Returns:
    X (list): A list of n images from the dataset.
    y (list): A list of n labels corresponding to the images in X.
    """
    indices = random.sample(range(len(dataset)), n)
    X = []
    y = []
    for idx in indices:
        image, label = dataset[idx]
        X.append(image)
        y.append(label)
    return X, y

def plot_confusion_matrix(cm, classes=["Dog", "Cat"], normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.

    Parameters:
    cm (numpy.ndarray): The confusion matrix to be plotted.
    classes (list): List of class names to be displayed on the x and y axes.
    normalize (bool): If True, the confusion matrix will be normalized.
    title (str): Title of the plot.
    cmap (matplotlib.colors.Colormap): Colormap for the confusion matrix.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_training_metrics(losses, accuracies, val_accuracies):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    
    # loss
    ax[0].plot(losses, label='Loss', color='orange')
    ax[0].set_title('Training loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    # accuracy
    ax[1].plot(accuracies, label='Accuracy', color='blue')
    ax[1].set_title('Batch accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Acc')
    ax[1].legend()

    # validation accuracy
    ax[2].plot(val_accuracies, label='Val accuracy', color='purple')
    ax[2].set_title('Validation Accuracy')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Val acc')
    ax[2].legend()

    plt.tight_layout()
    plt.show()


def show_random_samples(dataset, class_names, num_samples=8):
    """
    Affiche un échantillon aléatoire d'images d'un dataset PyTorch.
    
    :param dataset: PyTorch Dataset ou Subset à partir duquel échantillonner
    :param class_names: Liste des noms des classes (ex. ["Dog", "Cat"])
    :param num_samples: Nombre d'images à afficher (par défaut : 8)
    """
    # Charger un batch aléatoire
    indices = torch.randint(len(dataset), size=(num_samples,))
    samples = [dataset[i] for i in indices]
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, (image, label) in enumerate(samples):
        # Enlever la dimension supplémentaire pour les images en niveaux de gris (1 canal)
        if image.shape[0] == 1:  # Vérifie si l'image est en niveaux de gris
            image = image.squeeze(0)  # Retire le canal unique pour le niveau de gris
        
        axes[i].imshow(image, cmap="gray" if len(image.shape) == 2 else None)  # Mode gris
        axes[i].set_title(class_names[label])
        axes[i].axis("off")

    plt.tight_layout()
