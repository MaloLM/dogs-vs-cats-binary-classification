import matplotlib.pyplot as plt
import numpy as np
import itertools
import torch
import random
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(true_labels, prediction_scores):
    """
    Plots the ROC curve based on true labels and prediction scores.

    Args:
        true_labels (array-like): True binary labels (0 or 1).
        prediction_scores (array-like): Prediction scores or probabilities for the positive class.
    """
    # Calculate FPR, TPR, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(true_labels, prediction_scores)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'Courbe ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Ligne de hasard')  # Random guess line
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    plt.show()
    return fpr, tpr, thresholds

def plot_sensitivity_specificity_vs_thresholds(thresholds, tpr, fpr):
    """
    Plots sensitivity (TPR) and specificity as functions of thresholds,
    with a vertical line indicating where TPR and specificity approximately intersect.

    Args:
        thresholds (array-like): List or array of threshold values.
        tpr (array-like): True positive rate (sensitivity) values corresponding to each threshold.
        fpr (array-like): False positive rate values corresponding to each threshold.
    """
    # Calculate specificity
    specificity = 1 - fpr

    # Find the index where TPR and specificity are closest (smallest difference)
    diffs = np.abs(tpr - specificity)
    intersection_index = np.argmin(diffs)
    intersection_threshold = thresholds[intersection_index]

    # Plot sensitivity (TPR) and specificity
    plt.plot(thresholds, tpr, label="Sensibilité (TPR)", color="blue")
    plt.plot(thresholds, specificity, label="Spécificité", color="orange")

    # Add vertical line at the approximate intersection point
    plt.axvline(x=intersection_threshold, color='black', linestyle='--', 
                label=f"Seuil d'intersection ~ {intersection_threshold:.2f}")

    # Labels and title
    plt.xlabel("Seuil")
    plt.ylabel("Taux")
    plt.title("Sensibilité et Spécificité en fonction du seuil")
    plt.legend()
    plt.show()


def plot_score_distribution(prediction_scores, true_labels, classes, threshold=0.5):
    """
    Display histogram of prediction scores for each class and plot confusion matrix.

    Args:
        prediction_scores (list): List of raw prediction scores (before thresholding).
        true_labels (list): List of true binary labels (0 or 1).
        classes (list): List of class names, e.g., ['Dog', 'Cat'].
        threshold (float): Threshold to use for binary classification.
    """
    # Split scores by class for histogram plotting
    dog_scores = [score[0] for score, label in zip(prediction_scores, true_labels) if label == 0]
    cat_scores = [score[0] for score, label in zip(prediction_scores, true_labels) if label == 1]

    # Plot histogram of scores with different colors for each class
    plt.figure(figsize=(10, 5))
    plt.hist(dog_scores, bins=30, alpha=0.5, color='blue', label=classes[0])
    plt.hist(cat_scores, bins=30, alpha=0.5, color='orange', label=classes[1])
    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Prediction Scores by Class')
    plt.legend(loc='upper right')
    plt.show()

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
