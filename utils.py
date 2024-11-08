import os
import json
import random
import datetime
import itertools
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def save_training_data(file_path, hyperparams, losses, accuracies, val_accuracies):
    """
    Saves the hyperparameters and training data into a JSON file.
    Creates the directory for the file if it doesn't exist.
    """
    timestamp = datetime.datetime.now().isoformat()

    data = {
        'timestamp': timestamp,
        'hyperparams': hyperparams,
        'losses': losses,
        'accuracies': accuracies,
        'val_accuracies': val_accuracies
    }

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    try:
        with open(file_path, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        all_data = {}

    all_data[timestamp] = data

    with open(file_path, 'w') as f:
        json.dump(all_data, f, indent=4)




def load_training_data(file_path):
    """
    Loads the training data from the JSON file.
    """
    with open(file_path, 'r') as f:
        all_data = json.load(f)
    return all_data

def plot_training_data(all_data):
    """
    Displays the training data (loss, accuracy, and validation accuracy)
    for each training session as side-by-side graphs.
    Also displays the hyperparameters for each session.
    """
    num_trainings = len(all_data)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for timestamp, training_data in all_data.items():
        losses = training_data['losses']
        accuracies = training_data['accuracies']
        val_accuracies = training_data['val_accuracies']

        axs[0].plot(losses, label=timestamp)
        axs[1].plot(accuracies, label=timestamp)
        axs[2].plot(val_accuracies, label=timestamp)

    axs[0].set_title('Loss')
    axs[1].set_title('Training Accuracy')
    axs[2].set_title('Validation Accuracy')
    for ax in axs:
        ax.legend()
        ax.set_xlabel('Epochs')

    plt.tight_layout()
    plt.show()

    print("Hyperparameters for different trainings:\n")
    for timestamp, training_data in all_data.items():
        print(f"Timestamp: {timestamp}")
        for param, value in training_data['hyperparams'].items():
            print(f"{param}: {value}")
        print("-" * 30)


def remove_corrupted_images(directory):
    """
    Remove corrupted image files from a specified directory.

    Parameters:
    directory (str): The path to the directory containing the image files.

    Returns:
    None
    """
    initial_count = len([name for name in os.listdir(
        directory) if name.endswith('.jpg')])
    corrupted_files = []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        try:
            with Image.open(file_path) as img:
                img.verify()
        except (IOError, SyntaxError):
            print(f"Corrupted file detected and removed: {file_path}")
            corrupted_files.append(file_path)
            os.remove(file_path)

    remaining_count = initial_count - len(corrupted_files)
    print(f"Total images in {directory}: {
          remaining_count} (after removing {len(corrupted_files)} corrupted files)")


def load_nearest_checkpoint(model, target_epoch, directory="./checkpoints"):
    """
    Load the nearest checkpoint of the model's state dictionary.

    Parameters:
    model (torch.nn.Module): The PyTorch model to load the checkpoint into.
    target_epoch (int): The target epoch number. The function will load the nearest checkpoint
                        that is less than or equal to this target epoch.
    directory (str, optional): The directory where the checkpoint files are located.
                               Defaults to "./checkpoints".

    Returns:
    int: The epoch number of the loaded checkpoint, or 0 if no suitable checkpoint was found.
    """
    if not os.path.exists(directory) or not os.listdir(directory):
        print("No checkpoints found. Starting from epoch 0.")
        return 0

    available_epochs = []
    for file in os.listdir(directory):
        if file.endswith(".pt"):
            try:
                epoch = int(file.split(".")[0])
                available_epochs.append(epoch)
            except ValueError:
                pass

    if not available_epochs:
        print("No checkpoints found. Starting from epoch 0.")
        return 0

    nearest_epoch = max(
        (epoch for epoch in available_epochs if epoch <= target_epoch), default=None)

    if nearest_epoch is not None:
        checkpoint_path = os.path.join(directory, f"{nearest_epoch}.pt")
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Checkpoint loaded: {checkpoint_path} (Epoch {nearest_epoch})")
        return nearest_epoch
    else:
        print("No suitable checkpoint found. Starting from epoch 0.")
        return 0


def save_checkpoint(model, epoch, directory="./checkpoints"):
    """
    Save a checkpoint of the model's state dictionary.

    Parameters:
    model (torch.nn.Module): The PyTorch model to save.
    epoch (int): The current epoch number, used for naming the checkpoint file.
    directory (str, optional): The directory where the checkpoint file will be saved.
                               Defaults to "./checkpoints".

    Returns:
    None
    """
    os.makedirs(directory, exist_ok=True)

    filename = os.path.join(directory, f"{epoch}.pt")
    torch.save(model.state_dict(), filename)
    print(f"Checkpoint saved: {filename}")


def plot_roc_curve(true_labels, prediction_scores):
    """
    Plots the ROC curve based on true labels and prediction scores.

    Args:
        true_labels (array-like): True binary labels (0 or 1).
        prediction_scores (array-like): Prediction scores or probabilities for the positive class.
    """
    fpr, tpr, thresholds = roc_curve(true_labels, prediction_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'Courbe ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--',
             label='Line of No Discrimination')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

    plt.title('ROC curve')
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
    plt.plot(thresholds, tpr, label="Sensitivity (TPR)", color="blue")
    plt.plot(thresholds, specificity, label="Specificity", color="orange")

    # Add vertical line at the approximate intersection point
    plt.axvline(x=intersection_threshold, color='black', linestyle='--',
                label=f"Intersection threshold ~ {intersection_threshold:.2f}")

    # Labels and title
    plt.xlabel("Seuil")
    plt.ylabel("Taux")
    plt.title("Sensitivity and Specificity as a Function of Threshold")
    plt.legend()
    plt.show()

    return intersection_threshold


def plot_score_distribution(prediction_scores, true_labels, classes, threshold=0.5):
    """
    Display histogram of prediction scores for each class and plot confusion matrix.

    Args:
        prediction_scores (list): List of raw prediction scores (before thresholding).
        true_labels (list): List of true binary labels (0 or 1).
        classes (list): List of class names, e.g., ['Dog', 'Cat'].
        threshold (float): Threshold to use for binary classification.
    """
    dog_scores = [score[0] for score, label in zip(
        prediction_scores, true_labels) if label == 0]
    cat_scores = [score[0] for score, label in zip(
        prediction_scores, true_labels) if label == 1]

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

    ax[0].plot(losses, label='Loss', color='orange')
    ax[0].set_title('Training loss')
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(accuracies, label='Accuracy', color='blue')
    ax[1].set_title('Batch accuracy')
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Acc')
    ax[1].legend()

    ax[2].plot(val_accuracies, label='Val accuracy', color='purple')
    ax[2].set_title('Validation Accuracy')
    ax[2].set_xlabel('Epoch')
    ax[2].set_ylabel('Val acc')
    ax[2].legend()

    plt.tight_layout()
    plt.show()


def show_random_samples(dataset, class_names, num_samples=8):
    """
    Displays a random sample of images from a PyTorch Dataset or Subset.
    
    :param dataset: PyTorch Dataset or Subset from which to sample
    :param class_names: List of class names (e.g. ["Dog", "Cat"])
    :param num_samples: Number of images to display (default: 8)
    """
    indices = torch.randint(len(dataset), size=(num_samples,))
    samples = [dataset[i] for i in indices]

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, (image, label) in enumerate(samples):
    
        if image.shape[0] == 1: 
           
            image = image.squeeze(0)

        axes[i].imshow(image, cmap="gray" if len(
            image.shape) == 2 else None)
        axes[i].set_title(class_names[label])
        axes[i].axis("off")

    plt.tight_layout()


def show_colored_random_samples(dataset, class_names, num_samples=8):
    """
    Displays a random sample of images from a PyTorch Dataset or Subset.
    
    :param dataset: PyTorch Dataset or Subset from which to sample
    :param class_names: List of class names (e.g. ["Dog", "Cat"])
    :param num_samples: Number of images to display (default: 8)
    """
    indices = torch.randint(len(dataset), size=(num_samples,))
    samples = [dataset[i] for i in indices]

    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i, (image, label) in enumerate(samples):
      
        image = image.permute(1, 2, 0).numpy() 
        image = image * [0.229, 0.224, 0.225] + \
            [0.485, 0.456, 0.406] 
        image = np.clip(image, 0, 1)

        axes[i].imshow(image) 
        axes[i].set_title(class_names[label])
        axes[i].axis("off")

    plt.tight_layout()
