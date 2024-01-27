import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Tuple
from neural_networks import FNN, RNN


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model: torch.nn.Module, data_loader: DataLoader) -> Tuple[float, float, float, np.ndarray]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            # Flatten images if the model is FNN or RNN
            if isinstance(model, (FNN, RNN)):
                images = images.view(images.size(0), -1)

            outputs = model(images)
            _, predictions = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_predictions)

    return accuracy, precision, recall, conf_matrix


def plot_confusion_matrix(conf_matrix: np.ndarray, class_names, save: bool = False) -> None:
    # Normalize confusion matrix
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plotting the confusion matrix
    plt.figure(figsize=(len(class_names) + 1, len(class_names)))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", linewidths=.5, square=True,
                xticklabels=class_names, yticklabels=class_names, cbar_kws={"shrink": 0.8})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save:
        directory = 'results'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, 'confusion_matrix.png'))
    else:
        plt.show()
