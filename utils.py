import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from typing import Tuple
from neural_networks import FNN, RNN  # Assuming you have FNN and RNN defined in 'neural_networks'

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def evaluate(model: tf.keras.Model, data_loader: tf.data.Dataset) -> Tuple[float, float, float, np.ndarray]:
    all_labels = []
    all_predictions = []

    for images, labels in data_loader:
        images, labels = images.numpy(), labels.numpy()

        # Flatten images if the model is FNN or RNN
        if isinstance(model, (FNN, RNN)):
            images = images.reshape(images.shape[0], -1)

        outputs = model(images)
        predictions = np.argmax(outputs, axis=1)

        all_labels.extend(labels)
        all_predictions.extend(predictions)

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
