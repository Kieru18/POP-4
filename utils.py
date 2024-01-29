import os
import json
from typing import Tuple, Iterable, List

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.utils import set_random_seed
from keras.models import Model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

from CONST import SEED, HISTORY_PATH


def set_seed(seed: int = SEED) -> None:
    # np.random.seed(seed)
    # tf.random.set_seed(seed)
    # lines above are obsolete as keras handles all of them in one function
    set_random_seed(seed)


def make_predictions(model: Model, images: np.ndarray) -> np.ndarray:
    predictions = model.predict(images)

    predicted_labels = np.argmax(predictions, axis=1)

    return predicted_labels


def calculate_metrics(predicted_labels: np.ndarray, true_labels: np.ndarray) -> Tuple[float, float, float, np.ndarray]:
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted')
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    conf_matrix = confusion_matrix(true_labels, predicted_labels)

    return accuracy, precision, recall, conf_matrix


def plot_confusion_matrix(conf_matrix: np.ndarray, 
                          class_names: Iterable[str], 
                          title: str,
                          save: bool = False) -> None:
    # Normalize confusion matrix
    conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

    # Plotting the confusion matrix
    plt.figure(figsize=(len(class_names) + 1, len(class_names)))
    sns.set(font_scale=1.2)
    sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", linewidths=.5, square=True,
                xticklabels=class_names, yticklabels=class_names, cbar_kws={"shrink": 0.8})

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)

    if save:
        directory = 'results/confusion_matrix'
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(os.path.join(directory, f'{title}.png'))
    else:
        plt.show()


def save_metrics(name: str, **kwargs):
    directory = 'results/metrics'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, f'{name}.json')
    with open(filepath, 'w') as file:
            json.dump(kwargs, file)

def save_history(model_name: str, history: List[float]):
    with open(f'{HISTORY_PATH}{model_name}.json', 'w') as f:
            json.dump(history, f)
