import keras
import matplotlib
import numpy as np
import tensorflow.keras.backend as K
from matplotlib import pyplot as plt
from scipy.stats import pearsonr


def pearson_r(y_true, y_pred) -> float:
    """
    Calculates the Pearson correlation coefficient between the true and predicted values.
    :param y_true: true values
    :param y_pred: predicted values
    :return: pearson correlation coefficient
    """
    mean_y_true = K.mean(y_true)
    mean_y_pred = K.mean(y_pred)
    num = K.sum((y_true - mean_y_true) * (y_pred - mean_y_pred))
    den = K.sqrt(K.sum(K.square(y_true - mean_y_true)) * K.sum(K.square(y_pred - mean_y_pred)))
    return float(num / (den + K.epsilon()))


def visualize_predictions(model: keras.Model, x_test: np.ndarray, y_test: np.ndarray) -> matplotlib.figure.Figure:
    """
    Visualizes the predictions of the model against the true values and calculates the Pearson R value.

    Args:
        model (keras.Model): A trained model.
        x_test (numpy array): Testing data input.
        y_test (numpy array): Testing data target values.

    Returns:
        matplotlib.figure.Figure: A figure object containing the scatter plot.
    """

    # Get predictions
    y_pred = model.predict(x_test)

    # Calculate Pearson R value
    r, _ = pearsonr(y_test, y_pred.flatten())

    # Create scatter plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.5)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title(f'Scatter Plot of True vs. Predicted Values (Pearson R = {r:.3f})')

    return fig