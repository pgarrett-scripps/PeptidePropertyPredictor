"""
This module contains functions for training a model to predict peptide properties.
"""

from typing import List, Tuple

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import keras
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, Bidirectional, MaxPooling1D, GRU, \
    BatchNormalization
from tensorflow.keras.models import Sequential
from keras.layers import Input, Concatenate
from keras.models import Model

from .predict import get_valid_sequence_indices
from .preprocessing import preprocess_sequences, IP2_SEQUENCE_CHARACTERS


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


def build_default_model_big(input_shape) -> keras.Model:
    """
    Builds a default model for peptide property prediction.
    """
    model = Sequential()
    model.add(Conv1D(512, 3, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # model.add(Bidirectional(CuDNNGRU(50, return_sequences=True)))
    model.add(Bidirectional(GRU(50, return_sequences=True)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    return model


def build_default_model_small(input_shape) -> keras.Model:
    """
    Builds a default model for peptide property prediction.
    """
    model = Sequential()
    model.add(Conv1D(128, 3, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # model.add(Bidirectional(CuDNNGRU(50, return_sequences=True)))
    model.add(Bidirectional(GRU(25, return_sequences=True)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    return model





def split_data(sequences: List[str], property_values: np.ndarray, max_seq_len: int = 64, test_size=0.2) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Splits and preprocesses the input data into training and testing sets.

    Args:
        sequences (list): List of amino acid sequences.
        property_values (list or numpy array): List of corresponding property values.
        max_seq_len (int, optional): Maximum sequence length allowed. Default is 64.

    Returns:
        tuple: A tuple containing the training and testing data (X_train, X_test, y_train, y_test).
    """

    valid_indices = get_valid_sequence_indices(sequences, max_seq_len)
    valid_sequences = [sequences[idx] for idx in valid_indices]
    valid_property_values = np.array([property_values[idx] for idx in valid_indices])

    # Normalize values
    rt_scaler = MinMaxScaler()
    rt_values_normalized = rt_scaler.fit_transform(valid_property_values.reshape(-1, 1)).flatten()

    # Preprocessing: One-hot encoding
    padded_sequences = preprocess_sequences(valid_sequences, max_seq_len)

    # Split the dataset
    if test_size != 0.0:
        x_train, x_test, y_train, y_test = train_test_split(padded_sequences, rt_values_normalized, test_size=test_size,
                                                            random_state=42)
    else:
        x_train, x_test, y_train, y_test = padded_sequences, np.array([]), rt_values_normalized, np.array([])

    return x_train, x_test, y_train, y_test


def train_model(x_train: np.ndarray,
                x_test: np.ndarray,
                y_train: np.ndarray,
                y_test: np.ndarray,
                epochs: int = 100,
                batch_size: int = 128,
                min_delta: float = 0.0001,
                patience: int = 5,
                use_charge_state: bool = False) -> keras.Model:
    """
    Trains a model using the provided training data and validates it using the testing data.

    Args:
        x_train (numpy array): Training data input.
        x_test (numpy array): Testing data input.
        y_train (numpy array): Training data target values.
        y_test (numpy array): Testing data target values.
        epochs (int, optional): Number of training epochs. Default is 100.
        batch_size (int, optional): Batch size for training. Default is 128.
        min_delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement.
            Default is 0.0001.
        patience (int, optional): Number of epochs with no improvement after which training will be stopped.
            Default is 5.

    Returns:
        keras.Model: A trained model.
    """
    if use_charge_state is False:
        model = build_default_model_small((len(x_train[0]), len(IP2_SEQUENCE_CHARACTERS)+1))
    else:
        model = build_charge_model_small((len(x_train[0]), len(IP2_SEQUENCE_CHARACTERS)+1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    total_params = model.count_params()
    print(f"Total number of parameters in the model: {total_params}")

    # Train the model
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[early_stopping_callback])

    return model


def fine_tune(model: keras.Model,
              x_train: np.ndarray,
              x_test: np.ndarray,
              y_train: np.ndarray,
              y_test: np.ndarray,
              epochs: int = 10,
              batch_size: int = 128,
              min_delta: float = 0.0001,
              patience: int = 5):
    # Train the model
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience)
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
              callbacks=[early_stopping_callback])

    return model


def build_charge_model_small(input_shape) -> keras.Model:
    """
    Builds a default model for peptide property prediction with sequence and charge state inputs.
    """
    sequence_input = Input(shape=input_shape, name='sequence_input')
    charge_input = Input(shape=(1,), name='charge_input')

    x = Conv1D(128, 3, padding='same')(sequence_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(GRU(25, return_sequences=True))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.45)(x)

    x = Concatenate()(
        [x, charge_input])  # concatenate the output of the sequence input processing and the charge input

    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.45)(x)

    output = Dense(1, kernel_initializer='normal', activation='sigmoid')(x)  # final output layer

    model = Model(inputs=[sequence_input, charge_input], outputs=output)  # create model with two inputs

    return model


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
