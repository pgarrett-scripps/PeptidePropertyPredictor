"""
This module contains functions for training a model to predict peptide properties.
"""
from typing import List

import numpy as np


import keras
from tensorflow.keras.callbacks import EarlyStopping

from .model import build_default_model_small, build_charge_model_small
from .sequence_processing import IP2_SEQUENCE_CHARACTERS, split_sequences, get_train_flags, verify_sequence, \
    get_test_flags, prepare_data


def train_model(x_train: np.ndarray,
                x_test: np.ndarray,
                y_train: np.ndarray,
                y_test: np.ndarray,
                epochs: int = 100,
                batch_size: int = 128,
                min_delta: float = 0.0001,
                patience: int = 5,
                use_charge: bool = False) -> keras.Model:
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
    if use_charge is False:
        model = build_default_model_small((len(x_train[0]), len(IP2_SEQUENCE_CHARACTERS) + 1))
    else:
        model = build_charge_model_small((len(x_train[0][0]), len(IP2_SEQUENCE_CHARACTERS) + 1))

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


def train_rt(sequences: list[str], rts: List[float], max_len: int, model=None, epochs: int = 100,
             batch_size: int = 128, min_delta: float = 0.0001, patience: int = 5):
    valid_flags = [verify_sequence(sequence, max_len) for sequence in sequences]
    valid_sequences = [sequence for sequence, flag in zip(sequences, valid_flags) if flag is True]
    valid_rts = [rt for rt, flag in zip(rts, valid_flags) if flag is True]

    max_rt = max(valid_rts)
    scaled_rts = [rt / max_rt for rt in valid_rts]

    train_sequences, test_sequences = split_sequences(valid_sequences)
    train_flags = get_train_flags(valid_sequences, test_sequences)
    test_flags = get_test_flags(valid_sequences, test_sequences)

    train_sequences = [sequence for sequence, flag in zip(valid_sequences, train_flags) if flag is True]
    train_rts = [rt for rt, flag in zip(scaled_rts, train_flags) if flag is True]

    test_sequences = [sequence for sequence, flag in zip(valid_sequences, test_flags) if flag is True]
    test_rts = [rt for rt, flag in zip(scaled_rts, test_flags) if flag is True]

    X_train, Y_train = prepare_data(train_sequences, train_rts, None, max_len + 1)
    X_test, Y_test = prepare_data(test_sequences, test_rts, None, max_len + 1)

    if model is None:
        model = train_model(x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                            epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                            patience=patience)
    else:
        model = fine_tune(model=model, x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                          epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                          patience=patience)

    return model


def train_im(sequences: list[str], charges: List[int], ims: List[float], max_len: int, model=None, epochs: int = 100,
             batch_size: int = 128, min_delta: float = 0.0001, patience: int = 5):

    valid_flags = [verify_sequence(sequence, max_len) for sequence in sequences]
    valid_sequences = [sequence for sequence, flag in zip(sequences, valid_flags) if flag is True]
    valid_charges = [charge for charge, flag in zip(charges, valid_flags) if flag is True]
    valid_ims = [rt for rt, flag in zip(ims, valid_flags) if flag is True]

    max_im = max(valid_ims)

    scaled_ims = [rt / max_im for rt in valid_ims]

    train_sequences, test_sequences = split_sequences(valid_sequences)
    train_flags = get_train_flags(valid_sequences, test_sequences)
    test_flags = get_test_flags(valid_sequences, test_sequences)

    train_sequences = [sequence for sequence, flag in zip(valid_sequences, train_flags) if flag is True]
    train_charges = [charge for charge, flag in zip(valid_charges, train_flags) if flag is True]
    train_ims = [rt for rt, flag in zip(scaled_ims, train_flags) if flag is True]

    test_sequences = [sequence for sequence, flag in zip(valid_sequences, test_flags) if flag is True]
    test_charges = [charge for charge, flag in zip(valid_charges, test_flags) if flag is True]
    test_ims = [rt for rt, flag in zip(scaled_ims, test_flags) if flag is True]

    X_train, Y_train = prepare_data(train_sequences, train_ims, train_charges, max_len + 1)
    X_test, Y_test = prepare_data(test_sequences, test_ims, test_charges, max_len + 1)

    if model is None:
        model = train_model(x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                            epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                            patience=patience, use_charge=True)
    else:
        model = fine_tune(model=model, x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                          epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                          patience=patience)

    return model

