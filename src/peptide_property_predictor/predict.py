"""
This module contains the PropertyPredictor class, which is used for predicting the properties of amino acid sequences
"""

import os
from typing import List, Union

import keras
import numpy as np
from tensorflow.keras.models import load_model

from peptide_property_predictor.sequence_processing import verify_sequence, encode_and_pad_sequence, \
    get_valid_sequence_indices


class PropertyPredictor:
    """
    A class for predicting the properties of amino acid sequences using a trained model.
    """

    def __init__(self, model: Union[keras.Model, str]):
        """
        Initializes the PropertyPredictor with a model or a path to a saved model.

        Args:
            model (str or keras.Model): A trained model or the name of a saved model file.
        """

        self.model = None
        self.load_model(model)

    def load_model(self, model: Union[keras.Model, str]) -> None:
        """
        Replaces the loaded model with a new model or a path to a saved model.

        Args:
            model (str or keras.Model): A trained model or the name of a saved model file.
        """
        if isinstance(model, str):
            script_dir = os.path.dirname(os.path.realpath(__file__))
            model_path = os.path.join(script_dir, "models", f"{model}.h5")
            self.model = load_model(model_path)
            print(f'loaded model: {model_path}')
        else:
            self.model = model

    def predict(self, sequences: List[str], charges: List[int] = None) -> List[Union[float, None]]:
        """
        Predicts the property values for the given amino acid sequences.

        Args:
            sequences (list): List of amino acid sequences.

        Returns:
            list: List of predicted property values for the input sequences.
        """
        if charges is not None:
            max_len = self.model.input_shape[0][1] - 1
        else:
            max_len = self.model.input_shape[1] - 1

        valid_indices = get_valid_sequence_indices(sequences, max_len)
        valid_sequences = [sequences[idx] for idx in valid_indices]

        if charges is not None:
            valid_charges = [charges[idx] for idx in valid_indices]
            valid_predictions = _predict_property(valid_sequences, self.model, valid_charges)
        else:
            valid_predictions = _predict_property(valid_sequences, self.model)

        predictions = [None] * len(sequences)
        for idx, pred in zip(valid_indices, valid_predictions):
            predictions[idx] = float(pred)

        return predictions


def _predict_property(sequences: List[str], model: keras.Model, charges: List[int] = None) -> np.ndarray:
    """
    A helper function that predicts the property values for the given amino acid sequences using the provided model.

    Args:
        sequences (list): List of amino acid sequences.
        model (keras.Model): A trained model.

    Returns:
        numpy.array: A numpy array containing the predicted property values for the input sequences.
    """

    if charges is not None:
        max_len = model.input_shape[0][1] - 1
    else:
        max_len = model.input_shape[1] - 1

    # Preprocessing: One-hot encoding
    X_sequences = np.array([encode_and_pad_sequence(seq, max_len) for seq in sequences]).astype(np.float32)

    if charges is not None:
        X_charge = np.array([float(charge) for charge in charges]).astype(np.float32)
        X_data = (X_sequences, X_charge)

    else:
        X_data = X_sequences

    return model.predict(X_data).flatten()
