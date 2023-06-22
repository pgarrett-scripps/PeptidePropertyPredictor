"""
This module contains the PropertyPredictor class, which is used for predicting the properties of amino acid sequences
"""

import os
from typing import List, Union

import keras
import numpy as np
from tensorflow.keras.models import load_model

from .preprocessing import preprocess_sequences, verify_all_characters_are_in_subset


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

    def predict(self, sequences: List[str]) -> List[Union[float, None]]:
        """
        Predicts the property values for the given amino acid sequences.

        Args:
            sequences (list): List of amino acid sequences.

        Returns:
            list: List of predicted property values for the input sequences.
        """

        max_len = self.model.layers[0].input_shape[1]
        valid_indices = get_valid_sequence_indices(sequences, max_len)
        valid_sequences = [sequences[idx] for idx in valid_indices]
        valid_predictions = _predict_property(valid_sequences, self.model)

        predictions = [None] * len(sequences)
        for idx, pred in zip(valid_indices, valid_predictions):
            predictions[idx] = float(pred)

        return predictions


def get_valid_sequence_indices(sequences: List[str], max_len) -> List[int]:
    valid_indices = []
    for idx, seq in enumerate(sequences):
        if 0 < len(seq) <= max_len and verify_all_characters_are_in_subset(seq) is True:
            valid_indices.append(idx)
    return valid_indices


def _predict_property(sequences: List[str], model: keras.Model) -> np.ndarray:
    """
    A helper function that predicts the property values for the given amino acid sequences using the provided model.

    Args:
        sequences (list): List of amino acid sequences.
        model (keras.Model): A trained model.

    Returns:
        numpy.array: A numpy array containing the predicted property values for the input sequences.
    """

    # Preprocessing: One-hot encoding
    padded_sequences = preprocess_sequences(sequences, model.layers[0].input_shape[1])
    return model.predict(padded_sequences).flatten()
