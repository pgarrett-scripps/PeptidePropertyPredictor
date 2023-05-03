"""
This module contains functions for preprocessing amino acid sequences.
"""

from typing import List

import numpy as np
from sklearn.preprocessing import OneHotEncoder

IP2_SEQUENCE_CHARACTERS = "ACDEFGHIKLMNPQRSTVWY0123456789().-"


def get_one_hot_encoder():
    """
    Creates and fits a OneHotEncoder for amino acid sequences using the ip2_encoding.

    Returns:
    sklearn.preprocessing.OneHotEncoder: A fitted OneHotEncoder for amino acid sequences.
    """

    one_hot_encoder = OneHotEncoder(categories=[sorted(list(IP2_SEQUENCE_CHARACTERS))], sparse_output=False)
    one_hot_encoder.fit([[aa] for aa in IP2_SEQUENCE_CHARACTERS])
    return one_hot_encoder


def preprocess_sequences(sequences: List[str], max_seq_len: int):
    """
    Preprocesses the input sequences by one-hot encoding and padding to the maximum sequence length.

    Args:
    sequences (list): List of amino acid sequences.
    max_seq_len (int): Maximum sequence length allowed.

    Returns:
    numpy.array: A numpy array containing the preprocessed sequences.
    """

    # One-hot encoding
    encoded_sequences = get_one_hot_encoder().transform([[aa] for seq in sequences for aa in seq])
    encoded_sequences = np.split(encoded_sequences, np.cumsum([len(seq) for seq in sequences])[:-1])

    # Pad sequences
    padded_sequences = np.array([np.pad(seq, ((0, max_seq_len - len(seq)), (0, 0))) for seq in encoded_sequences])

    return padded_sequences


def verify_all_characters_are_in_subset(sequence: str):
    """
    Verifies that all characters in the sequence are in the subset of characters used for training the model.

    Args:
        sequence (str): An amino acid sequence.

    Raises:
        ValueError: If any of the characters in the sequence are not in the subset of characters used for training
            the model.
    """

    for char in sequence:
        if char not in IP2_SEQUENCE_CHARACTERS:
            return False
    return True
