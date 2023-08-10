"""
This module contains functions for preprocessing amino acid sequences.
"""

from typing import Optional, List, Tuple, Union, Set

import numpy as np
from keras.utils import pad_sequences
from peptacular.sequence import parse_modified_sequence, strip_modifications, create_modified_sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

IP2_SEQUENCE_CHARACTERS = "ACDEFGHIKLMNPQRSTVWY@"
VALID_CAHRS = set(IP2_SEQUENCE_CHARACTERS)


def get_one_hot_encoder() -> OneHotEncoder:
    """
    Creates and fits a OneHotEncoder for amino acid sequences using the ip2_encoding.

    Returns:
        sklearn.preprocessing.OneHotEncoder: A fitted OneHotEncoder for amino acid sequences.
    """
    one_hot_encoder = OneHotEncoder(categories=[sorted(list(IP2_SEQUENCE_CHARACTERS))], sparse_output=False)
    one_hot_encoder.fit(np.array(list(IP2_SEQUENCE_CHARACTERS)).reshape(-1, 1))
    return one_hot_encoder


def encode_sequence(sequence: str, encoder: Optional[OneHotEncoder] = None) -> np.ndarray:
    """
    One-hot encodes an amino acid sequence and appends modification mass.

    Args:
        sequence (str): An amino acid sequence.
        encoder (Optional[OneHotEncoder]): A fitted OneHotEncoder for amino acid sequences.

    Returns:
        np.ndarray: The one-hot encoded sequence with appended modification mass.
    """
    encoder = encoder or get_one_hot_encoder()
    mod_dict = parse_modified_sequence(sequence)
    encoded_sequence = encoder.transform(np.array(list('@' + strip_modifications(sequence))).reshape(-1, 1))

    encoded_sequence = np.hstack((encoded_sequence, np.zeros((len(encoded_sequence), 1))))

    for i, mass in mod_dict.items():
        encoded_sequence[i + 1, -1] = float(mass)

    return encoded_sequence


def decode_sequence(encoded_sequence: np.ndarray, encoder: Optional[OneHotEncoder] = None) -> str:
    """
    Decodes a one-hot encoded amino acid sequence and reconstructs modifications.

    Args:
        encoded_sequence (np.ndarray): One-hot encoded sequence with appended modification mass.
        encoder (Optional[OneHotEncoder]): A fitted OneHotEncoder for amino acid sequences.

    Returns:
        str: The decoded amino acid sequence with reconstructed modifications.
    """
    encoder = encoder or get_one_hot_encoder()

    mod_dict = {i - 1: float(row[-1]) for i, row in enumerate(encoded_sequence) if row[-1] != 0.0}

    sequence = encoder.inverse_transform(encoded_sequence[:, :-1])
    sequence = ''.join(sequence.flatten())[1:]

    return create_modified_sequence(sequence, mod_dict)


def encode_and_pad_sequence(sequence: str, max_seq_len: int) -> np.ndarray:
    """
    Preprocesses the input sequences by one-hot encoding and padding to the maximum sequence length.

    Args:
    sequences (list): List of amino acid sequences.
    max_seq_len (int): Maximum sequence length allowed.

    Returns:
    numpy.array: A numpy array containing the preprocessed sequences.
    """

    # Pad sequences
    return pad_sequences([encode_sequence(sequence)], maxlen=max_seq_len + 1)[0]


def remove_padding(sequence: np.ndarray) -> np.ndarray:
    """
    Removes padding from one-hot encoded sequences.

    Args:
    one_hot_sequences (numpy.array): A numpy array of one-hot encoded sequences.

    Returns:
    list: The original sequences with padding removed.
    """
    return sequence[np.any(sequence != 0, axis=1)]


def unpad_and_decode_sequence(encoded_sequence: np.ndarray) -> str:
    """
    Preprocesses the input sequences by removing padding and reconstructing the amino acid sequences.

    Args:
    encoded_sequences (list): List of encoded sequences.
    max_seq_len (int): Maximum sequence length allowed.

    Returns:
    list: List of preprocessed sequences.
    """

    return decode_sequence(remove_padding(encoded_sequence))


def verify_sequence(sequence: str, max_seq_len: int = None) -> bool:
    """
    Verifies that all characters in the sequence are in the subset of characters used for training the model.

    Args:
        sequence (str): An amino acid sequence.

    Returns:
        bool: True if the sequence is valid, False otherwise.
    """

    mods = parse_modified_sequence(sequence)
    stripped_sequence = strip_modifications(sequence)

    sequence_chars = set(stripped_sequence)

    if not sequence_chars.issubset(VALID_CAHRS):
        return False

    # verify that all modifications are floats
    for mod in mods.values():
        if not is_float(mod):
            return False

    if max_seq_len is not None:
        if len(stripped_sequence) > max_seq_len:
            return False

    return True


def is_float(value: str) -> bool:
    """
    Check if a string can be converted to a float.

    Args:
        value (str): The string to check.

    Returns:
        bool: True if the string can be converted to a float, False otherwise.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


def get_valid_sequence_indices(sequences: List[str], max_len) -> List[int]:
    """
    Returns a list of indices of valid sequences.

    Args:
        sequences (list): List of amino acid sequences.
        max_len (int): Maximum sequence length allowed.

    Returns:
        list: List of indices of valid sequences.
    """
    valid_indices = []
    for idx, seq in enumerate(sequences):
        if verify_sequence(seq, max_len) is True:
            valid_indices.append(idx)
    return valid_indices


def prepare_data(sequences: List[str],
                 values: List[float],
                 charges: Optional[List[int]] = None,
                 max_seq_len: int = 30) -> \
        Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]:
    """
    Prepares the data for model training/testing.

    Args:
        sequences (List[str]): List of sequences.
        values (List[float]): List of target values.
        charges (Optional[List[int]]): List of charges, if available.
        max_seq_len (int): The maximum length for sequence padding.

    Returns:
        Union[Tuple[np.ndarray, np.ndarray], Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]]: Prepared data (X_data, y_data).
    """
    # Encode and pad sequences
    X_sequences = np.array([encode_and_pad_sequence(sequence, max_seq_len) for sequence in sequences]).astype(np.float32)

    # If 'charges' is not None, stack the encoded sequences with the 'Charge' values along a new axis
    if charges is not None:
        X_charge = np.array([float(charge) for charge in charges]).astype(np.float32)
        X_data = (X_sequences, X_charge)
    else:
        X_data = X_sequences

    # Convert values to a float32 numpy array
    y_data = np.array(values).astype(np.float32)

    return X_data, y_data


def split_sequences(sequences: List[str], test_size: float = 0.2, random_state: int = 42) -> Tuple[Set[str], Set[str]]:
    """
    Splits the list of sequences into training and testing sets ensuring no sequence overlap.

    Args:
        sequences (List[str]): List of sequences to split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Random state for reproducibility.

    Returns:
        Tuple[Set[str], Set[str]]: Sets of train and test sequences.
    """
    # Generate a set of unique sequences
    unique_sequences = set(sequences)

    # Split the sequences into training and test sets
    train_sequences, test_sequences = train_test_split(list(unique_sequences), test_size=test_size, random_state=random_state)

    return set(train_sequences), set(test_sequences)


def get_test_flags(sequences: List[str], test_sequences: Set[str]) -> List[bool]:
    """
    Generates a list of boolean flags indicating whether each sequence is in the test set.

    Args:
        sequences (List[str]): List of sequences.
        test_sequences (Set[str]): Set of test sequences.

    Returns:
        List[bool]: List of boolean flags.
    """
    test_flags = [sequence in test_sequences for sequence in sequences]
    return test_flags


def get_train_flags(sequences: List[str], test_sequences: Set[str]) -> List[bool]:
    """
    Generates a list of boolean flags indicating whether each sequence is in the training set.

    Args:
        sequences (List[str]): List of sequences.
        test_sequences (Set[str]): Set of test sequences.

    Returns:
        List[bool]: List of boolean flags.
    """
    train_flags = [sequence not in test_sequences for sequence in sequences]
    return train_flags
