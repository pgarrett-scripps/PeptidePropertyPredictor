"""
This module contains functions for preprocessing amino acid sequences.
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from peptacular.peptide import parse_modified_peptide, strip_modifications, create_modified_peptide
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

IP2_SEQUENCE_CHARACTERS = "ACDEFGHIKLMNPQRSTVWY@"


def get_one_hot_encoder():
    """
    Creates and fits a OneHotEncoder for amino acid sequences using the ip2_encoding.

    Returns:
    sklearn.preprocessing.OneHotEncoder: A fitted OneHotEncoder for amino acid sequences.
    """

    one_hot_encoder = OneHotEncoder(categories=[sorted(list(IP2_SEQUENCE_CHARACTERS))], sparse_output=False)
    one_hot_encoder.fit([[aa] for aa in IP2_SEQUENCE_CHARACTERS])
    return one_hot_encoder


def process_sequence(sequence: str) -> np.ndarray:
    mod_dict = parse_modified_peptide(sequence)
    encoded_sequence = get_one_hot_encoder().transform([[aa] for aa in '@' + strip_modifications(sequence)])

    # Append the modification to the end of each encoded amino acid
    encoded_sequence = np.concatenate((encoded_sequence, np.zeros((len(encoded_sequence), 1))), axis=1)

    # Add the mass of the modification to the last column of each row
    for i, mass in mod_dict.items():
        encoded_sequence[i + 1, -1] = float(mass)

    return encoded_sequence


def process_encoded_sequence(encoded_sequence: np.ndarray) -> str:
    # Get the modification mass from the last column of each row
    mod_dict = {}
    for i, row in enumerate(encoded_sequence):
        if row[-1] != 0.0:
            mod_dict[i - 1] = row[-1]

    # Remove the modification from the end of each encoded amino acid
    encoded_sequence = encoded_sequence[:, :-1]

    # Convert to amino acid sequence
    sequence = get_one_hot_encoder().inverse_transform(encoded_sequence)
    sequence = ''.join(sequence.flatten())[1:]

    return create_modified_peptide(sequence, mod_dict)


def preprocess_sequences(sequences: List[str], max_seq_len: int):
    """
    Preprocesses the input sequences by one-hot encoding and padding to the maximum sequence length.

    Args:
    sequences (list): List of amino acid sequences.
    max_seq_len (int): Maximum sequence length allowed.

    Returns:
    numpy.array: A numpy array containing the preprocessed sequences.
    """

    encoded_sequences = [process_sequence(sequence) for sequence in sequences]

    # Pad sequences
    padded_sequences = pad_sequences(encoded_sequences, maxlen=max_seq_len)

    return padded_sequences


def remove_padding_one_hot(one_hot_sequences):
    """
    Removes padding from one-hot encoded sequences.

    Args:
    one_hot_sequences (numpy.array): A numpy array of one-hot encoded sequences.

    Returns:
    list: The original sequences with padding removed.
    """
    return [sequence[np.any(sequence != 0, axis=1)] for sequence in one_hot_sequences]


def preprocess_encoded_sequences(encoded_sequences: List[np.ndarray]):
    """
    Preprocesses the input sequences by removing padding and reconstructing the amino acid sequences.

    Args:
    encoded_sequences (list): List of encoded sequences.
    max_seq_len (int): Maximum sequence length allowed.

    Returns:
    list: List of preprocessed sequences.
    """

    encoded_sequences = remove_padding_one_hot(encoded_sequences)
    sequences = [process_encoded_sequence(sequence) for sequence in encoded_sequences]

    return sequences


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


def split_df(df, sequence_column='Sequence', test_size=0.2, random_state=42):
    """
    Splits the dataframe into training and testing sets ensuring no sequence overlap.

    Args:
    df (pandas.DataFrame): DataFrame to split.
    sequence_column (str): The name of the column that contains the sequences.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): Random state for reproducibility.

    Returns:
    pandas.DataFrame: Train and test DataFrames.
    """
    # Generate a list of unique sequences
    sequences = df[sequence_column].unique()

    # Split the sequences into training and test sets
    train_sequences, test_sequences = train_test_split(sequences, test_size=test_size, random_state=random_state)

    # Create boolean masks for selecting rows from the dataframe
    train_mask = df[sequence_column].isin(train_sequences)
    test_mask = df[sequence_column].isin(test_sequences)

    # Select the rows for the training and test sets
    train = df[train_mask]
    test = df[test_mask]

    return train, test


def get_charge_data(df: pd.DataFrame, value_column: str, max_seq_len=30) -> Tuple[np.ndarray, np.ndarray]:
    X_data = preprocess_sequences(df['Sequence'], max_seq_len=max_seq_len)
    X_charge = df['Charge'].values
    y_data = df[value_column].values

    return list(zip(X_data, X_charge)), y_data


def get_data(df: pd.DataFrame, value_column: str, max_seq_len=30) -> Tuple[np.ndarray, np.ndarray]:
    X_data = preprocess_sequences(df['Sequence'], max_seq_len=max_seq_len)
    y_data = df[value_column].values.astype(np.float32)

    return X_data, y_data
