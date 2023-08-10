from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

from peptide_property_predictor.sequence_processing import encode_and_pad_sequence, verify_sequence
from peptide_property_predictor.train import train_model, fine_tune


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
    train_mask = df[sequence_column].isin(set(train_sequences))
    test_mask = df[sequence_column].isin(set(test_sequences))

    # Select the rows for the training and test sets
    train = df[train_mask]
    test = df[test_mask]

    return train, test


def prepare_df(df: pd.DataFrame, value_column: str, max_seq_len: int = 30, use_charge: bool = False) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Prepares the data for model training/testing.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        value_column (str): The name of the column that contains the target values.
        max_seq_len (int): The maximum length for sequence padding.
        use_charge (bool): Whether to use the 'Charge' column in the data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Prepared data (X_data, y_data).
    """
    # Encode and pad sequences
    X_sequences = np.stack(df['Sequence'].apply(lambda sequence: encode_and_pad_sequence(sequence, max_seq_len)).values)

    # If 'use_charge' is True, stack the encoded sequences with the 'Charge' values along a new axis
    if use_charge:
        X_charge = np.stack(df['Charge'].values.astype(np.float32))
        X_data = [X_sequences, X_charge]
    else:
        X_data = X_sequences

    # Get target values and convert to float32
    y_data = df[value_column].values.astype(np.float32)

    return X_data, y_data


def train_rt(peptide_df: pd.DataFrame, max_len: int, model=None, epochs: int = 100,
             batch_size: int = 128,
             min_delta: float = 0.0001,
             patience: int = 5,
             gradient: int = None):
    # align retention times / values
    peptide_df = peptide_df[['Sequence', 'RetTime']]
    peptide_df = peptide_df.groupby('Sequence').median().reset_index()

    peptide_df['valid'] = peptide_df['Sequence'].apply(lambda x: verify_sequence(x, max_len))
    peptide_df = peptide_df[peptide_df['valid'] == True]

    if gradient is None:
        max_rt = max(peptide_df['RetTime'])
    else:
        max_rt = gradient

    peptide_df['RetTime'] = peptide_df['RetTime'] / max(peptide_df['RetTime'])
    peptide_df['RetTime'] = peptide_df['RetTime'].astype('float32')

    if 'Train' in peptide_df.columns:
        train_df = peptide_df[peptide_df['Test'] == False]
        test_df = peptide_df[peptide_df['Test'] == True]
    else:
        train_df, test_df = split_df(peptide_df, sequence_column='Sequence', test_size=0.2, random_state=42)
        train_df['Test'] = False
        test_df['Test'] = True
        peptide_df = pd.concat([train_df, test_df])

    X_train, Y_train = prepare_df(train_df, 'RetTime', max_len, use_charge=False)
    X_test, Y_test = prepare_df(test_df, 'RetTime', max_len, use_charge=False)

    if model is None:
        model = train_model(x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                            epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                            patience=patience)
    else:
        if isinstance(model, str):
            model = load_model(model)

        model = fine_tune(model=model, x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                          epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                          patience=patience)

    X, Y = prepare_df(peptide_df, 'RetTime', max_len, use_charge=False)
    peptide_df['predicted'] = model.predict(X)
    peptide_df['predicted'] = peptide_df['predicted'] * max_rt
    peptide_df['RetTime'] = peptide_df['RetTime'] * max_rt

    return model, peptide_df


def train_im(peptide_df: pd.DataFrame, max_len: int, model=None, epochs: int = 100,
             batch_size: int = 128,
             min_delta: float = 0.0001,
             patience: int = 5):

    # align retention times / values
    peptide_df = peptide_df[['Sequence', 'Charge', 'IonMobility']]
    peptide_df = peptide_df.groupby(['Sequence', 'Charge']).median().reset_index()

    peptide_df['valid'] = peptide_df['Sequence'].apply(lambda x: verify_sequence(x, max_len))
    peptide_df = peptide_df[peptide_df['valid'] == True]

    peptide_df['IonMobility'] = peptide_df['IonMobility']
    peptide_df['IonMobility'] = peptide_df['IonMobility'].astype('float32')
    peptide_df['Charge'] = peptide_df['Charge'].astype('float32')

    train_df, test_df = split_df(peptide_df, sequence_column='Sequence', test_size=0.2, random_state=42)
    peptide_df['Test'] = peptide_df['Sequence'].isin(test_df['Sequence'])

    X_train, Y_train = prepare_df(train_df, 'IonMobility', max_len, use_charge=True)
    X_test, Y_test = prepare_df(test_df, 'IonMobility', max_len, use_charge=True)

    if model is None:
        model = train_model(x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                            epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                            patience=patience, use_charge=True)
    else:
        if isinstance(model, str):
            model = load_model(model)

        model = fine_tune(model=model, x_train=X_train, x_test=X_test, y_train=Y_train, y_test=Y_test,
                          epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                          patience=patience)

    X, Y = prepare_df(peptide_df, 'IonMobility', max_len, use_charge=True)
    peptide_df['predicted'] = model.predict(X)
    peptide_df['IonMobility'] = peptide_df['IonMobility']

    return model, peptide_df
