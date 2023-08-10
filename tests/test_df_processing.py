import numpy as np
import pandas as pd

from peptide_property_predictor.df_processing import prepare_df
from peptide_property_predictor.sequence_processing import IP2_SEQUENCE_CHARACTERS


def test_prepare_data():
    # Create a sample dataframe
    df = pd.DataFrame({
        'Sequence': ['AGCT', 'CGTA', 'TACG', 'GATC'],
        'Charge': [1.0, 2.0, 3.0, 4.0],
        'Value': [0.1, 0.2, 0.3, 0.4]
    })
    value_column = 'Value'
    max_seq_len = 5
    use_charge = False

    # Call the function with the sample data
    X_data, y_data = prepare_df(df, value_column, max_seq_len, use_charge)

    # Check that the output types are correct
    assert isinstance(X_data, np.ndarray)
    assert isinstance(y_data, np.ndarray)

    # Check that the shapes of the output arrays are correct
    assert X_data.shape == (len(df), max_seq_len + 1, len(IP2_SEQUENCE_CHARACTERS) + 1)
    assert y_data.shape == (len(df),)

    # Check that the values in the output arrays are correct
    assert np.allclose(y_data, df[value_column].values.astype(np.float32))
    # For X_data, you might want to add more specific checks depending on the behavior of encode_and_pad_sequence
