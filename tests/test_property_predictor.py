import numpy as np

from peptide_property_predictor.predict import PropertyPredictor
from peptide_property_predictor.preprocessing import IP2_SEQUENCE_CHARACTERS, \
    preprocess_sequences, process_sequence, process_encoded_sequence, preprocess_encoded_sequences
from peptide_property_predictor.train import split_data, train_model


def test_property_predictor():
    # Load a pre-trained model (replace 'model_name' with an actual model file in your models directory)
    predictor = PropertyPredictor("small_rt")

    # Predict properties for a list of peptide sequences
    sequences = ["ACDF", "GHIL", "KMNP", "QRST", "VWYPEPTIDE", "", "PEPTIDE" * 100, "PEPTX?!@"]
    predictions = predictor.predict(sequences)

    # Check if the predictions are returned in the correct format
    assert len(predictions) == len(sequences)
    assert isinstance(predictions, list)
    assert all(isinstance(pred, (float, type(None))) for pred in predictions)
    assert isinstance(predictions[-1], type(None))
    assert isinstance(predictions[-2], type(None))
    assert isinstance(predictions[-3], type(None))


def test_train_model():
    # Create a small synthetic dataset
    sequences = ["ACDF", "GHIL", "KMNP", "QRST", "VWYPEPTIDE", "", "PEPTIDE" * 100, "PEPTX?!@"] * 10
    property_values = np.random.rand(len(sequences))

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(sequences, property_values, max_seq_len=5)

    # Train the model
    model = train_model(X_train, X_test, y_train, y_test, epochs=5, batch_size=8)

    # Check if the model has the expected input and output shapes
    input_shape = model.layers[0].input_shape[1:]
    output_shape = model.layers[-1].output_shape[1:]
    assert input_shape == (5, len(IP2_SEQUENCE_CHARACTERS))
    assert output_shape == (1,)


def test_process_sequence():
    SEQUENCE = 'PEP(58)TIDE'
    encoded_sequence = process_sequence(SEQUENCE)
    sequence = process_encoded_sequence(encoded_sequence)
    assert SEQUENCE == sequence


def test_preprocess_sequences():
    SEQUENCES = ['PEP(58)TIDE']
    encoded_sequences = preprocess_sequences(SEQUENCES, 20)
    print(encoded_sequences)
    sequence = preprocess_encoded_sequences(encoded_sequences)
    print(sequence)
    assert SEQUENCES == sequence
