import numpy as np

from peptide_property_predictor.predict import PropertyPredictor


def test_property_predictor_rt():
    # Load a pre-trained model (replace 'model_name' with an actual model file in your models directory)
    predictor = PropertyPredictor("default_rt")

    # Predict properties for a list of peptide sequences
    sequences = ["P"*30, "ACDF", "GHIL", "KMNP", "QRST", "VWYPEPTIDE", "", "P"*31, "PEPTIDE" * 100, "PEPTX?!@"]
    predictions = predictor.predict(sequences)

    # Check if the predictions are returned in the correct format
    assert len(predictions) == len(sequences)
    assert isinstance(predictions, list)
    assert all(isinstance(pred, (float, type(None))) for pred in predictions)
    assert isinstance(predictions[-1], type(None))
    assert isinstance(predictions[-2], type(None))
    assert isinstance(predictions[-3], type(None))


def test_property_predictor_im():
    # Load a pre-trained model (replace 'model_name' with an actual model file in your models directory)
    predictor = PropertyPredictor("default_im")

    # Predict properties for a list of peptide sequences
    sequences = ["P"*30, "ACDF", "GHIL", "KMNP", "QRST", "VWYPEPTIDE", "", "P"*31, "PEPTIDE" * 100, "PEPTX?!@"]
    charges = [1, 2, 3, 4, 5, 6, 7, 8]
    predictions = predictor.predict(sequences, charges)

    # Check if the predictions are returned in the correct format
    assert len(predictions) == len(sequences)
    assert isinstance(predictions, list)
    assert all(isinstance(pred, (float, type(None))) for pred in predictions)
    assert isinstance(predictions[-1], type(None))
    assert isinstance(predictions[-2], type(None))
    assert isinstance(predictions[-3], type(None))



