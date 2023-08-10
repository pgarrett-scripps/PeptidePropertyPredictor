import numpy as np
from peptacular.sequence import strip_modifications
from sklearn.preprocessing import OneHotEncoder

from peptide_property_predictor.sequence_processing import encode_sequence, decode_sequence, get_one_hot_encoder, \
    encode_and_pad_sequence, IP2_SEQUENCE_CHARACTERS, remove_padding, unpad_and_decode_sequence, verify_sequence, \
    get_train_flags, get_test_flags, split_sequences


def test_process_sequence():
    SEQUENCE = 'PEP(58.0)TIDE'
    encoded_sequence = encode_sequence(SEQUENCE)
    sequence = decode_sequence(encoded_sequence)
    assert SEQUENCE == sequence


def test_get_one_hot_encoder():
    encoder = get_one_hot_encoder()
    assert isinstance(encoder, OneHotEncoder)


def test_encode_sequence():
    sequence = 'AG'
    encoder = get_one_hot_encoder()
    encoded_sequence = encode_sequence(sequence, encoder)
    assert isinstance(encoded_sequence, np.ndarray)


def test_decode_sequence():
    sequence = 'AG'
    encoder = get_one_hot_encoder()
    encoded_sequence = encode_sequence(sequence, encoder)
    decoded_sequence = decode_sequence(encoded_sequence, encoder)
    assert isinstance(decoded_sequence, str)
    assert sequence == decoded_sequence


def test_encode_sequence_without_ohe():
    sequence = 'AG'
    encoded_sequence = encode_sequence(sequence)
    assert isinstance(encoded_sequence, np.ndarray)


def test_decode_sequence_without_ohe():
    sequence = 'AG'
    encoded_sequence = encode_sequence(sequence)
    decoded_sequence = decode_sequence(encoded_sequence)
    assert isinstance(decoded_sequence, str)
    assert sequence == decoded_sequence


def test_encode_modified_sequence():
    sequence = '(100.0)A(100.0)G(100.0)'
    encoded_sequence = encode_sequence(sequence)
    assert isinstance(encoded_sequence, np.ndarray)


def test_decode_modified_sequence():
    sequence = '(100.0)A(100.0)G(100.0)'
    encoded_sequence = encode_sequence(sequence)
    decoded_sequence = decode_sequence(encoded_sequence)
    assert isinstance(decoded_sequence, str)
    assert sequence == decoded_sequence


def test_encode_and_pad_sequence():
    sequence = 'AG'
    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    assert isinstance(padded_encoded_sequence, np.ndarray)
    assert padded_encoded_sequence.shape == (max_seq_len + 1, len(IP2_SEQUENCE_CHARACTERS) + 1)

def test_encode_and_pad_long_sequence():
    sequence = 'A'*100
    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    assert isinstance(padded_encoded_sequence, np.ndarray)
    assert padded_encoded_sequence.shape == (max_seq_len + 1, len(IP2_SEQUENCE_CHARACTERS) + 1)

def test_remove_padding():
    sequence = 'AG'
    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    unpadded_sequence = remove_padding(padded_encoded_sequence)
    assert isinstance(unpadded_sequence, np.ndarray)
    assert unpadded_sequence.shape == (len(sequence) + 1, len(IP2_SEQUENCE_CHARACTERS) + 1)

def test_remove_padding_long_sequence():
    sequence = 'A'*100
    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    unpadded_sequence = remove_padding(padded_encoded_sequence)
    assert isinstance(unpadded_sequence, np.ndarray)
    assert unpadded_sequence.shape == (max_seq_len + 1, len(IP2_SEQUENCE_CHARACTERS) + 1)


def test_unpad_and_decode_sequence():
    sequence = 'AG'
    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    decoded_sequence = unpad_and_decode_sequence(padded_encoded_sequence)
    assert isinstance(decoded_sequence, str)
    assert sequence == decoded_sequence


def test_encode_and_pad_sequence_modified_sequence():
    sequence = '(100.0)A(100.0)G(100.0)'
    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    assert isinstance(padded_encoded_sequence, np.ndarray)
    assert padded_encoded_sequence.shape == (max_seq_len + 1, len(IP2_SEQUENCE_CHARACTERS) + 1)


def test_remove_padding_modified_sequence():
    sequence = '(100.0)A(100.0)G(100.0)'

    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    unpadded_sequence = remove_padding(padded_encoded_sequence)
    assert isinstance(unpadded_sequence, np.ndarray)
    assert unpadded_sequence.shape == (len(strip_modifications(sequence)) + 1, len(IP2_SEQUENCE_CHARACTERS) + 1)


def test_unpad_and_decode_sequence_modified_sequence():
    sequence = '(100.0)A(100.0)G(100.0)'
    max_seq_len = 5
    padded_encoded_sequence = encode_and_pad_sequence(sequence, max_seq_len)
    decoded_sequence = unpad_and_decode_sequence(padded_encoded_sequence)
    assert isinstance(decoded_sequence, str)
    assert sequence == decoded_sequence


def test_verify_all_characters_are_in_subset():
    valid_sequence = 'AG'  # You should replace 'AG' with a valid sequence for your case
    invalid_sequence = 'XYZ'  # You should replace 'XYZ' with an invalid sequence for your case
    valid_modified_sequence = '(100.0)A(100.0)G(100.0)'  # You should replace '(100.0)A(100.0)G(100.0)' with a valid sequence for your case
    invalid_modified_sequence = '(100.0)A(10c0.0)G(100.0)'  # You should replace '(100.0)A(100.0)G(100.0)' with a valid sequence for your case

    # Test with valid sequence
    is_valid = verify_sequence(valid_sequence)
    assert is_valid is True

    # Test with invalid sequence
    is_invalid = verify_sequence(invalid_sequence)
    assert is_invalid is False

    # Test with invalid modified sequence
    is_valid_modified = verify_sequence(valid_modified_sequence)
    assert is_valid_modified is True

    # Test with invalid modified sequence
    is_valid_modified = verify_sequence(invalid_modified_sequence)
    assert is_valid_modified is False


def test_split_sequences():
    sequences = ['AGCT', 'CGTA', 'TACG', 'GATC', 'AGCT', 'CGTA']  # Some duplicate sequences
    test_size = 0.2
    random_state = 42

    train_sequences, test_sequences = split_sequences(sequences, test_size, random_state)

    # Check that the output types are correct
    assert isinstance(train_sequences, set)
    assert isinstance(test_sequences, set)

    # Check that the sizes of the output sets are correct
    assert len(train_sequences) == round((1 - test_size) * len(set(sequences)))
    assert len(test_sequences) == round(test_size * len(set(sequences)))

    # Check that there is no overlap between the train and test sets
    assert len(train_sequences.intersection(test_sequences)) == 0


def test_get_test_flags():
    sequences = ['AGCT', 'CGTA', 'TACG', 'GATC', 'GATC']
    test_sequences = set(['CGTA', 'GATC'])

    test_flags = get_test_flags(sequences, test_sequences)

    # Check that the output type is correct
    assert isinstance(test_flags, list)

    # Check that the output values are correct
    assert test_flags == [False, True, False, True, True]


def test_get_train_flags():
    sequences = ['AGCT', 'CGTA', 'TACG', 'GATC', 'GATC']
    test_sequences = set(['CGTA', 'GATC'])

    train_flags = get_train_flags(sequences, test_sequences)

    # Check that the output type is correct
    assert isinstance(train_flags, list)

    # Check that the output values are correct
    assert train_flags == [True, False, True, False, False]


