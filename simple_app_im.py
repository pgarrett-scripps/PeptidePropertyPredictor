import streamlit as st
from peptide_property_predictor.predict import PropertyPredictor

with st.sidebar:
    st.title('IM Predictor')
    sequence_charge_pairs = st.text_area('Sequences', value='PEPTID(123.456)E,2\nPEPTIDE,2')
    sequence_charge_pairs = sequence_charge_pairs.split('\n')

    sequences, charges = [], []
    for pair in sequence_charge_pairs:
        if ',' in pair:
            sequence, charge = pair.split(',')
            sequences.append(sequence)
            charges.append(int(charge))
    run_button = st.button('Run')

if run_button:
    st.write('Running...')
    st.write((sequence, charge) for sequence, charge in zip(sequences, charges))

    predictor = PropertyPredictor("default_im")

    # Predict retention_time
    predictions = predictor.predict(sequences, charges)

    st.write(predictions)