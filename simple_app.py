import streamlit as st
from peptide_property_predictor.predict import PropertyPredictor

with st.sidebar:
    st.title('RT Predictor')
    sequences = st.text_area('Sequences', value='PEPTID(123.456)E\nPEPTIDE')
    sequences = sequences.split('\n')
    run_button = st.button('Run')

if run_button:
    st.write('Running...')
    st.write(sequences)

    predictor = PropertyPredictor("test")

    # Predict retention_time
    rt_predictions = predictor.predict(sequences)

    st.write(rt_predictions)