import ast
import base64
import os
from io import StringIO
from typing import Union

import pandas as pd
import  streamlit as st
from filterframes import from_dta_select_filter
from matplotlib import pyplot as plt
from peptacular.sequence import add_static_mods
from tensorflow.keras.models import load_model

from peptide_property_predictor.df_processing import train_rt, train_im

st.title('Property Trainer')

files = st.file_uploader("DTASelect-filter.txt files", type=["txt"], accept_multiple_files=True)
model = st.file_uploader("Model", type=["h5"])

model_type = st.radio("Train on Retention Time or Ion Mobility", ('Retention Time', 'Ion Mobility'))
static_mods = st.text_input('Static Modifications', value="{'C':57.021464}")
static_mods = ast.literal_eval(static_mods)

epochs = st.number_input('Epochs', value=10)
batch_size = st.number_input('Batch Size', value=128)
min_delta = st.number_input('Min Delta', value=0.0001)
patience = st.number_input('Patience', value=10)

def parse_dta_select_filter_files(files: Union[str, StringIO]):
    peptide_dfs = []
    for file in files:
        _, peptide_df, _, _ = from_dta_select_filter(file)
        peptide_dfs.append(peptide_df)

    peptide_df = pd.concat(peptide_dfs)
    peptide_df.reset_index(drop=True, inplace=True)
    peptide_df['Sequence'] = [seq[2:-2] for seq in peptide_df['Sequence']]
    return peptide_df

if st.button('Run'):

    if model is not None:
        #save model to tmp file
        with open('tmp.h5', 'wb') as f:
            f.write(model.getvalue())

        #load model
        model = load_model('tmp.h5')

    filter_ios = (fasta_file.getvalue().decode() for fasta_file in files)
    peptide_df = parse_dta_select_filter_files(filter_ios)
    peptide_df['Sequence'] = [add_static_mods(seq, static_mods) for seq in peptide_df['Sequence']]

    st.write(len(peptide_df))

    if model_type == 'Retention Time':
        model, peptide_df = train_rt(peptide_df, max_len=30, epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                         patience=patience, model=model)
    elif model_type == 'Ion Mobility':
        model, peptide_df = train_im(peptide_df, max_len=30, epochs=epochs, batch_size=batch_size, min_delta=min_delta,
                         patience=patience, model=model)
    else:
        st.write('Please select a model type')

    model_path = "model.h5"
    model.save(model_path)

    with open(model_path, "rb") as f:
        model_bytes = f.read()
        model_b64 = base64.b64encode(model_bytes).decode()

    st.markdown(f"<a href='data:file/h5;base64,{model_b64}' download='{model_path}'>Download trained model</a>",
                unsafe_allow_html=True)

    os.remove(model_path)

    st.write('Done')

    # create a pyplot figure with the predictions
    # Create scatter plot
    fig, ax = plt.subplots()
    test_df = peptide_df[peptide_df['Test'] == True]
    ax.scatter(test_df['RetTime' if model_type == 'Retention Time' else 'Ion Mobility'], test_df['predicted'], alpha=0.5)
    ax.set_xlabel('True Values')
    ax.set_ylabel('Predicted Values')
    st.pyplot(fig)







