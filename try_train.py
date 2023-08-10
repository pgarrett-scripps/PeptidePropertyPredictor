import pandas as pd
from matplotlib import pyplot as plt

from peptide_property_predictor.df_processing import split_df, get_data, prepare_df
from peptide_property_predictor.sequence_processing import verify_sequence
from peptide_property_predictor.train import train_model, fine_tune
from peptide_property_predictor.utils import visualize_predictions

MAX_SEQ_LENGTH = 30

data = pd.read_csv(r'PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_test.tsv', sep='\t')

peptide_df = pd.DataFrame()
peptide_df['Sequence'] = data['sequence']
peptide_df['RetTime'] = data['RT']


# align retention times / values
peptide_df = peptide_df[['Sequence', 'RetTime']]
peptide_df = peptide_df.groupby('Sequence').median().reset_index()

peptide_df['valid'] = peptide_df['Sequence'].apply(lambda seq: verify_sequence(seq, MAX_SEQ_LENGTH))
peptide_df = peptide_df[peptide_df['valid'] == True]
peptide_df['SeqLen'] = [len(seq) for seq in peptide_df['Sequence']]
peptide_df['RetTime'] = peptide_df['RetTime'] / max(peptide_df['RetTime'])

print(len(peptide_df))

train_df, test_df = split_df(peptide_df, sequence_column='Sequence', test_size=0.2, random_state=42)

X_train, Y_train = prepare_df(train_df, 'RetTime', MAX_SEQ_LENGTH + 1)
X_test, Y_test = prepare_df(test_df, 'RetTime', MAX_SEQ_LENGTH + 1)

model = train_model(X_train, X_test, Y_train, Y_test, use_charge=False)

model.save("test.h5")

f = visualize_predictions(model, X_train, Y_train)
plt.show()





