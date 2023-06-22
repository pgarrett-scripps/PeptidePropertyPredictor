import filterframes
from peptacular.peptide import add_static_mods, strip_modifications, parse_modified_peptide
from sklearn.preprocessing import MinMaxScaler

from peptide_property_predictor.preprocessing import split_df, get_data
from peptide_property_predictor.train import train_model

MAX_SEQ_LENGTH = 30

_, peptide_df, _, _ = filterframes.from_dta_select_filter(r"C:\Users\Ty\Downloads\Hela_DTASelect-filter.txt")

peptide_df['Sequence'] = [seq[2:-2] for seq in peptide_df['Sequence']]
peptide_df['Sequence'] = [add_static_mods(seq, {'C': 57}) for seq in peptide_df['Sequence']]

# align retention times / values
peptide_df = peptide_df[['Sequence', 'RetTime']]
peptide_df = peptide_df.groupby('Sequence').median().reset_index()

peptide_df['RetTime'] = peptide_df['RetTime']/max(peptide_df['RetTime'])
peptide_df['SeqLen'] = [len(seq) for seq in peptide_df['Sequence']]
peptide_df = peptide_df[peptide_df['SeqLen'] <= MAX_SEQ_LENGTH]

print(len(peptide_df))

train_df, test_df = split_df(peptide_df, sequence_column='Sequence', test_size=0.2, random_state=42)

X_train, Y_train = get_data(train_df, 'RetTime', MAX_SEQ_LENGTH+1)
X_test, Y_test = get_data(test_df, 'RetTime', MAX_SEQ_LENGTH+1)

model = train_model(X_train, X_test, Y_train, Y_test, use_charge_state=False)

