import keras
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv1D, Bidirectional, MaxPooling1D, GRU, \
    BatchNormalization
from tensorflow.keras.models import Sequential
from keras.layers import Input, Concatenate
from keras.models import Model



def build_default_model_big(input_shape) -> keras.Model:
    """
    Builds a default model for peptide property prediction.
    """
    model = Sequential()
    model.add(Conv1D(512, 3, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # model.add(Bidirectional(CuDNNGRU(50, return_sequences=True)))
    model.add(Bidirectional(GRU(50, return_sequences=True)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    return model


def build_default_model_small(input_shape) -> keras.Model:
    """
    Builds a default model for peptide property prediction.
    """
    model = Sequential()
    model.add(Conv1D(128, 3, padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(64, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))

    # model.add(Bidirectional(CuDNNGRU(50, return_sequences=True)))
    model.add(Bidirectional(GRU(25, return_sequences=True)))
    # model.add(SeqSelfAttention(attention_activation='sigmoid'))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.45))

    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    return model

def build_charge_model_small(input_shape) -> keras.Model:
    """
    Builds a default model for peptide property prediction with sequence and charge state inputs.
    """
    sequence_input = Input(shape=input_shape, name='sequence_input')
    charge_input = Input(shape=(1,), name='charge_input')

    x = Conv1D(128, 3, padding='same')(sequence_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(64, 3, padding='same')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(GRU(25, return_sequences=True))(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)

    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.45)(x)

    x = Concatenate()(
        [x, charge_input])  # concatenate the output of the sequence input processing and the charge input

    x = Dense(64)(x)
    x = Activation('relu')(x)
    x = Dropout(0.45)(x)

    output = Dense(1, kernel_initializer='normal', activation='sigmoid')(x)  # final output layer

    model = Model(inputs=[sequence_input, charge_input], outputs=output)  # create model with two inputs

    return model


