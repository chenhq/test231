from keras.layers import LSTM, Dense, BatchNormalization, TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop


def construct_lstm_model(params, input_size, output_size):
    model = Sequential()
    model.add(LSTM(int(params['units1']),
                   return_sequences=True,
                   input_shape=(params['time_steps'], input_size),
                   dropout=params['dropout'],
                   recurrent_dropout=params['recurrent_dropout'],
                   kernel_initializer=params['initializer'],
                   bias_initializer=params['initializer']))
    if params['is_BN_1']:
        model.add(BatchNormalization())

    model.add(LSTM(params['units2'],
                   return_sequences=True,
                   dropout=params['dropout'],
                   recurrent_dropout=params['recurrent_dropout'],
                   kernel_initializer=params['initializer'],
                   bias_initializer=params['initializer']))
    if params['is_BN_2']:
        model.add(BatchNormalization())

    model.add(LSTM(params['units3'],
                   return_sequences=True,
                   dropout=params['dropout'],
                   recurrent_dropout=params['recurrent_dropout'],
                   kernel_initializer=params['initializer'],
                   bias_initializer=params['initializer']))
    if params['is_BN_3']:
        model.add(BatchNormalization())

    model.add(TimeDistributed(Dense(output_size,
                                    kernel_initializer=params['initializer'],
                                    bias_initializer=params['initializer'],
                                    activation='softmax')))

    model.compile(optimizer=RMSprop(lr=params['lr']), loss='categorical_crossentropy', metrics=['accuracy'])

    return model
