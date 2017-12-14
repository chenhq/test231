from keras.layers import LSTM, Dense, BatchNormalization, TimeDistributed
from keras.models import Sequential
from keras.optimizers import RMSprop
import numpy as np
import os
try:
    import _pickle as pickle
except:
    import pickle


def construct_lstm_model(params, input_size, output_size, loss='categorical_crossentropy'):
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

    model.compile(optimizer=RMSprop(lr=params['lr']), loss=loss, metrics=['accuracy'])

    return model


def model_predict(model, raw_data, X, tag, log_dir, performance_func):
    Y_predict = model.predict(X)
    Y_predict = np.reshape(Y_predict, (-1, Y_predict.shape[-1]))
    performances = performance_func(raw_data['pct_chg'], Y_predict)
    with open(os.path.join(log_dir, '%s.pkl' % tag), 'wb') as output:
        pickle.dump(performances, output)
    return performances



