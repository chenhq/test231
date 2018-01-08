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
    model.add(LSTM(int(params['layer1']['units']),
                   # activation=params['layer1']['activation'],
                   return_sequences=True,
                   input_shape=(params['time_steps'], input_size),
                   dropout=params['dropout'],
                   recurrent_dropout=params['recurrent_dropout'],
                   kernel_initializer=params['kernel_initializer'],
                   bias_initializer=params['bias_initializer']))
    if params['layer1']['is_BN']:
        model.add(BatchNormalization())

    model.add(LSTM(params['layer2']['units'],
                   # activation=params['layer2']['activation'],
                   return_sequences=True,
                   dropout=params['dropout'],
                   recurrent_dropout=params['recurrent_dropout'],
                   kernel_initializer=params['kernel_initializer'],
                   bias_initializer=params['bias_initializer']))
    if params['layer2']['is_BN']:
        model.add(BatchNormalization())

    model.add(LSTM(params['layer3']['units'],
                   # activation=params['layer3']['activation'],
                   return_sequences=True,
                   dropout=params['dropout'],
                   recurrent_dropout=params['recurrent_dropout'],
                   kernel_initializer=params['kernel_initializer'],
                   bias_initializer=params['bias_initializer']))
    if params['layer3']['is_BN']:
        model.add(BatchNormalization())

    model.add(TimeDistributed(Dense(output_size,
                                    kernel_initializer=params['kernel_initializer'],
                                    bias_initializer=params['bias_initializer'],
                                    activation=params['activation_last'])))

    model.compile(optimizer=RMSprop(lr=params['lr']), loss=loss, metrics=['accuracy'])

    return model


def model_predict(model, raw_data, X, performance_func, reset_status=False):
    if reset_status:
        Y_predict_list = []
        for i in range(len(X)):
            model.reset_states()
            Y_batch = model.predict(X[i].reshape(-1, X.shape[1], X.shape[2]))
            Y_predict_list.append(Y_batch)
        Y_predict = np.concatenate(Y_predict_list)
    else:
        Y_predict = model.predict(X)

    Y_predict = np.reshape(Y_predict, (-1, Y_predict.shape[-1]))
    performances = performance_func(raw_data['pct_chg'], Y_predict)
    return performances


