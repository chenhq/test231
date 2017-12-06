from keras.initializers import Orthogonal, glorot_uniform, he_uniform, lecun_uniform
from data_prepare import *
from model import *
from log_history import *
from collections import defaultdict
import pandas as pd


default_params = {
    'time_steps': 16,
    'batch_size': 256,
    'epochs': 800,

    'units1': 20,
    'units2': 20,
    'units3': 16,

    'is_BN_1': True,
    'is_BN_2': False,
    'is_BN_3': False,

    'lr': 0.00036589019672292165,
    'dropout': 0.3,
    'recurrent_dropout': 0.3,
    'initializer': glorot_uniform(seed=123)
}


def models_bagging(data, params, num_models, predict_column='label'):
    models = []

    for loop in range(num_models):
        train, validate, _ = split_data_set(data, params['batch_size'] * params['time_steps'], 4, 1, 0)
        X_train, Y_train = reform_X_Y(train, params['batch_size'], params['time_steps'])
        X_validate, Y_validate = reform_X_Y(validate, params['batch_size'], params['time_steps'])

        model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1])

        history = LogHistory()

        model.fit(X_train, Y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  verbose=0,
                  validation_data=(X_validate, Y_validate),
                  shuffle=False,
                  callbacks=[history])
        history.loss_plot('epoch')
        models.append(model)
    return models

def models_predict(models, X):
    for model in models:
        model.predict(X)
