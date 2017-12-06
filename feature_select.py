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


def feature_score(data, params, loops, predict_column='label'):
    feature_columns = data.columns
    feature_columns = feature_columns.remove(predict_column)
    scores = pd.DataFrame(index=feature_columns, column=list(range(loops)))

    for loop in range(loops):
        train, validate, _ = split_data_set(data, params['batch_size'] * params['time_steps'], 3, 2, 0)
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

        loss_and_metrics = model.evaluate(X_validate, Y_validate, batch_size=params['batch_size'])

        # total_profit = profit_pred[-1]
        # if total_profit <= 0:
        #     print("参数不合适，收益为负")
        #     break
        for column in feature_columns:
            validate_shuffle = validate.copy()
            np.random.shuffle(validate_shuffle[column].values)
            X_validate_shuffle, Y_validate_shuffle = reform_X_Y(validate_shuffle, params['batch_size'], params['time_steps'])
            loss_and_metrics_shuffle = model.evaluate(X_validate_shuffle, Y_validate_shuffle, batch_size=params['batch_size'])
            scores.loc[column][loop] = (loss_and_metrics[0] - loss_and_metrics_shuffle[0]) / loss_and_metrics[0]

    return scores
