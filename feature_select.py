from keras.initializers import Orthogonal, glorot_uniform, he_uniform, lecun_uniform
from data_prepare import *
from model import *
from log_history import *
import pandas as pd
import matplotlib.pyplot as plt


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


def train(params, data, namespace, performance_func):
    train, validate, test = split_data_set(data, params['batch_size'] * params['time_steps'], 3, 1, 1)
    X_train, Y_train = reform_X_Y(train, params['batch_size'], params['time_steps'])
    X_validate, Y_validate = reform_X_Y(validate, params['batch_size'], params['time_steps'])
    X_test, Y_test = reform_X_Y(test, params['batch_size'], params['time_steps'])
    model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1])
    log_histroy = LogHistory(namespace)
    model.fit(X_train, Y_train,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=0,
              validation_data=(X_validate, Y_validate),
              shuffle=False,
              callbacks=[log_histroy])

    # loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=params['batch_size'])
    Y_test_predict = model.predict(X_test)
    Y_test_predict = np.reshape(Y_test_predict, (-1, Y_test_predict.shape[-1]))
    cum_returns, measure = performance_func(validate['pct_chg'], Y_test_predict)
    return cum_returns, measure


def feature_score(data, params, loops, performance_func, predict_column='label', namespace='score'):
    feature_columns = data.columns
    feature_columns = feature_columns.remove(predict_column)
    loss_scores = pd.DataFrame(column=feature_columns)
    measures_scores = pd.DataFrame(column=feature_columns)
    all_cum_returns = pd.DataFrame()

    for loop in range(loops):
        loss_scores.iloc[loop] = np.full(len(feature_columns), 0)
        measures_scores.iloc[loop] = np.full(len(feature_columns), 0)

        train, validate, _ = split_data_set(data, params['batch_size'] * params['time_steps'], 3, 2, 0)
        X_train, Y_train = reform_X_Y(train, params['batch_size'], params['time_steps'])
        X_validate, Y_validate = reform_X_Y(validate, params['batch_size'], params['time_steps'])
        model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1])

        log_histroy = LogHistory(namespace)

        model.fit(X_train, Y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  verbose=0,
                  validation_data=(X_validate, Y_validate),
                  shuffle=False,
                  callbacks=[log_histroy])

        loss_and_metrics = model.evaluate(X_validate, Y_validate, batch_size=params['batch_size'])

        Y_validate_predict = model.predict(X_validate)
        Y_validate_predict = np.reshape(Y_validate_predict, (-1, Y_validate_predict.shape[-1]))
        cum_returns, measure = performance_func(validate['pct_chg'], Y_validate_predict)
        all_cum_returns = pd.concat([all_cum_returns, cum_returns], axis=1)

        # total_profit = profit_pred[-1]
        # if total_profit <= 0:
        #     print("参数不合适，收益为负")
        #     break

        for column in feature_columns:
            validate_shuffle = validate.copy()
            np.random.shuffle(validate_shuffle[column].values)
            X_validate_shuffle, Y_validate_shuffle = reform_X_Y(validate_shuffle, params['batch_size'],
                                                                params['time_steps'])
            loss_and_metrics_shuffle = model.evaluate(X_validate_shuffle, Y_validate_shuffle,
                                                      batch_size=params['batch_size'])

            Y_validate_shuffle_predict = model.predict(X_validate_shuffle)
            Y_validate_shuffle_predict = np.reshape(Y_validate_shuffle_predict, (-1, Y_validate_shuffle_predict.shape[-1]))
            _, measure_shuffle = performance_func(validate['pct_chg'], Y_validate_shuffle_predict)

            loss_scores.iloc[loop][column] = (loss_and_metrics[0] - loss_and_metrics_shuffle[0]) / loss_and_metrics[0]
            measures_scores.iloc[loop][column] = (measure - measure_shuffle) / measure

    all_cum_returns.columns = [str(loop) for loop in range(loops)]
    all_cum_returns = all_cum_returns.ffill().fillna(0)
    all_cum_returns['mean'] = all_cum_returns.mean(axis=1)
    cum_returns_plot_file = os.path.join(namespace, "cum_returns.png")
    ax = all_cum_returns.plot()
    fig = ax.get_figure()
    plt.legend()
    fig.savefig(cum_returns_plot_file)
    plt.close()

    return loss_scores, measures_scores
