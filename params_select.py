import matplotlib.pyplot as plt
from hyperopt import hp, STATUS_OK
from keras.initializers import glorot_uniform
from keras.callbacks import EarlyStopping
from data_prepare import *
from log_history import *
from model import *
import seaborn as snb

snb.set()

default_space = {
    'time_steps': hp.choice('time_steps', [16, 32, 64, 128]),
    'batch_size': hp.choice('batch_size', [2, 4, 8, 16, 32, 64, 128]),
    'epochs': hp.choice('epochs', [100, 300, 500]),  # [100, 200, 500, 1000, 1500, 2000]
    'shuffle':  hp.choice('shuffle', [False, True]),

    'units1': hp.choice('units1', [16, 32, 64, 128, 256]),
    'units2': hp.choice('units2', [16, 32, 64, 128, 256]),
    'units3': hp.choice('units3', [16, 32, 64, 128, 256]),

    'is_BN_1': hp.choice('is_BN_1', [False, True]),
    'is_BN_2': hp.choice('is_BN_2', [False, True]),
    'is_BN_3': hp.choice('is_BN_3', [False, True]),

    'lr': hp.uniform('lr', 0.0001, 0.001),
    'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'recurrent_dropout': hp.choice('recurrent_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'initializer': hp.choice('initializer', [glorot_uniform(seed=123)]),
    'min_delta': hp.quniform('min_delta', 0.0002, 0.001, 0.002),
    'patience': hp.quniform('patience', 10, 100, 10),
}


# def construct_objective1(data, namespace, performance_func, loops=10):
#     def objective(params):
#         identity = str(uuid.uuid1())
#         print("identity: {0}, params: {1}".format(identity, params))
#
#         log_dir = os.path.join(namespace, identity)
#         if not os.path.exists(log_dir):
#             os.mkdir(log_dir)
#
#         params_file = os.path.join(log_dir, "params.txt")
#         with open(params_file, 'w') as output:
#             output.write(str(params))
#
#         all_cum_returns = pd.DataFrame()
#         measures = np.array([])
#
#         for loop in range(loops):
#             train, validate, test = split_data_by_sample(data, params['batch_size'] * params['time_steps'], 3, 1, 1)
#             X_train, Y_train = reform_X_Y(train, params['batch_size'], params['time_steps'])
#             X_validate, Y_validate = reform_X_Y(validate, params['batch_size'], params['time_steps'])
#             X_test, Y_test = reform_X_Y(test, params['batch_size'], params['time_steps'])
#             model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1])
#             log_histroy = LogHistory(os.path.join(log_dir, 'history.log'))
#             model.fit(X_train, Y_train,
#                       batch_size=params['batch_size'],
#                       epochs=params['epochs'],
#                       verbose=0,
#                       validation_data=(X_validate, Y_validate),
#                       shuffle=params['shuffle'],
#                       callbacks=[log_histroy])
#
#             # loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=params['batch_size'])
#             Y_test_predict = model.predict(X_test)
#             Y_test_predict = np.reshape(Y_test_predict, (-1, Y_test_predict.shape[-1]))
#             cum_returns, measure = performance_func(validate['pct_chg'], Y_test_predict)
#             measures = np.append(measures, measure)
#             all_cum_returns = pd.concat([all_cum_returns, cum_returns], axis=1)
#
#         all_cum_returns.columns = [str(loop) for loop in range(loops)]
#         all_cum_returns = all_cum_returns.ffill().fillna(0)
#         all_cum_returns['mean'] = all_cum_returns.mean(axis=1)
#
#         cum_returns_plot_file = os.path.join(log_dir, "cum_returns.png")
#         ax = all_cum_returns.plot()
#         fig = ax.get_figure()
#         plt.legend()
#         fig.savefig(cum_returns_plot_file)
#         plt.close()
#
#         loss = -measures.mean()
#         print("loss: {0}".format(loss))
#         return {'loss': loss, 'status': STATUS_OK}
#
#     return objective


def construct_objective2(data_set, namespace, performance_func, measure):
    def objective(params):
        identity = str(uuid.uuid1())
        print("identity: {0}, params: {1}".format(identity, params))

        log_dir = os.path.join(namespace, identity)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        params_file = os.path.join(log_dir, "params.txt")
        with open(params_file, 'w') as output:
            output.write(str(params))

        train, validate = data_set['train'], data_set['validate']

        minimum_size = params['batch_size'] * params['time_steps']
        train_available_length = len(train) // minimum_size * minimum_size
        train = train.tail(train_available_length)
        validate_available_length = len(validate) // minimum_size * minimum_size
        validate = validate.tail(validate_available_length)

        X_train, Y_train = reform_X_Y(train, params['time_steps'])
        X_validate, Y_validate = reform_X_Y(validate, params['time_steps'])
        model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1])
        log_histroy = LogHistory(os.path.join(log_dir, 'history.log'))
        early_stop = EarlyStopping(monitor='val_loss', min_delta=params['min_delta'], patience=params['patience'], verbose=2, mode='auto')
        model.fit(X_train, Y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  verbose=0,
                  validation_data=(X_validate, Y_validate),
                  shuffle=params['shuffle'],
                  callbacks=[log_histroy, early_stop])

        Y_validate_predict = model.predict(X_validate)
        Y_validate_predict = np.reshape(Y_validate_predict, (-1, Y_validate_predict.shape[-1]))
        performances = performance_func(validate['pct_chg'], Y_validate_predict)

        if 'returns' in performances:
            performances['returns'].to_csv(os.path.join(log_dir, 'returns.log'))
        if 'cum_returns' in performances:
            performances['cum_returns'].to_csv(os.path.join(log_dir, 'cum_returns.log'))
        if 'annual_return' in performances:
            with open(os.path.join(log_dir, 'annual_return.log'), 'w') as output:
                output.write(str(performances['annual_return']))
        if 'sharpe_ratio' in performances:
            with open(os.path.join(log_dir, 'sharpe_ratio.log'), 'w') as output:
                output.write(str(performances['sharpe_ratio']))

        loss = -performances[measure]
        print("identity: {0}, loss: {1}".format(identity, loss))
        return {'loss': loss, 'status': STATUS_OK}
    return objective

