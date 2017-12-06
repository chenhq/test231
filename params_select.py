import matplotlib.pyplot as plt
from hyperopt import hp, STATUS_OK
from keras.initializers import glorot_uniform

from data_prepare import *
from log_history import *
from model import *
import seaborn as snb
snb.set()

default_space = {
    'time_steps': hp.choice('time_steps', [8, 16, 32]),
    'batch_size': hp.choice('batch_size', [2, 4, 8]),
    'epochs': hp.choice('epochs', [100, 200]),  # [100, 200, 500, 1000, 1500, 2000]

    'units1': hp.choice('units1', [8, 16, 32, 64]),
    'units2': hp.choice('units2', [8, 16, 32, 64]),
    'units3': hp.choice('units3', [8, 16, 32, 64]),

    'is_BN_1': hp.choice('is_BN_1', [False, True]),
    'is_BN_2': hp.choice('is_BN_2', [False, True]),
    'is_BN_3': hp.choice('is_BN_3', [False, True]),

    'lr': hp.uniform('lr', 0.0001, 0.001),
    'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'recurrent_dropout': hp.choice('recurrent_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'initializer': hp.choice('initializer', [glorot_uniform(seed=123)])
}


def construct_objective1(data, namespace, performance_func=None, loops=10):
    def objective(params):
        print(params)
        log_dir = os.path.join(namespace, str(uuid.uuid1()))
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        params_file = os.path.join(log_dir, "params.txt")
        with open(params_file, 'w') as output:
            output.write(str(params))

        performances = pd.DataFrame()

        for loop in range(loops):
            train, validate, test = split_data_set(data, params['batch_size'] * params['time_steps'], 3, 1, 1)
            X_train, Y_train = reform_X_Y(train, params['batch_size'], params['time_steps'])
            X_validate, Y_validate = reform_X_Y(validate, params['batch_size'], params['time_steps'])
            X_test, Y_test = reform_X_Y(test, params['batch_size'], params['time_steps'])
            model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1])
            log_histroy = LogHistory(log_dir)
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
            performance = performance_func(validate['pct_chg'], Y_test_predict)

            performances = pd.concat([performances, performance], axis=1)

        performances.columns = [str(loop) for loop in range(loops)]
        performances = performances.ffill().fillna(1)
        performances['mean'] = performances.mean(axis=1)

        performances_plot_file = os.path.join(log_dir, "performances.png")
        ax = performances.plot()
        fig = ax.get_figure()
        plt.legend()
        fig.savefig(performances_plot_file)

        # plt.savefig(performances_plot_file)
        # plt.close()

        # profit_pred, _, _ = utils.profit(model.predict(X_validate), cls_validate)
        # utils.plot_curve(result_file, Y_validate, model.predict(X_validate), cls_validate, 0)
        # utils.plot_curve(result_file, Y_test, model.predict(X_test), cls_test, 1)
        # return {'loss': -profit_pred[-1], 'status': STATUS_OK}
        # return {'loss': loss_and_metrics[0], 'status': STATUS_OK}
        loss = -performances.iloc[-1]['mean']
        print("loss: {0}".format(loss))
        return {'loss': loss, 'status': STATUS_OK}

    return objective

# space = default_space
# objective_func = construct_objective(features_categorical)
# trials = Trials()
# best = fmin(objective_func, space, algo=tpe.suggest, max_evals=40, trials=trials)
# print(best)
