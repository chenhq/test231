import seaborn as snb
from hyperopt import hp, STATUS_OK
from keras.initializers import glorot_uniform
import datetime

from data_prepare import *
from log_history import *
from model import *

try:
    import cPpickle as pickle
except:
    import pickle
snb.set()

default_space = {
    'time_steps': hp.choice('time_steps', [32, 64, 128]),
    'batch_size': hp.choice('batch_size', [64, 128]),
    'epochs': hp.choice('epochs', [100, 200, 300, 400, 500]),  # [100, 200, 500, 1000, 1500, 2000]
    'activation': hp.choice('activation', ['relu', 'sigmoid', 'tanh', 'linear']),
    'shuffle': hp.choice('shuffle', [False, True]),

    'units1': hp.choice('units1', [32, 64, 128, 256]),
    'units2': hp.choice('units2', [32, 64, 128, 256]),
    'units3': hp.choice('units3', [32, 64, 128, 256]),

    'is_BN_1': hp.choice('is_BN_1', [False, True]),
    'is_BN_2': hp.choice('is_BN_2', [False, True]),
    'is_BN_3': hp.choice('is_BN_3', [False, True]),

    'lr': hp.uniform('lr', 0.0001, 0.001),
    'dropout': hp.choice('dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'recurrent_dropout': hp.choice('recurrent_dropout', [0, 0.1, 0.2, 0.3, 0.4, 0.5]),
    'initializer': hp.choice('initializer', [glorot_uniform(seed=123)]),
    # 'min_delta': hp.quniform('min_delta', 0.0002, 0.001, 0.0002),
    # 'patience': hp.quniform('patience', 10, 100, 10),
}


def construct_objective(data_set, target_field, namespace, performance_func, measure,
                        loss='categorical_crossentropy', include_test_data=False, shuffle_test=False):
    def objective(params):
        identity = str(uuid.uuid1())
        print("time: {0}, identity: {1}, params: {2}".format(datetime.datetime.now(), identity, params))

        log_dir = os.path.join(namespace, identity)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        params_file = os.path.join(log_dir, "params.pkl")
        with open(params_file, 'wb') as output:
            pickle.dump(params, output)

        train, validate = data_set['train'], data_set['validate']

        minimum_size = params['time_steps']
        train_available_length = len(train) // minimum_size * minimum_size
        train = train.tail(train_available_length)
        validate_available_length = len(validate) // minimum_size * minimum_size
        validate = validate.tail(validate_available_length)

        X_train, Y_train = reform_X_Y(train, params['time_steps'], target_field)
        X_validate, Y_validate = reform_X_Y(validate, params['time_steps'], target_field)

        model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1], loss=loss)
        log_histroy = LogHistory(os.path.join(log_dir, 'history.pkl'))
        # early_stop = EarlyStopping(monitor='val_loss', min_delta=params['min_delta'], patience=params['patience'],
        #                            verbose=2, mode='auto')
        model.fit(X_train, Y_train,
                  batch_size=params['batch_size'],
                  epochs=params['epochs'],
                  verbose=0,
                  validation_data=(X_validate, Y_validate),
                  shuffle=params['shuffle'],
                  callbacks=[log_histroy])  # early_stop

        to_be_predict_set = {}
        to_be_predict_set['validate'] = [validate, X_validate]

        if include_test_data:
            test = data_set['test']
            test_available_length = len(test) // minimum_size * minimum_size
            test = test.tail(test_available_length)
            X_test, _ = reform_X_Y(test, params['time_steps'], target_field)

            to_be_predict_set['test'] = [test, X_test]

        loss_value = 0
        for tag in to_be_predict_set:
            performances = model_predict(model, to_be_predict_set[tag][0], to_be_predict_set[tag][1],
                                         tag, log_dir, performance_func)
            if tag == 'validate':
                loss_value = -performances[measure]

        # add for test reset_status
        # for tag in to_be_predict_set:
        #     reset_status_tag = tag + '_reset_status'
        #     model_predict(model, to_be_predict_set[tag][0], to_be_predict_set[tag][1],
        #                                  reset_status_tag, log_dir, performance_func, reset_status=True)

        if shuffle_test:
            feature_columns = validate.columns.tolist()
            feature_columns.remove(target_field)

            for column in feature_columns:
                validate_shuffle = validate.copy()
                np.random.shuffle(validate_shuffle[column].values)
                X_validate_shuffle, Y_validate_shuffle = reform_X_Y(validate_shuffle, params['time_steps'],
                                                                    target_field)
                tag = "shuffle_" + column
                model_predict(model, validate_shuffle, X_validate_shuffle, tag, log_dir,
                              performance_func)

        print("identity: {0}, loss: {1}".format(identity, loss_value))
        return {'loss': loss_value, 'status': STATUS_OK}

    return objective
