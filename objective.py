import seaborn as snb
from hyperopt import hp, STATUS_OK
from keras.initializers import glorot_uniform
import datetime
from loss import weighted_categorical_crossentropy3, weighted_categorical_crossentropy5
from feature.construct_feature import feature_kline, feature_ma, label_by_ma_price, construct_features
from functools import partial
from performance import *
from keras.callbacks import EarlyStopping

from data_prepare import *
from log_history import *
from model import *

try:
    import cPpickle as pickle
except:
    import pickle
snb.set()


def lstm_objective(params, data_set, target_field, namespace, performance_func, measure,
                   include_test_data=False, shuffle_test=False):
    if not os.path.exists(namespace):
        os.makedirs(namespace)

    train, validate = data_set['train'], data_set['validate']

    minimum_size = params['time_steps']
    train_available_length = len(train) // minimum_size * minimum_size
    train = train.tail(train_available_length)
    validate_available_length = len(validate) // minimum_size * minimum_size
    validate = validate.tail(validate_available_length)

    X_train, Y_train = reform_X_Y(train, params['time_steps'], target_field)
    X_validate, Y_validate = reform_X_Y(validate, params['time_steps'], target_field)

    # default loss
    loss = 'categorical_crossentropy'
    if params['loss_type'] == 'weighted_categorical_crossentropy':
        loss = weighted_categorical_crossentropy5

    model = construct_lstm_model(params, X_train.shape[-1], Y_train.shape[-1], loss=loss)
    log_histroy = LogHistory(os.path.join(namespace, 'history.pkl'))
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=100,
                               verbose=2, mode='auto')
    model.fit(X_train, Y_train,
              batch_size=params['batch_size'],
              epochs=params['epochs'],
              verbose=1,
              validation_data=(X_validate, Y_validate),
              shuffle=params['shuffle'],
              callbacks=[log_histroy, early_stop])

    to_be_predict_set = {}
    to_be_predict_set['validate'] = [validate, X_validate, Y_validate]

    if include_test_data:
        test = data_set['test']
        test_available_length = len(test) // minimum_size * minimum_size
        test = test.tail(test_available_length)
        X_test, Y_test = reform_X_Y(test, params['time_steps'], target_field)

        to_be_predict_set['test'] = [test, X_test, Y_test]

    performances = {}
    for tag in to_be_predict_set:
        performances[tag] = model_predict(model, to_be_predict_set[tag][0], to_be_predict_set[tag][1],
                                          performance_func)
        scores = model.evaluate(to_be_predict_set[tag][1], to_be_predict_set[tag][2], verbose=1)
        performances[tag]['loss'] = scores[0]
        performances[tag]['metrics'] = scores[1]

    with open(os.path.join(namespace, 'performances.pkl'), 'wb') as output:
        pickle.dump(performances, output)

    if measure in ['loss']:
        loss_value = performances['validate'][measure]
    else:
        loss_value = -performances['validate'][measure]

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
            model_predict(model, validate_shuffle, X_validate_shuffle, tag, namespace,
                          performance_func)

    print("namespace: {0}, loss: {1}".format(namespace, loss_value))
    return {'loss': loss_value, 'status': STATUS_OK}


def features_objective(params, ohlcv_list):
    params_list = []
    func_list = []

    # k line
    if 'kline' in params:
        kline_params = params['kline']
        params_list.append(kline_params)
        func_list.append(feature_kline)

    # ma
    if 'ma' in params:
        ma_params = params['ma']
        params_list.append(ma_params)
        func_list.append(feature_ma)

    # label
    if 'label_by_ma_price' in params:
        label_by_ma_price_params = params['label_by_ma_price']
        params_list.append(label_by_ma_price_params)
        func_list.append(label_by_ma_price)

    construct_feature_func = partial(construct_features, params_list=params_list, func_list=func_list, test=False)
    stk_features_list = construct_features_for_stocks(ohlcv_list, construct_feature_func)
    return stk_features_list


def objective(params, ohlcv_list, namespace):
    identity = str(uuid.uuid1())
    print('time: {}, identity: {}, params: {}'.format(datetime.datetime.now(), identity, params))
    sub_namespace = os.path.join(namespace, identity)

    if not os.path.exists(sub_namespace):
        os.makedirs(sub_namespace)

    params_file = os.path.join(sub_namespace, 'params.pkl')
    pickle.dump(params, open(params_file, 'wb'))

    features_list = features_objective(params['features'], ohlcv_list)
    features_list_file = os.path.join(sub_namespace, 'features_list.pkl')
    pickle.dump(features_list, open(features_list_file, 'wb'))

    data_set = split_data_set_by_date(features_list, params['split_dates'], minimum_size=64)
    data_set_file = os.path.join(sub_namespace, 'data_set.pkl')
    pickle.dump(data_set, open(data_set_file, 'wb'))

    quantile_list = params['features']['label_by_ma_price']['quantile_list']
    class_list = [i for i in range(len(quantile_list)-1)]
    nb_class = len(class_list)
    _, reverse_categorical = categorical_factory(class_list)
    performance_func = performance_factory(reverse_categorical,
                                           performance_types=['Y0', 'Y', 'returns', 'cum_returns', 'annual_return',
                                                              'sharpe_ratio'],
                                           mid_type=(nb_class-1) / 2.0, epsilon=0.6)
    return lstm_objective(params['lstm'], data_set, target_field='label', namespace=sub_namespace,
                          performance_func=performance_func, measure='loss', include_test_data=True,
                          shuffle_test=False)
