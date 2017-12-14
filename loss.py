from functools import partial
from itertools import product

import keras.backend as K
import numpy as np

'''
weights formant:
actual                   predicted
   0                1.0,    2.0,    4.0
   1                2.0,    1.0,    2.0
   2                4.0,    2.0,    1.0
'''


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[..., 0])
    y_pred_max = K.max(y_pred, axis=-1)
    y_pred_max = K.expand_dims(y_pred_max, -1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (
            K.cast(weights[c_t, c_p], K.floatx()) * K.cast(y_pred_max_mat[..., c_p], K.floatx()) * K.cast(y_true[..., c_t],
                                                                                                        K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


w_array = np.array(
    [[1.0, 2.0, 4.0],
     [2.0, 1.0, 2.0],
     [4.0, 2.0, 1.0]]
)
weighted_loss = partial(w_categorical_crossentropy, weights=w_array)
weighted_loss.__name__ = 'w_categorical_crossentropy'
