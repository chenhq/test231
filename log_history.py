import keras
import uuid
import numpy as np
import os
import pandas as pd
try:
    import _pickle as pickle
except:
    import pickle


class LogHistory(keras.callbacks.Callback):
    def __init__(self, filename):
        self.filename = filename
        super(LogHistory, self).__init__()

    def set_filename(self, filename):
        self.filename = filename

    def on_train_begin(self, logs={}):
        columns = ['loss', 'acc', 'val_loss', 'val_acc']
        self.epoch_history_data = pd.DataFrame(columns=columns)

    def on_train_end(self, logs=None):
        dirs = os.path.dirname(self.filename)
        if not os.path.exists(dirs):
            os.makedirs(dirs)
        with open(self.filename, 'wb') as output:
            pickle.dump(self.epoch_history_data, output)

        # do not draw
        # if len(self.epoch_history_data) > 0:
        #     history_plot_file = os.path.splitext(self.filename)[0]+'.png'
        #     ax = self.epoch_history_data.plot()
        #     fig = ax.get_figure()
        #     plt.legend()
        #     fig.savefig(history_plot_file)
        #     plt.close()


    # def on_batch_end(self, batch, logs={}):
    #
    #     self.losses['batch'].append(logs.get('loss'))
    #     self.accuracy['batch'].append(logs.get('acc'))
    #     self.val_loss['batch'].append(logs.get('val_loss'))
    #     self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, epoch, logs={}):
        performance = [logs.get('loss'), logs.get('acc'), logs.get('val_loss'), logs.get('val_acc')]
        self.epoch_history_data.loc[len(self.epoch_history_data)] = performance


class LogModel(keras.callbacks.Callback):
    def __init__(self, log_dir, measure, measure_values, mode):
        self.log_dir = log_dir
        self.measure = measure
        self.measure_values = measure_values
        if mode == 'min':
            self.monitor_op = np.less
            self.measure_values.sort(reversed=True)
        elif mode == 'max':
            self.monitor_op = np.greater
            self.measure_values.sort()
        super(LogModel, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        if self.monitor_op(logs.get(self.measure), self.measure_values[0]):
            file_name = os.path.join(self.log_dir, '{}_{}.h5'.format(self.measure, self.measure_values[0]))
            print('\n log model into file {}....'.format(file_name))
            self.model.save(file_name)
            del self.measure_values[0]
        if len(self.measure_values) < 1:
            self.model.stop_training = True

