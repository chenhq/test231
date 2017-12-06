import keras
import uuid
import os
import pandas as pd


class LogHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        super(LogHistory, self).__init__()

    def on_train_begin(self, logs={}):
        log_name = str(uuid.uuid1()) + ".log"
        self.log_file = os.path.join(self.log_dir, log_name)
        columns = ['loss', 'acc', 'val_loss', 'val_acc']
        self.epoch_history_data = pd.DataFrame(columns=columns)

    def on_train_end(self, logs=None):
        self.epoch_history_data.to_csv(self.log_file, index=False)

    # def on_batch_end(self, batch, logs={}):
    #
    #     self.losses['batch'].append(logs.get('loss'))
    #     self.accuracy['batch'].append(logs.get('acc'))
    #     self.val_loss['batch'].append(logs.get('val_loss'))
    #     self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        performance = [logs.get('loss'), logs.get('acc'), logs.get('val_loss'), logs.get('val_acc')]
        self.epoch_history_data.loc[len(self.epoch_history_data)] = performance
