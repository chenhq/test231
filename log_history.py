import keras
import uuid
import os
import pandas as pd
import matplotlib.pyplot as plt


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
        self.epoch_history_data.to_csv(self.filename)

        if len(self.epoch_history_data) > 0:
            history_plot_file = os.path.splitext(self.filename)[0]+'.png'
            ax = self.epoch_history_data.plot()
            fig = ax.get_figure()
            plt.legend()
            fig.savefig(history_plot_file)
            plt.close()


    # def on_batch_end(self, batch, logs={}):
    #
    #     self.losses['batch'].append(logs.get('loss'))
    #     self.accuracy['batch'].append(logs.get('acc'))
    #     self.val_loss['batch'].append(logs.get('val_loss'))
    #     self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, epoch, logs={}):
        performance = [logs.get('loss'), logs.get('acc'), logs.get('val_loss'), logs.get('val_acc')]
        self.epoch_history_data.loc[len(self.epoch_history_data)] = performance
