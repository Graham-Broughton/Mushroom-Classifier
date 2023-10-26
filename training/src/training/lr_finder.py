import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from tempfile import TemporaryFile


class LRFinder():

    def __init__(self, model, begin_lr=1e-8, end_lr=1e-1, num_epochs=10, period=5):
        # lr_finder generates learning schedules for finding upper and lower bounds on the best learning rate, as well as 
        # a cyclical learning rate schedule based on those bounds
        self.period = period
        # make a copy of the model to train through a sweep of learning rates
        self.f = TemporaryFile(suffix='h5')
        model.save_weights(self.f)
        self.model = model

        # define bounds to sweep through
        self.begin_lr = np.log(begin_lr)/np.log(10)
        self.end_lr = np.log(end_lr)/np.log(10)
        self.num_epochs = num_epochs
        self.lower_bound = begin_lr
        self.upper_bound = 1e-2  # end_lr
        # define learning rates to use in schedules
        self.lr = np.logspace(self.begin_lr, self.end_lr, self.num_epochs)
        self.clr = np.logspace(np.log(self.lower_bound)/np.log(10), np.log(self.upper_bound)/np.log(10), self.period)

    def reset_model(self, model):
        # reset the model to find new lr bounds
        self.begin_lr = -10
        self.end_lr = 0
        self.lr = np.logspace(self.begin_lr, self.end_lr, self.num_epochs)
        self.model = self.model.load_weights(self.f)

    def lr_schedule(self, epoch):
        # return lr according to a sweeping schedule
        if epoch < self.num_epochs:
            return self.lr[epoch]
        else:
            return self.lr[0]

    def clr_schedule(self, epoch, period=5):
        # return lr according to cyclical learning rate schedule
        my_epoch = int(epoch % self.period)
        return self.clr[my_epoch]

    def lr_vector(self, epochs):
        # return the vector of learning rates used in a schedule
        lrv = []
        for ck in range(epochs):
            lrv.append(self.lr_schedule(ck))
        return lrv

    def lr_plot(self, history_loss, please_plot=True):
        # plot the lr sweep results and set upper and lower bounds on learning rate
        x_axis = self.lr_vector(self.num_epochs)
        y_axis = history_loss

        d_loss = []
        for cc in range(1, len(y_axis)):
            if cc == 1:
                d_loss.append(y_axis[cc] - y_axis[cc-1])
            else:
                d_loss.append(0.8*(y_axis[cc] - y_axis[cc-1])+0.2*(y_axis[cc-1]-y_axis[cc-2]))
        d_loss = np.array(d_loss)

        self.lower_bound = x_axis[d_loss.argmin()]
        self.upper_bound = x_axis[np.array(y_axis).argmin()]
        self.clr = np.logspace(np.log(self.lower_bound)/np.log(10), np.log(self.upper_bound)/np.log(10), self.period)

        print("recommended learning rate: more than %.2e, less than %.2e " % (self.lower_bound, self.upper_bound))
        if (please_plot):
            plt.figure(figsize=(10, 5))
            plt.loglog(x_axis, y_axis)
            plt.xlabel('learning rate')
            plt.ylabel('loss')
            plt.title('Loss / learning rate progression')
            plt.show()

    def get_lr(self, epoch):
        # return the geometric mean of the upper and lower bound learning rates
        return (self.lower_bound * self.upper_bound)**(1/2)