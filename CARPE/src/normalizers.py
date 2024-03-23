import numpy as np
from scipy import signal

from IPython import embed

class SavGolFilter(object):
    def __init__(self, window_length: int=71, polyorder: int=8):
        self.wl = window_length
        self.po = polyorder

    def transform(self, ischemia_dataset, X, channel):
        X[channel, :] = signal.savgol_filter(X[channel, :], self.wl, self.po)

class QNormalizer(object):
    def __init__(self, q_low: float=0.005, q_high: float=0.995):
        self.q_low = q_low
        self.q_high = q_high

    def transform(self, ischemia_dataset, X, channel):
        q_high = np.quantile(X[channel, :], self.q_high)
        q_low = np.quantile(X[channel, :], self.q_low)
        X[channel, :][np.where(X[channel, :] > q_high)] = q_high
        X[channel, :][np.where(X[channel, :] < q_low)] = q_low


class MinMaxNormalizer(object):
    def transform(self, ischemia_dataset, X, channel):
        min_ = np.min(X[channel, :])
        max_ = np.max(X[channel, :])
        X[channel, :] = (X[channel, :] - min_)/(max_ - min_)


class ZNormalizer(object):
    def __init__(self):
        self.is_fitted = False
        print('init')

    def transform(self, X):
        X = np.asarray(X)
        if self.is_fitted:
            X -= np.repeat(self.means, X.shape[2]).reshape(X.shape[1:])
            X /= np.repeat(self.stds, X.shape[2]).reshape(X.shape[1:])
        else:
            # Compute the mean/std first over all samples, then over the
            # complete time series.
            self.means = X.mean(axis=0).mean(axis=1)
            self.stds = X.mean(axis=0).std(axis=1)
            X -= np.repeat(self.means, X.shape[2]).reshape(X.shape[1:])
            X /= np.repeat(self.stds, X.shape[2]).reshape(X.shape[1:])
            
            self.is_fitted = True
        return X
