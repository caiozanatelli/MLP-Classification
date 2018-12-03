from keras.utils import np_utils
import numpy as np

class Singleton(type):

    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance

class Utils(object):
    __metaclass__ = Singleton

    def to_numerical(self, var, dic):
        for loc, num in dic.items():
            var[np.where(var == loc)] = num
        return np.array(var, dtype=float)

    def to_categorical(self, array):
        return np_utils.to_categorical(array)

    def kfold_cross_validation(self, X, K, randomise=False):
        if randomise:
            np.random.shuffle(X)
        for k in range(K):
            size = len(X)
            validation_indexes = np.arange(start=k, stop=size, step=K)
            indexes = np.setdiff1d(np.arange(size), validation_indexes)
            yield indexes, validation_indexes
