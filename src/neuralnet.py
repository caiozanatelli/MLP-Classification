import numpy as np

class NeuralNetwork(object):

    def __init__(self, hidden_layers, nneurons, nclasses, nattr, 
                activation, epoch, batch_size, learning_rate, seed):
        self.__hidden_layers = hidden_layers
        self.__nneurons = nneurons
        self.__nclasses = nclasses
        self.__nattr = nattr
        self.__activation = activation
        self.__epoch = epoch
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__seed = seed
        np.random.seed(self.__seed)

    def build_model(self):
        pass

    def fit_model(self, X, Y, train, valid):
        pass
