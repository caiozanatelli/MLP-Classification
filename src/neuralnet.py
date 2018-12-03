import numpy as np
from keras import optimizers
from keras.layers import Dense
from keras.models import Sequential
from keras.initializers import lecun_uniform

class NeuralNetwork(object):

    def __init__(self, hidden_layers, nneurons, nclasses, nattr, 
                activation, epochs, batch_size, learning_rate, decay, seed):
        self.__nneurons = nneurons
        self.__hidden_layers = hidden_layers
        self.__epochs = epochs
        self.__input_dim = nattr
        self.__size_output = nclasses
        self.__activation  = activation
        self.__batch_size  = batch_size
        self.__learning_rate = learning_rate
        self.__decay = decay
        self.__seed  = seed

        self.__model = None
        np.random.seed(self.__seed)

    def build_model(self):
        model = Sequential()

        # Input Layer
        model.add(Dense(self.__nneurons, input_dim=self.__input_dim,
                kernel_initializer=lecun_uniform(self.__seed),
                activation=self.__activation))
        # Hidden Layer
        for i in range(self.__hidden_layers):
            model.add(Dense(self.__nneurons, 
                    kernel_initializer=lecun_uniform(self.__seed),
                    activation=self.__activation))
        # Output Layer
        model.add(Dense(self.__size_output, 
                kernel_initializer=lecun_uniform(self.__seed),
                activation='softmax'))

        # Define SGD (Stochastic Gradient Descent) model as optmizer for the model
        sgd = optimizers.SGD(lr=self.__learning_rate, decay=self.__decay)

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        self.__model = model

    def fit_model(self, X, Y, train, valid):
        history = self.__model.fit(X[train], Y[train], epochs=self.__epochs,
                                    batch_size=self.__batch_size, verbose=0)
        # Keep track of the accuracy for each epoch in the training phase
        epochs  = np.arange(0, self.__epochs)
        accbyep = np.column_stack((epochs, np.array(history.history['acc'])))
        result  = self.__model.evaluate(X[valid], Y[valid], verbose=0)
        valid_err = result[1]
        return accbyep, valid_err
