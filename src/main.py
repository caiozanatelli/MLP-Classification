import argparse
import numpy as np
from time import time
from os import getcwd
from neuralnet import NeuralNetwork
from os.path import join
from os.path import split
from keras.utils import np_utils
from ioutils import IOUtils
from utils import Utils

def parse_args():
    parser = argparse.ArgumentParser(description='Galaxy Classifier using Neural Networks')
    parser.add_argument('-i', '--input', action='store', type=str, required=True,
                        help='Input file path')
    parser.add_argument('-o', '--output', action='store', type=str, required=True,
                        help='Output file path')
    parser.add_argument('-s', '--seed', action='store', type=int, default=1,
                        help='Seed for random numbers generation')
    parser.add_argument('-n', '--neurons', action='store', type=int, default=17,
                        help='Number of neurons in each hidden layer')
    parser.add_argument('--hidden', action='store', type=int, default=1,
                        help='Number of hidden layers in the network')
    parser.add_argument('-a', '--activation', action='store', type=str,
                        choices=['relu', 'sigmoid', 'softmax'], default='relu',
                        help='Activation function for the neurons')
    parser.add_argument('-e', '--epoch', action='store', type=int, default=100,
                        help='Number of epoches for the network')
    parser.add_argument('-b', '--batch', action='store', type=int, default=10,
                        help='Batch size for the training phase')
    parser.add_argument('-l', '--learning-rate', action='store', type=float, default=0.01,
                        help='Network\' learning rate')
    parser.add_argument('-d', '--decay', action='store', type=float, default=0.00005,
                        help='Decay factor for the learning rate')
    
    args = parser.parse_args()
    return args



def main(args):
    ioutils = IOUtils()
    utils   = Utils()
    classes_dict = {'GALAXY':0, 'STAR':1, 'QSO':2}
    matrix   = ioutils.read_input(args.input)
    classes  = matrix[1:, -1]
    nclasses = len(classes_dict.keys())
    X = np.array(matrix[1:, :-1], dtype=float)
    examples, attributes = X.shape

    #TODO: One-Hot encoding
    Y = utils.to_categorical(utils.to_numerical(classes, classes_dict))
    neural_net = NeuralNetwork(args.hidden, args.neurons, nclasses, attributes,
                                args.activation, args.epoch, args.batch,
                                args.learning_rate, args.decay, args.seed)
    # Split dataset into train and validation data
    X_indexes = np.arange(examples)
    train_epochs = []
    valid_epochs = []
    times = []

    # Perfomr 3-Fold Cross Validation
    for train, valid in utils.kfold_cross_validation(X_indexes, 3, True):
        neural_net.build_model()
        t_start = time()
        train_err, valid_err = neural_net.fit_model(X, Y, train, valid)
        t_end = time()
        times.append(t_end - t_start)
        train_epochs.append(train_err)
        valid_epochs.append(valid_err)

    mean_time = np.mean(np.array(times), axis=0)
    mean_folds_train = np.mean(np.dstack((train_epochs[0], train_epochs[1], train_epochs[2])), axis=2)
    mean_folds_test  = np.mean(np.dstack((valid_epochs[0], valid_epochs[1], valid_epochs[2])), axis=2)

    np.savetxt(join(args.output, 'train_' + str(args.seed) + '.csv'), 
                    mean_folds_train, delimiter=',', fmt='%6f')
    with open(join(args.output, 'time_train_' + str(args.seed) + '.csv'), 'w') as fp:
        fp.write('%6f' % mean_time)
    np.savetxt(join(args.output, 'test_' + str(args.seed) + '.csv'), 
                    mean_folds_test, delimiter=',', fmt='%6f')

if __name__ == '__main__':
    args = parse_args()
    main(args)