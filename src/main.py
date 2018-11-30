import argparse
import numpy as np
from time import time
from os import getcwd
from neuralnet import NeuralNetwork
from os.path import join
from os.path import split
from keras.utils import np_utils
from ioutils import IOUtils

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
    classes_dict = {'GALAXY':0, 'STAR':1, 'QSO':2}
    matrix   = ioutils.read_input(args.input)
    classes  = matrix[1:, -1]
    nclasses = len(classes_dict.keys())
    X = np.array(matrix[1:, :-1], dtype=float)
    examples, attributes = X.shape

    #TODO: One-Hot encoding

if __name__ == '__main__':
    args = parse_args()
    main(args)