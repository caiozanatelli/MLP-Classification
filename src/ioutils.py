import numpy as np
import sys

class Singleton(type):

    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance

class IOUtils(object):
    '''
    A generic class for input and output manipulation.
    '''
    __metaclass__ = Singleton
    __logfile = None

    def __init__(self, logpath=None):
        if not logpath is None:
            self.open_log(logpath)

    def read_input(self, filepath):
        '''
        Read a csv input that contains 17 features and 1 class

        Arguments:
            [str] -- path of the input file.

        Returns:
            [numpy] -- np matrix
        '''
        matrix = np.loadtxt(filepath, dtype=object, delimiter=',')
        index  = np.where(matrix == 'class')
        column = int(index[1])
        if column != len(matrix[0]) - 1:
            matrix[:, [column, -1]] = matrix[:, [-1, column]]
        return matrix
    
    def write_data(self, data):
        '''
        Save a string data to the log file

        Arguments:
            [str] -- the string to be stored
        '''
        if self.__logfile is None:
            print('Log file not opened.')
        else:
            self.__logfile.write(data + '\n')

    def open_log(self, filepath):
        '''
        Open the log file for storing partial results
        
        Arguments:
            [str] -- path of a file for logging purposes
        '''
        self.__logfile = open(filepath, 'w')
