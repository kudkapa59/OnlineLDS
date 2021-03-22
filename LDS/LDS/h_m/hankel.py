"""Implements Hankel matrix."""
import numpy as np

class Hankel(object):
    """
    Class originated from onlinelds.py, which
    was the first version of the algorithm.
    Creates Hankel matrix.
    """
    def __init__(self, t_t):
        """
        Inits Hankel class with t_t argument.
        Stores Hankel matrix, its eigenvalues
        and normalized eigenvectors.

        Args:
            t_t: integer, size of Hankel matrix
        """
        self.mat = np.matrix([[2. / (pow(i + j + 2,\
             3) - (i + j + 2)) for j in range(t_t)] for i in range(t_t)])
        [self.V, self.matrix_d] = np.linalg.eig(self.mat)
