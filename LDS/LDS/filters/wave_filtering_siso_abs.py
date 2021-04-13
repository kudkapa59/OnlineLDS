"""Implements an ancestor of real classes WaveFilteringSISO and 
WaveFilteringSisoFtl."""

import numpy as np
from abc import abstractmethod
from LDS.filters.filtering_siso import FilteringSiso
from LDS.h_m.hankel import Hankel


class WaveFilteringSisoAbs(FilteringSiso):
    """
    Abstract class.
    Subclass of abstract class FilteringSiso.
    Superclass of classes WaveFilteringSISO and WaveFilteringSisoFtl.

    Hierarchy tree ((ABC)):

        Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC)
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """

    def __init__(self, sys, t_t, k):
        """
        Inherits init method of FilteringSiso.
        Inits WaveFilteringSisoAbs with args sys, t_t, k, which
        are used as attributes.

        Args:
            sys: instance of DynamicalSystem class
            t_t: integer. Will write what it is.
            k: integer
        """
        super().__init__(sys, t_t)
        self.k = k

    def var_calc(self):
        """
        self.n - input vector
        self.m - observation vector
        self.k_dash - 
        self.H - Hankel matrix
        self.M - 

        Calculating all parameters of the filter.
        """
        self.n = self.sys.n
        self.m = self.sys.m

        self.k_dash = self.n * self.k + 2 * self.n + self.m

        self.H = Hankel(self.t_t)
        self.M = np.matrix(np.eye(self.m, self.k_dash))

    @abstractmethod
    def predict(self):
        """
        Abstract method.

        Returns:
            y_pred_full: y prediction
            M: identity matrix
            pred_error: prediction error
            pred_error_persistent: error of the prediction method
        """
        y_pred_full = []
        pred_error = []
        pred_error_persistent = []
        M = self.M

        return y_pred_full, M, pred_error, pred_error_persistent
