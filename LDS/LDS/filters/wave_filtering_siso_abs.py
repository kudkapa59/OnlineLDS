"""Abstract class for creation of persistent and spectral filters."""

import numpy as np
from abc import abstractmethod
from LDS.filters.filtering_siso import FilteringSiso
from LDS.h_m.hankel import Hankel


class WaveFilteringSisoAbs(FilteringSiso):
    """
    Abstract class for creation of persistent and spectral filters.

    Hierarchy tree ((ABC)):

                                                        WaveFilteringSisoPersistent
                                                            ^
                                                            |
    Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC) -->WaveFilteringSisoFtlPersistent
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """

    def __init__(self, sys, t_t, k):
        """
        Inherits FilteringSiso method. Adds k

        Args:
            sys: linear dynamical system. DynamicalSystem object.
            t_t: time horizon.
            k: 
        """
        super().__init__(sys, t_t)
        self.k = k

    def var_calc(self):
        """
        n - input vector.
        m - observation vector.
        k_dash - 
        H - Hankel matrix.
        M - 

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
        Abstract method for calculating output predictions and errors.

        Returns:
            y_pred_full: output prediction.
            M: identity matrix ????
            pred_error: spectral filter prediction error.
            pred_error_persistent: persistent filter error.
        """
        y_pred_full = []
        pred_error = []
        pred_error_persistent = []
        M = self.M

        return y_pred_full, M, pred_error, pred_error_persistent
