"""In FilteringSiso we separated KalmanFilteringSISO. By doing this we dedicated 
the written below class to be the abstract class for spectral and persistent filters.
"""

import numpy as np
from abc import abstractmethod
from LDS.filters.filtering_siso import FilteringSiso
from LDS.h_m.hankel import Hankel


class WaveFilteringSisoAbs(FilteringSiso):
    """
    Abstract class for creation of persistent and spectral filters.
    The subclass WaveFilteringSISO is spectral filter only for symmetric transition matrix.
    The related work is
    "Learning Linear Dynamical Systems via Spectral Filtering" by E.Hazan, K.Singh and C.Zhang.
    WaveFilteringSisoFtl is the class for general case prediction.
    The related work is 
    "Spectral Filtering for General Linear Dynamical Systems" by E.Hazan, K.Singh, H.Lee 
                                                                            and C.Zhang. 
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
        Inherits FilteringSiso method.

        Args:
            sys : LDS. DynamicalSystem object.
            t_t : Time horizon.
            k   : Number of wave-filters for a spectral filter.
        """
        super().__init__(sys, t_t)
        self.k = k

    def var_calc(self):
        """
        Initializes spectral filter's parameters:
            n      : Input vector. Shape of processing noise.
            m      : Observation vector. Shape of observational error.
            k_dash : Siso filter parameter.
            H      : Hankel matrix.
            M      : Matrix specifying a linear map from featurized inputs to predictions. 
                     Siso filter parameter. 
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
            y_pred_full           : Output prediction.
            M                     : Matrix specifying a linear map from featurized inputs to predictions. 
                                    Siso filter parameter. 
            pred_error            : Spectral filter prediction error.
            pred_error_persistent : Persistent filter error.
        """
        y_pred_full = []
        pred_error = []
        pred_error_persistent = []
        M = self.M

        return y_pred_full, M, pred_error, pred_error_persistent
