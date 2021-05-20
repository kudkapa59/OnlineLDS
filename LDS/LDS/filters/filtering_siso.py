"""This script implements ABC class."""

import os, sys
from abc import abstractmethod
from LDS.filters.filtering_abc_class import Filtering

class FilteringSiso(Filtering):
    """
    Abstract class.
    Specifically written to separate Kalman filter and AR from spectral and persistent filters.

    Hierarchy tree ((ABC)):

                                                        WaveFilteringSisoPersistent
                                                            ^
                                                            |
    Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC) -->WaveFilteringSisoFtlPersistent
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl

    """

    def __init__(self, sys, t_t):
        """
        Inherits abstract init method of Filtering.
        Inits FilteringSiso with sys and t_t, which
        are used as attributes.

        Args:
            sys: linear dynamical system. DynamicalSystem object.
            t_t: time horizon.
        """

        super().__init__(sys, t_t)

    @abstractmethod
    def predict(self):
        """
        Abstract method for output prediction and prediction error.

        Returns:
            y_pred_full: output prediction.
            pred_error:  prediction error.
        """

        y_pred_full, pred_error = [], []
        return y_pred_full, pred_error
