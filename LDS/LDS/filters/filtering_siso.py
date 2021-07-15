"""This script implements ABC class."""

import os, sys
from abc import abstractmethod
from LDS.filters.filtering_abc_class import Filtering

class FilteringSiso(Filtering):
    """
    Abstract class.
    Specifically written to separate Kalman filter and auto-regression from spectral and 
    persistent filters.

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
        Inherits init method of Filtering.

        Args:
            sys : LDS. DynamicalSystem object.
            t_t : Time horizon.
        """

        super().__init__(sys, t_t)

    @abstractmethod
    def predict(self):
        """
        Creates empty lists for prediction and error of filters.

        Returns:
            y_pred_full : Output prediction.
            pred_error  : Prediction error.
        """

        y_pred_full, pred_error = [], []
        return y_pred_full, pred_error
