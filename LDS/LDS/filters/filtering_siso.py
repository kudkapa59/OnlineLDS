"""This script implements ABC class."""

import os, sys
from abc import abstractmethod
from LDS.filters.filtering_abc_class import Filtering

class FilteringSiso(Filtering):
    """
    Abstract class.
    Subclass of abstract class Filtering.
    Superclass of classes KalmanFilteringSISO and WaveFilteringSisoAbs.
    Implements the ancestor to KalmanFilteringSISO, WaveFilteringSISO and
    WaveFilteringSisoFtl classes, which have a real use.

    Hierarchy tree ((ABC)):

        Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC)
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl

    """

    def __init__(self, sys, t_t):
        """
        Inherits abstract init method of Filtering.
        Inits FilteringSiso with sys and t_t, which
        are used as attributes.

        Args:
            sys: instance of DynamicalSystem class
            t_t: integer
        """
        super().__init__(sys, t_t)

    @abstractmethod
    def predict(self):
        """
        Abstract method.
        Creates two empty arrays.

        Returns:
            y_pred_full: prediction of output
            pred_error:  error prediction
        """
        y_pred_full, pred_error = [], []
        return y_pred_full, pred_error
