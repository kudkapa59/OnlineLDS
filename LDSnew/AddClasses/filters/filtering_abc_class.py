"""This script implements ABC class."""
from abc import ABC, abstractmethod

class Filtering(ABC):
    """
    Abstract class for abstraction of filters.
    Superclass of class FilteringSiso.
    Implements the ancestor to KalmanFilteringSISO, WaveFilteringSISO and
    WaveFilteringSisoFtl classes, which have a real use.
    
    Hierarchy tree ((ABC)):

        Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC)
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """

    @abstractmethod
    def __init__(self, sys, t_t):
        """
        Abstract method.
        Inits Filtering with args sys and t_t, which
        are used as attributes.

        Args:
            sys: instance of DynamicalSystem class
            t_t: integer
        """
        self.sys = sys
        self.t_t = t_t
