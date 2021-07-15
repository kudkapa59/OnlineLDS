"""This script implements ABC class."""
from abc import ABC, abstractmethod

class Filtering(ABC):
    """
    Abstract class for creation of filters.
    
    Hierarchy tree ((ABC)):

                                                        WaveFilteringSisoPersistent
                                                            ^
                                                            |
    Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC) -->WaveFilteringSisoFtlPersistent
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """

    @abstractmethod
    def __init__(self, sys, t_t):
        """
        Initializing a basic filter.

        Args:
            sys : LDS. DynamicalSystem object.
            t_t : Time horizon.
        """
        self.sys = sys
        self.t_t = t_t
