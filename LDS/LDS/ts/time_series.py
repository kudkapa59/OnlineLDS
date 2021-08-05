"""Implements time series from inputlds.py"""

from __future__ import print_function
#import rlcompleter
import traceback
import math
import tables # Matlab loading
from scipy.io import loadmat


class TimeSeries(object):
    """
    Class originated from inputlds.py, which
    was the first version of the algorithm.
    """
    def __init__(self, matlabfile, varname):
        """
        Inits TimeSeries.

        Args:
            matlabfile: the matlab file './OARIMA_code_data/data/setting6.mat'
            varname: uses 'seq_d0'. 1 x 35701 double vector.

        Raises:
            HDF5ExtError: can't open given matlabfile.
        """
        f = None
        self.outputs = []
        try:
            f = tables.open_file(filename = matlabfile, mode='r')
            self.outputs = f.getNode('/' + varname)[:]
        except tables.exceptions.HDF5ExtError:
            print("Error in loading Matlab .dat from 7 upwards ... ")
            # print(traceback.format_exc())
        try:
            if not f: #In case we got error in previous try
                print("Loading Matlab .dat prior to version 7 instead.")
                print(loadmat(matlabfile).keys()) #Prints the variables of matlabfile
                self.outputs = list(loadmat(matlabfile)[varname][0]) #values of seq_d0
        except:
            print("Error in loading Matlab .dat prior to version 7: ")
            print(traceback.format_exc())
        self.m = 1   #Observation vector
        #Loaded 35701 elements of seq_d0
        print("Loaded %i elements in a series %s." % (len(self.outputs), varname))
        self.event_or_not = [False] * len(self.outputs)
        self.inputs = [0.0] * len(self.outputs)
        self.h_zero = 0
        self.n = 1   #Input vector

    def solve(self, h_zero = [], inputs = [], t_t = 100, **kwargs):
        """
        This just truncates the series loaded in the constructor.

        Raises:
            Exits if time horizon isn't an integer.
        """

        if not isinstance(t_t,int):
            print("t_t must be an integer. Anything less than 1 suggest no truncation is needed.")
            exit()
        print("Truncating to %i elements ..." % (t_t))
        self.h_zero=h_zero
        self.m = 1
        if t_t > 0:
            self.event_or_not = self.event_or_not[:t_t]
            self.inputs = self.inputs[:t_t]
            self.outputs = self.outputs[:t_t]

    def logratio(self):
        """
        Replaces the time series by a log-ratio of subsequent element therein.
        """
        t_t = len(self.outputs)
        newOutputs = []
        for (a, b) in zip(self.outputs[:t_t-1], self.outputs[1:t_t]):
            newOutputs.append( math.log(a / b) )
        self.outputs = newOutputs
