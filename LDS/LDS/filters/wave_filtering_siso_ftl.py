"""Implements spectral filtering class with follow-the-leader algorithm.
Originates from function wave_filtering_SISO_ftl from onlinelds.py."""

import numpy as np
import scipy.optimize as opt
import LDS.online_lds.print_verbose as print_verb
import LDS.online_lds.cost_ftl as cost_ftl
import LDS.online_lds.gradient_ftl as gradient_ftl
from LDS.filters.wave_filtering_siso_abs import WaveFilteringSisoAbs

class WaveFilteringSisoFtl(WaveFilteringSisoAbs):
    """
    Spectral filter with follow-the-leader algorithm.
    Hierarchy tree ((ABC)):

                                                        WaveFilteringSisoPersistent
                                                            ^
                                                            |
    Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC) -->WaveFilteringSisoFtlPersistent
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """
    def __init__(self, sys, t_t, k, verbose):
        """
        Inherits all the attributes of its superclass(see WaveFilteringSisoAbs).
        With initialization goes through all the methods and gets the predictions.
        
        Args:
            sys: linear dynamical system. DynamicalSystem object.
            t_t: time horizon.
            k: 
            verbose: Will be replaced.

        Variables initialized with var_calc():
        n - input vector.
        m - observation vector.
        k_dash - 
        H - Hankel matrix.
        M - 

        Uses method args4ftl_calc to create an array with m and k_dash.
        """
        super().__init__(sys, t_t, k)
        super().var_calc()
        self.args4ftl_calc()
        self.verbose = verbose
        self.y_pred_full, self.M,self.pred_error = self.predict()


    def args4ftl_calc(self):
        """
        Parameters calculation.Creates a 5-element array with m 
        on the zero position and k_dash on the first position. 
        All others are zeros.
        self.m - observation vector.
        self.k_dash - 
        """
        self.args4ftl = [0 for i in range(5)]
        self.args4ftl[0] = self.m
        self.args4ftl[1] = self.k_dash

    # def predict(self):
    #     """
    #     Returns:
    #         y_pred_full: y prediction
    #         M: identity matrix
    #         pred_error: spectral error prediction error

    #     """
    #     t_t = self.t_t
    #     k = self.k
    #     m = self.m
    #     H = self.H
    #     sys = self.sys
    #     M = self.M
    #     args4ftl = self.args4ftl
    #     k_dash = self.k_dash

    #     y_pred_full = []
    #     pred_error = []

    #     scalings = [pow(H.V[j], 0.25) for j in range(k)]
    #     for t in range(1, t_t):
    #         print_verb.print_verbose("step %d of %d" % (t + 1, t_t),self.verbose)
    #         X = []
    #         for j in range(k):
    #             scaling = scalings[j]
    #             conv = 0
    #             for u in range(t + 1):
    #                 conv += H.matrix_d[u, j] * sys.inputs[t - u]
    #             X.append(scaling * conv)

    #         X.append(sys.inputs[t - 1])
    #         X.append(sys.inputs[t])
    #         X.append(sys.outputs[t - 1])

    #         X = np.matrix(X).reshape(-1, 1)

    #         y_pred = np.real(M * X)
    #         y_pred = y_pred[0, 0]
    #         y_pred_full.append(y_pred)
    #         loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)

    #         args4ftl[2] = t

    #         try:
    #             args4ftl[3] = np.concatenate((args4ftl[3], sys.outputs[t]), 1)
    #             args4ftl[4] = np.concatenate((args4ftl[4], X), 1)
    #         except:
    #             args4ftl[3] = sys.outputs[t]
    #             args4ftl[4] = X

    #         args4ftl_tuple = tuple(i for i in args4ftl)

    #         # result = opt.minimize(cost_ftl.cost_ftl, M.reshape(-1,1),\
    #         # args=args4ftl_tuple, method='CG', jac=gradient_ftl.gradient_ftl)
    #         result = opt.minimize(cost_ftl.cost_ftl, M.reshape(-1, 1), args=args4ftl_tuple, jac=gradient_ftl.gradient_ftl)

    #         M = np.matrix(result.x).reshape(m, k_dash)
    #         pred_error.append(loss)
    #     return y_pred_full, M, pred_error

    '''Gian-Reto Wiher version'''
    def predict(self):
        """
        Returns:
            y_pred_full: y prediction
            M: identity matrix
            pred_error: spectral error prediction error

        """
        M = self.M
        args4ftl = self.args4ftl

        y_pred_full = []
        pred_error = []

        scalings = [pow(self.H.V[j], 0.25) for j in range(self.k)]
        for t in range(1, self.t_t):
            print_verb.print_verbose("step %d of %d" % (t + 1, self.t_t), self.verbose)
            X = np.zeros((self.m, self.k_dash))
            for j in range(self.k):
                scaling = scalings[j]
                conv = 0
                for u in range(t + 1):
                    conv += self.H.matrix_d[u, j] * self.sys.inputs[t - u]
                X[:, j] = scaling.real * conv.real

            X[:, -3] = self.sys.inputs[t - 1]
            X[:, -2] = self.sys.inputs[t]
            X[:, -1] = self.sys.outputs[t - 1]

            X = np.matrix(X).reshape(-1, 1)
            #After this row everything is the same as in the original code.

            y_pred = np.real(M * X)
            y_pred = y_pred[0, 0]
            y_pred_full.append(y_pred)
            loss = pow(np.linalg.norm(self.sys.outputs[t] - y_pred), 2)
            #print(self.sys.outputs[t],y_pred)

            args4ftl[2] = t

            try:
                args4ftl[3] = np.concatenate((args4ftl[3], self.sys.outputs[t]), 1)
                args4ftl[4] = np.concatenate((args4ftl[4], X), 1)
            except:
                args4ftl[3] = self.sys.outputs[t]
                args4ftl[4] = X

            args4ftl_tuple = tuple(i for i in args4ftl)

            # result = opt.minimize(cost_ftl.cost_ftl, M.reshape(-1,1),\
            # args=args4ftl_tuple, method='CG', jac=gradient_ftl.gradient_ftl)
            result = opt.minimize(cost_ftl.cost_ftl, M.reshape(-1, 1), args=args4ftl_tuple,\
                jac=gradient_ftl.gradient_ftl)

            M = np.matrix(result.x).reshape(self.m, self.k_dash)
            pred_error.append(loss)

            #print(self.sys.outputs[t],self.sys.outputs[t - 1])
        #print(y_pred_full[0],self.sys.outputs[0])
        #print(y_pred_full[1],self.sys.outputs[1])
        #print(y_pred_full[2],self.sys.outputs[2])
        return y_pred_full, M, pred_error