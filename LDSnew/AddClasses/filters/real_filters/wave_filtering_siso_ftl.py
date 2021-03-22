"""Implements WaveFilteringSisoFtl.
Originates from function wave_filtering_SISO_ftl from onlinelds.py."""

import numpy as np
import scipy.optimize as opt
import AddClasses.online_lds.print_verbose as print_verb
import AddClasses.online_lds.cost_ftl as cost_ftl
import AddClasses.online_lds.gradient_ftl as gradient_ftl
from AddClasses.filters.wave_filtering_siso_abs import WaveFilteringSisoAbs

class WaveFilteringSisoFtl(WaveFilteringSisoAbs):
    """
    Subclass of class WaveFilteringSisoAbs.
    This one is not abstract, as we really use it.
    Hierarchy tree ((ABC)):

        Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC)
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """
    def __init__(self, sys, t_t, k, verbose):
        """
        Inherits all the attributes of its superclass(see WaveFilteringSisoAbs).
        Uses method args4ftl_calc to create an array with m and k_dash.
        """
        super().__init__(sys, t_t, k)
        super().var_calc()
        self.args4ftl_calc()
        self.verbose = verbose
        self.y_pred_full, self.M,\
             self.pred_error, self.pred_error_persistent = self.predict()


    def args4ftl_calc(self):
        """
        Creates a 5-element array with m on the zero position
        and k_dash on the first position. All others are zeros.
        """
        self.args4ftl = [0 for i in range(5)]
        self.args4ftl[0] = self.m
        self.args4ftl[1] = self.k_dash
        #print(self.m,self.k_dash)

    def predict(self):
        """
        Returns:
            y_pred_full:
            M:
            pred_error:
            pred_error_persistent:

        """
        t_t = self.t_t
        k = self.k
        m = self.m
        H = self.H
        sys = self.sys
        M = self.M
        args4ftl = self.args4ftl
        k_dash = self.k_dash

        y_pred_full = []
        pred_error = []
        pred_error_persistent = []

        scalings = [pow(H.V[j], 0.25) for j in range(k)]
        for t in range(1, t_t):
            print_verb.print_verbose("step %d of %d" % (t + 1, t_t),self.verbose)
            X = []
            for j in range(k):
                scaling = scalings[j]
                conv = 0
                for u in range(t + 1):
                    conv += H.matrix_d[u, j] * sys.inputs[t - u]
                X.append(scaling * conv)

            X.append(sys.inputs[t - 1])
            X.append(sys.inputs[t])
            X.append(sys.outputs[t - 1])

            X = np.matrix(X).reshape(-1, 1)

            y_pred = np.real(M * X)
            y_pred = y_pred[0, 0]
            y_pred_full.append(y_pred)
            loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)

            args4ftl[2] = t

            try:
                args4ftl[3] = np.concatenate((args4ftl[3], sys.outputs[t]), 1)
                args4ftl[4] = np.concatenate((args4ftl[4], X), 1)
            except:
                args4ftl[3] = sys.outputs[t]
                args4ftl[4] = X

            args4ftl_tuple = tuple(i for i in args4ftl)

            # result = opt.minimize(cost_ftl.cost_ftl, M.reshape(-1,1),\
            # args=args4ftl_tuple, method='CG', jac=gradient_ftl.gradient_ftl)
            result = opt.minimize(cost_ftl.cost_ftl, M.reshape(-1, 1), args=args4ftl_tuple, jac=gradient_ftl.gradient_ftl)

            M = np.matrix(result.x).reshape(m, k_dash)
            pred_error.append(loss)
            pred_error_persistent.append(pow(np.linalg.norm(sys.outputs[t] - sys.outputs[t - 1]),\
                 2))
        return y_pred_full, M, pred_error, pred_error_persistent
