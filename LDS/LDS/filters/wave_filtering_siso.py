"""
Originates from function wave_filtering_SISO from onlineLDS.py.
The related work is 
"Spectral Filtering for General Linear Dynamical Systems" by E.Hazan, K.Singh, H.Lee 
                                                                            and C.Zhang. 
"""

import numpy as np
from LDS.filters.wave_filtering_siso_abs import WaveFilteringSisoAbs

class WaveFilteringSISO(WaveFilteringSisoAbs):
    """
    Implements spectral filter.

    Hierarchy tree ((ABC)):                                 
                                                        WaveFilteringSISOPersistent
                                                            ^
                                                            |
    Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC) -->WaveFilteringSisoFtlPersistent
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """

    def __init__(self, sys, t_t, k, eta, r_m):
        """
        Inits all the attributes of its superclass(see WaveFilteringSisoAbs) and
        adds eta and r_m. Goes through all the methods and gets the predictions.

        Args:
            sys: linear dynamical system. DynamicalSystem object.
            t_t: time horizon.
            k: 
            eta: 
            r_m: 
        """
        super().__init__(sys, t_t, k)
        self.eta = eta
        self.r_m = r_m

        super().var_calc()
        self.y_pred_full, self.M, self.pred_error = self.predict()

    def predict(self):
        """
        Calculation of output predictions and prediction errors.

        Returns:
            y_pred_full: signal prediction values.
            M: identity matrix. ???
            pred_error: spectral filter error.
        """

        t_t = self.t_t
        k = self.k
        H = self.H
        sys = self.sys
        M = self.M
        eta = self.eta
        r_m = self.r_m

        y_pred_full = []
        pred_error = []

        for t in range(1, t_t):
            X = []
            for j in range(k):
                scaling = pow(H.V[j], 0.25)
                conv = 0
                for u in range(0, t):
                    conv += H.matrix_d[u, j] * sys.inputs[t - u]
                X.append(scaling * conv)

            X.append(sys.inputs[t - 1])
            X.append(sys.inputs[t])
            X.append(sys.outputs[t - 1])

            X = np.matrix(X).reshape(-1, 1)

            y_pred = np.real(M * X)
            y_pred = y_pred[0, 0]
            y_pred_full.append(y_pred)
            # loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)
            loss = pow(np.linalg.norm(sys.outputs[t] + y_pred), 2)
            M = M - 2 * eta * (sys.outputs[t] - y_pred) * X.transpose()
            frobenius_norm = np.linalg.norm(M, 'fro')
            if frobenius_norm >= r_m:
                M = r_m / frobenius_norm * M

            pred_error.append(loss)


            # print(loss)

        return y_pred_full, M, pred_error
