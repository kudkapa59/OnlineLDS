"""Implements Kalman filter prediction.
Originates from function Kalman_filtering_SISO from onlinelds.py."""

import numpy as np
from AddClasses.filters.filtering_siso import FilteringSiso

class KalmanFilteringSISO(FilteringSiso):
    """
    Subclass of class FilteringSiso.
    This one is not abstract, as we really use it.

    Hierarchy tree ((ABC)):

        Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC)
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """

    def __init__(self, sys, t_t):
        """
        Inherits init method of FilteringSiso.
        Inits KalmanFilteringSISO with args sys, t_t, which
        are used as attributes.

        Args:
            sys: instance of DynamicalSystem class
            t_t: integer
        """
        super().__init__(sys, t_t)

    def predict(self):
        """
        Returns:
            y_pred_full:
            pred_error:
        """

        sys = self.sys
        t_t = self.t_t

        G = np.diag(np.array(np.ones(4)))
        n = G.shape[0]

        F = np.ones(n)[:, np.newaxis] / np.sqrt(n)
        Id = np.eye(n)
        m_prev = 0
        c_prev = np.zeros((n, n))

        y_pred_full = [0]
        pred_error = [sys.outputs[0]]

        for t in range(1, t_t):
            a = np.dot(G, m_prev)
            R = np.dot(G, np.dot(c_prev, G.t_t))  # + W

            f = np.dot(F.t_t, a)
            RF = np.dot(R, F)
            Q = np.dot(F.t_t, RF)  # + V
            matrix_a = RF
            try:
                matrix_a = RF / Q
            except:
                print("Zero Q? Check %s" % str(Q))

            # thats on purpose in a bit slower form, to test the equations
            y_pred = np.dot(F.t_t, np.dot(G, m_prev))
            m_prev = y_pred * matrix_a + np.dot((Id - np.dot(matrix_a, F.t_t)), a)
            c_prev = R - Q * np.dot(matrix_a, matrix_a.t_t)

            y_pred_full.append(y_pred)
            loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)
            pred_error.append(loss)
        print(y_pred_full)
        return y_pred_full, pred_error
