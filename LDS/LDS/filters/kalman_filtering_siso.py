"""Implements Kalman filter prediction.
Originates from function Kalman_filtering_SISO from onlinelds.py."""

import numpy as np
from LDS.filters.filtering_siso import FilteringSiso

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
            sys: instance of DynamicalSystem class.
            t_t: time horizon.
        """
        super().__init__(sys, t_t)

    def predict(self):
        """
        Creates G - square identity matrix 4x4
                n - number of rows of G
                F - matrix 4x1 of 0.5
                Id - identity matrix 4x4
                m_prev - 0
                c_prev - matrix nxn of zeros 
                
                Matrices from the paper:
                a = G.m_prev
                R
                F
                RF
                Q
                matrix_a

        Returns:
            y_pred_full:
            pred_error:

        Raises:
            Q can't be zero.
        """

        sys = self.sys
        t_t = self.t_t                       #time horizon

        G = np.diag(np.array(np.ones(4)))    #$G \in \RR^{n\times n}$ is the state transition 
                                             #matrix which defines the system dynamics
        n = G.shape[0]

        F = np.ones(n)[:, np.newaxis] / np.sqrt(n) #$F \in \RR^{n\times1}$ 
                                                   #is the observation direction
        Id = np.eye(n)
        m_prev = 0                           #m_{t-1} is the last hidden state
        c_prev = np.zeros((n, n))            #C_{t-1} is the covariance matrix of $\phi_{t-1}$ given 
                                             #$Y_0,\ldots,Y_{t-1}$. 

        y_pred_full = [0]
        pred_error = [sys.outputs[0]]

        for t in range(1, t_t):

            a = np.dot(G, m_prev)   #LaTeX a_t &=& G m_{t-1}
            R = np.dot(G, np.dot(c_prev, G.t_t))  # + W    #LaTeX R_t &=& G C_{t-1} G' + W

            f = np.dot(F.t_t, a)            #f_t = F' a_t. In particular, in this paper we refer
                                            #to the sequence $f_{t}$ as the 
                                            #Kalman filter associated with the LDS $L=(G,F,v,W)$. 
            RF = np.dot(R, F)               #R_tF
            Q = np.dot(F.t_t, RF)  # + V    #LaTeX Q_t &=& F'R_tF + v

            matrix_a = RF                   
            try:
                matrix_a = RF / Q           #LaTeX A_t &=& R_t F  / Q_t
            except:
                print("Zero Q? Check %s" % str(Q))

            # thats on purpose in a bit slower form, to test the equations
            y_pred = np.dot(F.t_t, np.dot(G, m_prev)) #Same as Kalman filter 
                                                      #f = np.dot(F.t_t, a)

            #m_t = A_t Y_t + (I - F \otimes A_t) a_t
            m_prev = y_pred * matrix_a + np.dot((Id - np.dot(matrix_a, F.t_t)), a) 
            #Why do we use here y_pred instead of sys.outputs[t]? We don't have them?

            #C_t &=& R_t - A_t Q_t A'_t
            c_prev = R - Q * np.dot(matrix_a, matrix_a.t_t)   
            #Why is Q not in the middle of multiplication

            y_pred_full.append(y_pred)

            # $The loss function (Y_t-y_pred)^2$.
            loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2) 

            pred_error.append(loss)
        #print(y_pred_full)
        return y_pred_full, pred_error
