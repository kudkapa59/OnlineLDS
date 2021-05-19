"""Implements Kalman filter prediction.
Originates from function Kalman_filtering_SISO from onlinelds.py."""

import numpy as np
from LDS.filters.filtering_siso import FilteringSiso

class KalmanFilteringSISO(FilteringSiso):
    """
    Calculates Kalman filter parameters. Finds the prediction for Kalman and AR.
    Subclass of class FilteringSiso.

                                                        WaveFilteringSisoPersistent
                                                            ^
                                                            |
    Filtering(ABC) --> FilteringSiso(ABC) -->  WaveFilteringSisoAbs(ABC) -->WaveFilteringSisoFtlPersistent
                                     |                 |                |
                    KalmanFilteringSISO    WaveFilteringSISO  WaveFilteringSisoFtl
    """

    def __init__(self, sys, G, f_dash,proc_noise_std, obs_noise_std, t_t, Y):
        """
        Inherits init method of FilteringSiso. Calls parameters method.

        Args:
            sys: linear dynamical system.
            G: state transition matrix.
            f_dash: observation direction.
            proc_noise_std: standard deviation of processing noise.
            obs_noise_std: standard deviation of observation noise.
            t_t: time horizon.
            Y: scalar observations.
        """
        super().__init__(sys, t_t)
        self.G = G
        self.f_dash = f_dash
        self.proc_noise_std = proc_noise_std
        self.obs_noise_std = obs_noise_std
        self.Y = Y
        #a, b = self.predict()
        self.parameters() #?

    def parameters(self):
        """
        Finds Kalman filter's parameters:
            W: processing noise covariance.
            V: observation noise covariance.

        Raises:  #Not raises yet
            Q can't be zero.
        """

        self.n = self.G.shape[0]   #input vector
        self.m = self.f_dash.shape[0] #observation vector

        self.W = self.proc_noise_std ** 2 * np.matrix(np.eye(self.n))
        self.V = self.obs_noise_std ** 2 * np.matrix(np.eye(self.m))

        # m_t = [np.matrix([[0],[0]])]
        self.matrix_c = [np.matrix(np.eye(2))]
        self.R = []
        self.Q = []
        self.matrix_a = []
        self.Z = []

        for t in range(self.t_t):
            self.R.append(self.G * self.matrix_c[-1] * self.G.transpose() + self.W)
            # if t == 1:
            #     print('muj')
            #     print(self.R)
            #     print('Kalman')
            self.Q.append(self.f_dash * self.R[-1] * self.f_dash.transpose() + self.V)

            self.matrix_a.append(self.R[-1] * self.f_dash.transpose() * np.linalg.inv(self.Q[-1]))

            #C_t &=& R_t - A_t Q_t A'_t
            self.matrix_c.append(self.R[-1] - self.matrix_a[-1] * self.Q[-1] *\
               self.matrix_a[-1].transpose())

            #In general, set $Z_t = G(I-F\otimes A_t)$ and $Z = G(I-F \otimes A)$.
            self.Z.append(self.G * (np.eye(2) - self.matrix_a[-1] * self.f_dash))

        #return n, m, W, V, matrix_c, R, Q, matrix_a, Z

        #Y_pred = prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y)
        #Y_kalman = prediction_kalman(t_t, f_dash, G, matrix_a, sys, Z, Y)

    def predict(self,s,error_AR1_data,error_kalman_data_new):

        y_pred_full = []
        for t in range(self.t_t):
            Y_pred_term1 = self.f_dash * self.G * self.matrix_a[t] * self.sys.outputs[t]
            if t == 0:
                y_pred_full.append(Y_pred_term1)
                continue

            self.accKalman = 0
            #We don't have range(min(t,s)+1) as we do for prediction function
            for j in range(min(t,s) + 1):
                for i in range(j + 1):
                    if i == 0:
                        ZZ = self.Z[t - i]
                        continue
                    ZZ = ZZ * self.Z[t - i]
                self.accKalman += ZZ * self.G * self.matrix_a[t - j - 1] * self.Y[t - j - 1]
            y_pred_full.append(Y_pred_term1 + self.f_dash * self.accKalman)

        if s == 1:  ###?
            if error_AR1_data is None:
                error_AR1_data = np.array([pow(np.linalg.norm(y_pred_full[i][0,0] - self.Y[i]),\
                        2) for i in range(len(self.Y))])   #quadratic loss
            else:
                #print(error_AR1_data.shape)
                error_AR1_data = np.vstack((error_AR1_data,\
                        [pow(np.linalg.norm(y_pred_full[i][0,0] - self.Y[i]), 2) for i\
                            in range(len(self.Y))]))
        
        if s == self.t_t: ###?
            #For the spectral filtering etc, we use:
            #loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)

            #I want to replace this chunk by kalman_filtering_siso.py
            if error_kalman_data_new is None:
                error_kalman_data_new = np.array([pow(np.linalg.norm(y_pred_full[i][0,0] - \
                    self.Y[i]), 2) for i in range(len(self.Y))])
            else:
                error_kalman_data_new = np.vstack((error_kalman_data_new,\
                        [pow(np.linalg.norm(y_pred_full[i][0,0] - self.Y[i]), 2) for i\
                            in range(len(self.Y))]))

        return y_pred_full, error_AR1_data, error_kalman_data_new



    def predict_kalman(self,s,error_AR1_data,error_kalman_data_new):

        y_pred_kalman = []
        for t in range(self.t_t):
            Y_pred_term1 = self.f_dash * self.G * self.matrix_a[t] * self.sys.outputs[t]
            if t == 0:
                y_pred_kalman.append(Y_pred_term1)
                continue

            self.accKalman = 0
            #We don't have range(min(t,s)+1) as we do for prediction function
            for j in range(t + 1):
                for i in range(j + 1):
                    if i == 0:
                        ZZ = self.Z[t - i]
                        continue
                    ZZ = ZZ * self.Z[t - i]
                self.accKalman += ZZ * self.G * self.matrix_a[t - j - 1] * self.Y[t - j - 1]
            y_pred_kalman.append(Y_pred_term1 + self.f_dash * self.accKalman)

        if s == 1: ###?
            if error_AR1_data is None:
                error_AR1_data = np.array([pow(np.linalg.norm(y_pred_kalman[i][0,0] - self.Y[i]),\
                        2) for i in range(len(self.Y))])   #quadratic loss
            else:
                #print(error_AR1_data.shape)
                error_AR1_data = np.vstack((error_AR1_data,\
                        [pow(np.linalg.norm(y_pred_kalman[i][0,0] - self.Y[i]), 2) for i\
                            in range(len(self.Y))]))
        
        if s == self.t_t:###?
            #For the spectral filtering etc, we use:
            #loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)

            #I want to replace this chunk by kalman_filtering_siso.py
            if error_kalman_data_new is None:
                error_kalman_data_new = np.array([pow(np.linalg.norm(y_pred_kalman[i][0,0] - \
                    self.Y[i]), 2) for i in range(len(self.Y))])
            else:
                error_kalman_data_new = np.vstack((error_kalman_data_new,\
                        [pow(np.linalg.norm(y_pred_kalman[i][0,0] - self.Y[i]), 2) for i\
                            in range(len(self.Y))]))

        return y_pred_kalman, error_AR1_data, error_kalman_data_new
        

        # sys = self.sys                       #LDS
        # t_t = self.t_t                       #time horizon

        # #G = np.diag(np.array(np.ones(4)))    #$G \in \RR^{n\times n}$ is the state transition 
        #                                      #matrix which defines the system dynamics
        
        # #G = np.matrix([[0.999,0],[0,0.5]])   #state transition matrix

        # #def pre_comp_filter_params(G, f_dash, proc_noise_std, obs_noise_std, t_t):
        # n = G.shape[0] #
        # m = f_dash.shape[0] #new
        # #No noise covariance matrix was written yet.
        # matrix_c = [np.matrix(np.eye(2))] #new
        # W = proc_noise_std ** 2 * np.matrix(np.eye(n))  #new, covariance matrix of process noise
        # V = obs_noise_std ** 2 * np.matrix(np.eye(m))   #new, observation noise covariance
        # #R = []
        # #Q = []
        # #matrix_a = []
        # #Z = []

        # F = np.ones(n)[:, np.newaxis] / np.sqrt(n)

        # #Take the example
        # f_dash = np.matrix([[1,1]])  #new                  #is the observation direction
        
        # Id = np.eye(n)
        # m_prev = 0                           #m_{t-1} is the last hidden state
        # c_prev = np.zeros((n, n))            #C_{t-1} is the covariance matrix of $\phi_{t-1}$ 
        #                                      #given $Y_0,\ldots,Y_{t-1}$. 

        # y_pred_full = [0]
        # pred_error = [sys.outputs[0]]
        
        # for t in range(t_t):                #Changed from range(1,t_t)

        #     a = np.dot(G, m_prev)
        #     #R.append(G * matrix_c[-1] * G.transpose() + W) #new

        #     #R = np.dot(G, np.dot(c_prev, G.t_t)) + W
        #     R = np.dot(G, np.dot(matrix_c[-1], G.transpose())) + W

        #     """Result from the main file. We need to make it the same
        #     [matrix([[1.248001, 0.      ],
        #     [0.      , 0.5     ]]), matrix([[ 0.71753214, -0.15600005],
        #     [-0.15600005,  0.34371873]])]"""
        #     if t == 0:
        #         print(R) 

        #     f = np.dot(f_dash, a)           #f_t = F' a_t.   
        #     RF = np.dot(R, F)               #R_tF
        #     Q = np.dot(f_dash, RF)  # + V   

        #     matrix_a = RF                   
        #     try:
        #         matrix_a = RF / Q           #LaTeX A_t &=& R_t F  / Q_t
        #     except:
        #         print("Zero Q? Check %s" % str(Q))

        #     # thats on purpose in a bit slower form, to test the equations
        #     y_pred = np.dot(F.t_t, np.dot(G, m_prev)) #Same as Kalman filter 
        #                                               #f = np.dot(F.t_t, a)

        #     #m_t = A_t Y_t + (I - F \otimes A_t) a_t
        #     m_prev = y_pred * matrix_a + np.dot((Id - np.dot(matrix_a, F.t_t)), a) 
        #     #Why do we use here y_pred instead of sys.outputs[t]? We don't have them?

        #     #C_t &=& R_t - A_t Q_t A'_t
        #     c_prev = R - Q * np.dot(matrix_a, matrix_a.t_t)   
        #     #Why is Q not in the middle of multiplication

        #     y_pred_full.append(y_pred)

        #     # $The loss function (Y_t-y_pred)^2$.
        #     loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2) 

        #     pred_error.append(loss)
        # #print(y_pred_full)
        # return y_pred_full, pred_error
