#This library is compilation of python scripts (experiments.py, onlinelds.py, inputlds.py)
#and matlab scripts (example.m, arima_ogd.m, arima_ons.m)
# source for each function is given in comments (experiments.py, onlinelds.py, inputlds.py, etc)
###

#The papers, which is mentioned here is 
#Mark Kozdoba, Jakub Marecek,Tigran Tchrakian, and Shie Mannor
# On-Line Learning of Linear Dynamical Systems:Exponential Forgetting in Kalman Filters




from __future__ import print_function
import rlcompleter
import traceback
import pdb
#import time
#import timeit
#import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.metrics import f1_score
#import tables # Matlab loading
from scipy.io import loadmat
from LDS.ds.dynamical_system import DynamicalSystem
from LDS.ts.time_series import TimeSeries
from LDS.filters.wave_filtering_siso import WaveFilteringSISO
from LDS.filters.wave_filtering_siso_ftl import WaveFilteringSisoFtl
from LDS.filters.wave_filtering_siso_ftl_persistent import WaveFilteringSisoFtlPersistent
from LDS.filters.kalman_filtering_siso import KalmanFilteringSISO
from LDS.matlab_options.matlab_class_options import ClassOptions

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# debugging
pdb.Pdb.complete=rlcompleter.Completer(locals()).complete

#from def4lib import *

#VERBOSE = False
G_mat = np.random.rand(2,2)
f_dash_mat = np.random.rand(1,2)

def close_all_figs():
    """

    Closes all the figures. Originally the function comes from experiments.py file.

    """
    plt.close('all')

def test_identification(sys, filename_stub = "test", no_runs = 2,
                       t_t = 100, k = 5, eta_zeros = None, ymin = None, ymax = None,
                       sequence_label = None, have_spectral_persistent = True):
    """

    Implements here On-line Gradient Descent Algorithm 1 by the use of cost_ar and gradient_ar
    functions. 
    Data found is used by plot_p1,plot_p2, plot_p3 functions which create "seq0", "logration" pdfs.    
    Implements Example 8 from Experiments section of the original paper. 
    Plots Figures 7,8.
    Originally the function comes from experiments.py file.

    Args:
        sys                     : LDS.
        filename_stub           : Name of the output file.
        no_runs                 : Number of runs.
        t_t                     : Time horizon.
        k                       : Number of filters.
        eta_zeros               : Learning rate. 
        y_min                   : Minimal value of y-axis.
        y_max                   : Maximal value of y-axis.
        sequence_label          : 
        have_spectral_persistent: False if there's no need to plot spectral and persistent filters.
                                  Default value - True.

    Raises:
        Exits if k > t_t.

    """


    if k>t_t:
        print("Number of filters (k) must be less than or equal",\
            "to the number of time-steps (t_t).")
        exit()
    if not eta_zeros:
        eta_zeros = [1.0, 2500.0]
    print("eta_zeros:")
    print(eta_zeros)

    filename = './outputs/' + filename_stub+'.pdf'
    p_p = PdfPages(filename)

    error_ar_data = None        #auto-regression error
    error_spec_data = None      #spectral filter error
    error_persist_data = None   #last-value prediction error

    for i in range(no_runs):
        print("run %i" % i)
        inputs = np.zeros(t_t)
        sys.solve([[1],[0]],inputs,t_t)

        if have_spectral_persistent: #Checks if we need spectral and persistent filters
            #using class WaveFilteringSisoFtl instead function WaveFilteringSisoFtl
            wf_siso_ftl = WaveFilteringSisoFtl(sys, t_t, k)
            predicted_spectral, M, error_spec = \
                wf_siso_ftl.y_pred_full, wf_siso_ftl.M,\
                    wf_siso_ftl.pred_error #wf_siso_ftl.pred_error_persistent

            if error_spec_data is None:
                error_spec_data = error_spec
            else:
                error_spec_data = np.vstack((error_spec_data, error_spec))

            wf_siso_persistent = WaveFilteringSisoFtlPersistent(sys, t_t, k)
            #Here I replaced error_persist_data with error_persist
            predicted_persistent, M, error_persist = \
                wf_siso_persistent.y_pred_full, wf_siso_persistent.M,\
                    wf_siso_persistent.pred_error_persistent #wf_siso_ftl.pred_error_persistent
            if error_persist_data is None:
                error_persist_data = error_persist
            else:
                error_persist_data = np.vstack((error_persist_data, error_persist))


        for eta_zero in eta_zeros:
            error_ar = np.zeros(t_t)
            predicted_ar = np.zeros(t_t) #predicted outputs
            s=2   #AR(2)
            matrix_d=1.
            theta = [0 for i in range(s)]  #regression coefficients
            for t in range(s,t_t):
                eta = pow(float(t),-0.5) / eta_zero #learning rate
                Y = sys.outputs[t]    #output values
                loss = cost_ar(theta, Y, list(reversed(sys.outputs[t-s:t]))) #quadratic loss
                error_ar[t] = pow(loss, 0.5) #individual loss
                grad = gradient_ar(theta, Y, list(reversed(sys.outputs[t-s:t])))#gradient of loss
                #print("Loss: at time step %d :" % (t), loss)
                theta = [theta[i] -eta*grad[i] for i in range(len(theta))] #gradient step
                norm_theta = np.linalg.norm(theta)
                if norm_theta>matrix_d:
                    theta = [matrix_d*i/norm_theta for i in theta] #projection step
                predicted_ar[t] = np.dot(list(reversed(sys.outputs[t-s:t])),theta)

            if error_ar_data is None:
                error_ar_data = error_ar
            else: #appending error values
                error_ar_data = np.vstack((error_ar_data, error_ar))

            if not have_spectral_persistent: #If we don't plot spectal and persistent filters
                predicted_spectral = []
                error_spec, error_persist = [], []
            plot_p1(ymin, ymax, inputs, sequence_label, have_spectral_persistent,
                    predicted_spectral, predicted_ar,
                    sys, p_p)
            plot_p2(have_spectral_persistent, error_spec, error_persist, error_ar, lab, p_p)

    error_ar_mean = np.mean(error_ar_data, 0)
    error_ar_std = np.std(error_ar_data, 0)
    if have_spectral_persistent:
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = \
            error_stat(error_spec_data, error_persist_data)
    else:
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = [], [], [], []

    plot_p3(ymin, ymax, have_spectral_persistent, error_spec_mean, error_spec_std,
            error_persist_mean, error_persist_std,
            error_ar_mean, error_ar_std,
            t_t, p_p)


    p_p.close()
    print("See the output in " + filename)



def test_identification2(t_t = 100, no_runs = 10, s_choices = [15,3,1],
                        have_kalman = False, have_spectral_persistent = True,
                        G = G_mat, f_dash = f_dash_mat,sequence_label = ""):
                        #G = np.matrix([[0.999,0],[0,0.5]]),
                        #f_dash = np.matrix([[1,1]]) ):
    """

    Implements Example 7 from Experiments section of the paper.
    Creates './outputs/AR.pdf'.Finds all the filters' errors and
    uses function p3_for_test_identification2 for plotting them.
    Plots Figure 2,5 of the main paper. Originally the function comes from experiments.py file.

    Args:
        t_t                     : Time horizon.
        no_runs                 : Number of runs.
        s_choices               : 
        have_kalman             : False if there's no need to plot kalman filter.
                                  Default value - True.  
        have_spectral_persistent: False if there's no need to plot spectral and persistent filters.
                                  Default value - True.
        G                       : State matrix.
        f_dash                  : first derivative of the observation direction.
        sequence_label          : 

    Raises:
        Exits if number of runs is less than 2.

    """
    if have_kalman:
        s_choices = s_choices + [t_t]
    if len(sequence_label) > 0:
        sequence_label = " (" + sequence_label + ")"

    if no_runs < 2:
        print("Number of runs has to be larger than 1.")
        exit()

    filename = './outputs/AR.pdf'
    p_p = PdfPages(filename)

    ################# SYSTEM ###################
    proc_noise_std = 0.5             #w = 0.5
    obs_noise_std  = 0.5             #v = 0.5

    error_spec_data = None           #error of the spectral filtering
    error_persist_data = None        #error of persistance prediction
    error_AR1_data = None            #error of auto-regression
    error_Kalman_data = None         #error of Kalman filtering
    error_kalman_data_new = None 

    for runNo in range(no_runs):
        sys = DynamicalSystem(G,np.zeros((2,1)),f_dash,np.zeros((1,1)),
                               process_noise='gaussian',
                               observation_noise='gaussian',
                               process_noise_std=proc_noise_std,
                               observation_noise_std=obs_noise_std,
                               timevarying_multiplier_b = None)
        inputs = np.zeros(t_t)
        sys.solve([[1],[1]],inputs,t_t)
        Y = [i[0,0] for i in sys.outputs]    #real outputs
        #pdb.set_trace()
        ############################################

        #Replacing with Kalman class #Need to check if we don't have repetitive vars
        #if have_kalman
        kalman_siso = KalmanFilteringSISO(sys, G, f_dash,proc_noise_std, obs_noise_std, t_t,Y)
        #predicted_kalman, error_kalman = kalman_siso.y_pred_full, kalman_siso.pred_error
        


        ########## PRE-COMPUTE FILTER PARAMS ###################
        #n, m, W, V, matrix_c, R, Q, matrix_a, Z = pre_comp_filter_params(G, f_dash,\
        #    proc_noise_std, obs_noise_std, t_t)           #Kalman filtering results

        #PREDICTION
        plt.plot(Y, label='Output', color='#000000', linewidth=2, antialiased = True)

        for s in s_choices: #Chose and fix some $s\geq 1$. Then for any $t\geq s+1$, 
                            #the expectation \eqref{thatexpintext} has the form displayed in 
                            #Figure 1.
            #Prediction with no remainder term
            #Y_pred = prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y)
            Y_pred_new, error_AR1_data, error_kalman_data_new = kalman_siso.predict(s,\
                error_AR1_data, error_kalman_data_new)

            #print('Check')
            #print(np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))]))
            #print(np.linalg.norm([Y_pred_new[i][0,0] - Y[i] for i in range(len(Y))]))
            #print('Mate')

            #print(lab)
            #if s == 1:
            #    if error_AR1_data is None:
            #        error_AR1_data = np.array([pow(np.linalg.norm(Y_pred_new[i][0,0] - Y[i]),\
            #             2) for i in range(len(Y))])   #quadratic loss
            #    else:
                    #print(error_AR1_data.shape)
            #        error_AR1_data = np.vstack((error_AR1_data,\
            #             [pow(np.linalg.norm(Y_pred_new[i][0,0] - Y[i]), 2) for i in range(len(Y))]))
            #if s == t_t:
                # For the spectral filtering etc, we use:
                # loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)

                #I want to replace this chunk by kalman_filtering_siso.py
                #if error_kalman_data_new is None:
                    #error_Kalman_data = np.array([pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]),\
                    #     2) for i in range(len(Y))])
                #    error_kalman_data_new = np.array([pow(np.linalg.norm(Y_pred_new[i][0,0] - \
                #        Y[i]), 2) for i in range(len(Y))])
                #else:
                    #error_Kalman_data = np.vstack((error_Kalman_data,\
                    #     [pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]), 2) for i in range(len(Y))]))
                #    error_kalman_data_new = np.vstack((error_kalman_data_new,\
                #         [pow(np.linalg.norm(Y_pred_new[i][0,0] - Y[i]), 2) for i in range(len(Y))]))


#            loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2) 

#            pred_error.append(loss)

                #plt.plot([i[0,0] for i in Y_pred], label="Kalman" + sequence_label,\
                #     color=(42.0/255.0, 204.0 / 255.0, 200.0/255.0),\
                #          linewidth=2, antialiased = True)
            print(Y_pred_new)
            if s == t_t: 
                plt.plot([i[0,0] for i in Y_pred_new], label="Kalman_new" + sequence_label,\
                     color=(255.0/255.0, 165.0/255.0, 0),\
                          linewidth=2, antialiased = True)
            else:
                plt.plot([i[0,0] for i in Y_pred_new], label='AR(%i)' % (s+1)  + sequence_label,\
                     color=(42.0/255.0, 204.0 / 255.0, float(min(255.0,s))/255.0),\
                          linewidth=2, antialiased = True)

            plt.xlabel('Time')
            plt.ylabel('Prediction')
            #try:
            #print('AR')
            #print(error_AR1_data==error_AR1_data_1)
            #print('Kalman')
            #print(error_kalman_data_new==error_kalman_data_new_1)
            #except:
                #None

        if have_spectral_persistent:   #Spectral filtering and last-value prediction
            #using class WaveFilteringSisoFtl instead fubction WaveFilteringSisoFtl
            #predicted_output, M, error_spec, error_persist = WaveFilteringSisoFtl(sys, t_t, 5)
            wf_siso_ftl = WaveFilteringSisoFtl(sys, t_t, 5)
            predicted_output, M, error_spec = \
                wf_siso_ftl.y_pred_full, wf_siso_ftl.M,\
                     wf_siso_ftl.pred_error #wf_siso_ftl.pred_error_persistent

            wf_siso_persistent = WaveFilteringSisoFtlPersistent(sys, t_t, 5)
            #Here I replaced error_persist_data with error_persist
            predicted_persistent, M, error_persist = \
                wf_siso_persistent.y_pred_full, wf_siso_persistent.M,\
                    wf_siso_persistent.pred_error_persistent

            plt.plot(predicted_output, label='Spectral' + sequence_label,\
                 color='#1B2ACC', linewidth=2, antialiased = True)
            #spectral error
            if error_spec_data is None: error_spec_data = error_spec
            else: error_spec_data = np.vstack((error_spec_data, error_spec))
            #last-value error
            if error_persist_data is None: error_persist_data = error_persist
            else: error_persist_data = np.vstack((error_persist_data, error_persist))

        else:
            error_spec = []
            error_persist = []

        plt.legend()
        plt.savefig(p_p, format='pdf')
        plt.close('all')
        #plt.show()

    #In case we don't plot any of the filter, we assigns its error
    #parameters to [].

    if have_spectral_persistent:#means and stdevs of methods' errors
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = \
            error_stat(error_spec_data, error_persist_data)
    else:
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = [], [], [], []

    #Mean and stdev of auto-regression error
    error_AR1_mean = np.mean(error_AR1_data, 0)
    error_AR1_std = np.std(error_AR1_data, 0)
    if have_kalman:#mean and stdev of kalman filter
        #error_Kalman_mean = np.mean(error_Kalman_data, 0)
        #error_Kalman_std = np.std(error_Kalman_data, 0)
        error_kalman_mean_new = np.mean(error_kalman_data_new, 0)
        error_kalman_std_new = np.std(error_kalman_data_new, 0)
    else:
        #error_Kalman_mean, error_Kalman_std = [], []
        error_kalman_mean_new, error_kalman_std_new = [], []

    #if error_spec is None: error_spec = []
    #if error_persist is None: error_persist = []
    for (ylim, alphaValue) in [((0, 100.0), 0.2), ((0.0, 1.0), 0.05)]:
        for Tlim in [t_t-1, min(t_t-1, 20)]:
            #Plots Figure 2 and './outputs/AR.pdf'
            p3_for_test_identification2(ylim, have_spectral_persistent, Tlim, error_spec,
                               sequence_label, error_spec_mean, error_spec_std, alphaValue,
                               error_persist, error_persist_mean, error_persist_std,
                               error_AR1_mean, error_AR1_std,
                               have_kalman, error_kalman_mean_new, error_kalman_std_new, p_p)


    p_p.close()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    The function is taken from pyplot documentation.
    Create a heatmap from a numpy array and two lists of labels.
    Used by testNoiseImpact to implement Figure 3 and Figure 6.
    Originally the function comes from experiments.py file.

    Args:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def testNoiseImpact(t_t = 50, no_runs = 10, discretisation = 10):
    """
    Produces './outputs/noise.pdf'. Plots heatmap of process noise variance
    vs observation noise variance based on relative error between any two
    predictive algorithms. LaTeX shows the example of the ratio of the errors 
    of Kalman filter and AR(2)(see Figure 3 of Example 7).
    Originally the function comes from experiments.py file.

    Plots RMSE of AR Figure 6(left): 
    average RMSE of predictions of AR(s+ 1) as a function of the variance of the
    process noise (vertical axis) and observation noise (horizontal axis).

    Plots Figure 6(center): 
    The differences in average RMSE of Kalman filters and AR(s + 1) as a function
    of the variance of the process noise (vertical axis) and observation noise (horizontal axis).

    Plots Figure 6(right):
    The ratio (70) of the errors of Kalman filters and AR(s + 1) as a function of
    the variance of the process noise (vertical axis) and observation noise (horizontal axis).

    Args:
        t_t            : Time horizon.
        no_runs        : Number of runs.
        discretisation : Number of trajectories.  
    """
    
    filename = './outputs/noise_new.pdf'
    p_p = PdfPages(filename)

    for s in [1, 2, 3, 7]:
        data = np.zeros((discretisation, discretisation))
        diff = np.zeros((discretisation, discretisation))
        ratio = np.zeros((discretisation, discretisation))
        errKalman = np.zeros((discretisation, discretisation))
        errAR = np.zeros((discretisation, discretisation))
        ################# SYSTEM ###################
        #G = np.matrix([[0.999,0],[0,0.5]])   #G = \diag([0.999,0.5])
        G = G_mat
        #f_dash = np.matrix([[1,1]])          #F' = [1, 1]
        f_dash = f_dash_mat

        #Finding stdev of process and observation noises
        for proc_noise_i in range(discretisation):
            proc_noise_std = float(proc_noise_i + 1) / (discretisation - 1)
            for obs_noise_i in range(discretisation):
                obs_noise_std  = float(obs_noise_i + 1) / (discretisation - 1)

                for runNo in range(no_runs):
                    #Initalizing a Linear Dynamical System(LDS)
                    #As usual in the literature \cite{WestHarrison}, 
                    # we define a linear system $L = (G,F,v,W)$ as:
                    #\begin{eqnarray}
                    #\phi_{t} = G h_{t-1} + \omega_t \\
                    #Y_t = F' \phi_t + \nu_t,
                    #\end{eqnarray}
                    sys = DynamicalSystem(G,np.zeros((2,1)),f_dash,np.zeros((1,1)),
                                           process_noise='gaussian',
                                           observation_noise='gaussian',
                                           process_noise_std=proc_noise_std,
                                           observation_noise_std=obs_noise_std,
                                           timevarying_multiplier_b = None)
                    inputs = np.zeros(t_t)
                    sys.solve([[1],[1]],inputs,t_t)  #Finds true outputs of LDS
                    Y = [i[0,0] for i in sys.outputs]
                    #pdb.set_trace()
                    ############################################
                    kalman_siso = KalmanFilteringSISO(sys, G, f_dash,proc_noise_std, \
                        obs_noise_std, t_t,Y)
                    ########## PRE-COMPUTE FILTER PARAMS ###################
                    n, m, W, V, matrix_c, R, Q, matrix_a, Z = pre_comp_filter_params(G,\
                        f_dash, proc_noise_std, obs_noise_std, t_t)

                    #PREDICTION
                    #AR prediction
                    Y_pred_new,error_AR1_data,_ = \
                        kalman_siso.predict(s,errAR,errKalman)
                    Y_pred = prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y)
                    #Kalman prediction
                    Y_kalman_new,_,error_kalman_data_new = kalman_siso.predict_kalman()
                    Y_kalman = prediction_kalman(t_t, f_dash, G, matrix_a, sys, Z, Y)

                    '''Root-mean-square error(RMSE) of AR'''
                    data[proc_noise_i][obs_noise_i] += np.linalg.norm([Y_pred[i][0,\
                        0] - Y[i] for i in range(len(Y))])
                    diffHere = np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))])
                    #print(Y_kalman[0][0,0])
                    diffHere -= np.linalg.norm([Y_kalman[i][0,0] - Y[i] for i in range(min(len(Y),\
                        len(Y_kalman)))])
                    #print(diffHere)
                    '''Difference in RMSEs of AR and Kalman filter'''
                    diff[proc_noise_i][obs_noise_i] += diffHere
                    #print(len(Y))
                    #print(len(Y_kalman))
                    #Kalman filter error
                    errKalman[proc_noise_i][obs_noise_i] += pow(np.linalg.norm([Y_kalman[i][0,\
                        0] - Y[i] for i in range(min(len(Y),len(Y_kalman)))]), 2)
                    #Auto-regression error
                    errAR[proc_noise_i][obs_noise_i] += pow(np.linalg.norm([Y_pred[i][0,\
                        0] - Y[i] for i in range(len(Y))]), 2)
                    print(error_AR1_data==errAR)
                    print(error_kalman_data_new==errKalman)

        '''Calculating the average'''
        data = data / no_runs
        fig, ax = plt.subplots()
        tickLabels = [str(float(i+1) / 10) for i in range(11)]
        im, cbar = heatmap(data, tickLabels, tickLabels, ax=ax, cmap="YlGn",\
            cbarlabel="Avg. RMSE of AR(%i), %s runs" % (s+1, no_runs))
        plt.ylabel('Variance of process noise')
        plt.xlabel('Variance of observation noise')
        fig.tight_layout()
        plt.savefig(p_p, format='pdf')
        #plt.show()

        '''Calculating the average'''
        diff = diff / no_runs
        fig, ax = plt.subplots()
        tickLabels = [str(float(i+1) / 10) for i in range(11)]
        im, cbar = heatmap(diff, tickLabels, tickLabels, ax=ax, cmap="YlOrRd",\
            cbarlabel="Avg. diff. in RMSEs of AR(%i) and Kalman filter, %s runs" % (s+1, no_runs))
        plt.ylabel('Variance of process noise')
        plt.xlabel('Variance of observation noise')
        fig.tight_layout()
        plt.savefig(p_p, format='pdf')
        #plt.show()

        ratio = pow(errKalman / errAR, 2)
        fig, ax = plt.subplots()
        tickLabels = [str(float(i+1) / 10) for i in range(11)]
        im, cbar = heatmap(ratio, tickLabels, tickLabels, ax=ax, cmap="PuBu",\
            cbarlabel="Ratios of agg. errors of Kalman and AR(%i), %s runs" % (s+1, no_runs))
        plt.ylabel('Variance of process noise')
        plt.xlabel('Variance of observation noise')
        fig.tight_layout()
        plt.savefig(p_p, format='pdf')

    p_p.close()


def testImpactOfS(t_t = 200, no_runs = 100, sMax = 15):
    """
    Creates file './outputs/impacts.pdf', which stores plots of average error of auto-regression 
    as a function of regression depth s. In the main paper we present it again with Example 7 
    and Figure 4.
    Increasing s decreases the error, until the error approaches that of the Kalman filter. 
    For a given value of the observation noise, the convergence w.r.t s is slower for 
    smaller process noise.
    Originally the function comes from experiments.py file.

    Args:
        t_t     : Time horizon.
        no_runs : Number of runs.
        sMax    : Number of auto-regressive terms.

    Raises:
        Exits if sMax > t_t. 
    """

    if sMax > t_t:
        print("The number of s to test must be less than the horizon t_t.")
        exit()

    filename = './outputs/impacts.pdf'
    p_p = PdfPages(filename)

    for (proc_noise_std, obs_noise_std, linestyle) in [ (0.1, 0.1, "dotted"),\
        (0.1, 1.0, "dashdot"),  (1.0, 0.1, "dashed"), (1.0, 1.0, "solid") ]:
        errAR = np.zeros((sMax+1, no_runs))  #Auto-regression
        ################# SYSTEM ###################
        #G = np.matrix([[0.999,0],[0,0.5]])
        #f_dash = np.matrix([[1,1]])
        G = G_mat
        f_dash = f_dash_mat
        for s in range(1, sMax):

            for runNo in range(no_runs):
                sys = DynamicalSystem(G,np.zeros((2,1)),f_dash,np.zeros((1,1)),
                                       process_noise='gaussian',
                                       observation_noise='gaussian',
                                       process_noise_std=proc_noise_std,
                                       observation_noise_std=obs_noise_std,
                                       timevarying_multiplier_b = None)
                inputs = np.zeros(t_t)
                sys.solve([[1],[1]],inputs,t_t)
                Y = [i[0,0] for i in sys.outputs]
                #pdb.set_trace()
                ############################################

                ########## PRE-COMPUTE FILTER PARAMS ###################
                n, m, W, V, matrix_c, R, Q, matrix_a, Z = pre_comp_filter_params(G, f_dash,\
                    proc_noise_std, obs_noise_std, t_t)

                #AR PREDICTION
                Y_pred = prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y)


                errAR[s][runNo] = pow(np.linalg.norm([Y_pred[i][0,\
                    0] - Y[i] for i in range(min(len(Y), len(Y_pred)))]), 2) / t_t


        error_AR1_mean = np.mean(errAR, 1)
        error_AR1_std = np.std(errAR, 1)
        print(len(error_AR1_mean))
        alphaValue = 0.2
        cAR1 = (proc_noise_std, obs_noise_std, 1.0/255)
        #plt.plot(range(1, sMax), error_AR1_mean[1:], label='AR(2)', color=cAR1,\
        # linewidth=2, antialiased = True)
        #plt.fill_between(range(1, sMax), (error_AR1_mean-error_AR1_std)[1:],\
        # (error_AR1_mean+error_AR1_std)[1:], alpha=alphaValue, edgecolor=cAR1,\
        # linewidth=2, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue,
        lab = "W = %.2f, V = %.2f" % (proc_noise_std, obs_noise_std)
        plt.plot(range(sMax+1)[1:-1], error_AR1_mean[1:-1], color=cAR1, linewidth=2,\
            antialiased = True, label = lab, linestyle= linestyle)
        plt.fill_between(range(sMax+1)[1:-1], (error_AR1_mean-error_AR1_std)[1:-1],\
            (error_AR1_mean+error_AR1_std)[1:-1], alpha=alphaValue, facecolor = cAR1,\
                edgecolor=cAR1, linewidth=2, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue,
        plt.xlabel('Number s of auto-regressive terms, past the first one')
        plt.ylabel('Avg. error of AR(s), %i runs' % no_runs )
        plt.ylim(0, 1.5)
        plt.legend()
        plt.savefig(p_p, format='pdf')
    p_p.close()


def testSeqD0(no_runs = 100):
    """
    Makes several initiations of test_identification function so as to plot "logratio.pdf" and
    "seq0.pdf", "seq1.pdf", "seq2.pdf". Originally the function comes from experiments.py file.

    Args:
        no_runs: Number of runs.
    """

    plain = False
    lr = True

    matlabfile_in = './OARIMA_code_data/data/setting6.mat'

    if plain:
        varname_in = "seq_d0"
        ts = TimeSeries(matlabfile = matlabfile_in, varname="seq_d0")
        t_t = len(ts.outputs)
        test_identification(ts, "%s-complete"%varname_in, no_runs, t_t, 5,\
            sequence_label = varname_in, have_spectral_persistent = False)
        t_t = min(20000, len(ts.outputs))
        test_identification(ts, "%s-20000"%varname_in, no_runs, t_t, 5,\
            sequence_label = varname_in, have_spectral_persistent = False)
        t_t = min(2000, len(ts.outputs))
        test_identification(ts, "%s-2000"%varname_in, no_runs, t_t, 5,\
            sequence_label = varname_in, have_spectral_persistent = False)
        t_t = min(200, len(ts.outputs))
        test_identification(ts, "%s-200"%varname_in, no_runs, t_t, 5,\
            sequence_label = varname_in, have_spectral_persistent = False)
        t_t = min(100, len(ts.outputs))
        test_identification(ts, "%s-short-k5"%varname_in, 1, t_t, 5,\
            sequence_label = varname_in)
        #test_identification(ts, "seq0-short-k50", 1, t_t, 50, 27, 37, sequence_label = "seq_d0")
        #test_identification(ts, "seq0-short-k5", 1, t_t, 5, sequence_label = "seq_d0")
        #test_identification(ts, "seq0-short-k50", 1, t_t, 50, sequence_label = "seq_d0")
    if lr:
        varname_in = "lr_d0"
        ts = TimeSeries(matlabfile = matlabfile_in, varname="seq_d0")
        ts.logratio()
        t_t = len(ts.outputs) # has to go after the log-ratio truncation by one
        test_identification(ts, "logratio-complete", no_runs, t_t, 5, sequence_label = varname_in,\
            have_spectral_persistent = False)
        t_t = min(20000, len(ts.outputs))
        test_identification(ts, "logratio-20000", no_runs, t_t, 5,  sequence_label = varname_in,\
            have_spectral_persistent = False)
        t_t = min(2000, len(ts.outputs))
        test_identification(ts, "logratio-2000", no_runs, t_t, 5, sequence_label = varname_in,\
            have_spectral_persistent = False)
        t_t = min(200, len(ts.outputs))
        test_identification(ts, "logratio-200", no_runs, t_t, 5, sequence_label = varname_in,\
            have_spectral_persistent = False)
        t_t = min(100, len(ts.outputs))
        test_identification(ts, "logratio-short-k5", no_runs, t_t, 5, sequence_label = varname_in)

def test_AR():
    """
    Function implements Algorithm 1(On-line Gradient Descent).
    Originally the function comes from experiments.py file.
    """
    matlabfile_in = './OARIMA_code_data/data/setting6.mat'
    varname_in = "seq_d0"

    ts = TimeSeries(matlabfile = matlabfile_in, varname=varname_in)
    t_t = min(100, len(ts.outputs))
    s=10
    matrix_d=10.
    theta = [0 for i in range(s)]

    for t in range(s,t_t):
        eta = pow(float(t),-0.5) #learning rate

        Y = ts.outputs[t]        #true outputs

        loss = cost_ar(theta, Y, list(reversed(ts.outputs[t-s:t]))) #loss function
        grad = gradient_ar(theta, Y, list(reversed(ts.outputs[t-s:t]))) #gradient

        print("Loss: at time step %d :" % (t), loss)
        theta = [theta[i] -eta*grad[i] for i in range(len(theta))] #gradient step
        norm_theta = np.linalg.norm(theta)

        if norm_theta>matrix_d: theta = [matrix_d*i/norm_theta for i in theta] #projection step

########################  inputlds ################



############# onlinelds ##############

# def do_filter_step(G,F,V,W, Id, Y_curr, m_prev,c_prev):


def cost_ar(theta, *args):
    """
    Loss function of auto-regression.
    After the prediction is made, the true observation is revealed to 
    the algorithm, and a loss associated with the prediction is computed.
    Here we consider the quadratic loss for simplicity.
    Originally the function comes from onlinelds.py file.

    Args:
        theta   : auto-regressive parameters.
        args[0] : observation at time t
        args[1] : past s observations (most to least recent: t-1 to t-1-s)

    Returns:
        Quadratic loss function of auto-regression.
    """
    #\hat{y}_t(\theta) = \sum_{i=0}^{s-1} \theta_{i} Y_{t-i-1}
    #\ell_t(\theta) := \ell(Y_t,\hat{y}_t(\theta))

    return pow(float(args[0]) - np.dot(args[1], theta), 2)


def gradient_ar(theta, *args):
    """
    Gradient function of auto-regression.
    We use the general scheme of on-line gradient decent algorithms,
    where the update goes against the direction of the gradient of the current loss. 
    In addition, it is useful to restrict the state to a bounded domain.
    Originally the function comes from onlinelds.py file.

    Args:
        theta   : s parameters.
        args[0] : Observation.
        args[1] : Past s observations.

    Returns:
        Gradient function of auto-regression.
    """
    #& -2\Brack{Y_t - \sum_{i=0}^{s-1} \theta_i Y_{t-i-1}}
    #\Brack{Y_{t-1},Y_{t-2},\ldots, Y_{t-s}}
    g = [(float(args[0]) - np.dot(args[1], theta)) * i for i in args[1]]

    return np.squeeze(-2 * np.array(g).reshape(-1, 1))


###### MATLAB functions ##########


def test_arima_ogd(i, mk, lrate, data):
    """
    Used to test arima_ogd function for i=10 case.
    The test cases are based on MATLAB:
    The test numbers were taken from the output of MATLAB function,
    the random array w is fixed.

    Args:
        i     : Iterative number. In range from mk till data - 1.
        mk    : Integer number. Here we used 10.
        lrate : Learning rate. Assigned 1 in example.py.
        data  : Array of 10000 elements.

    Raises:

    """
    # the random array w is fixed here
    w = np.array([[0.276926, 0.023466, 0.480833, 0.507039, 0.710869, 0.188331, 0.374130,\
        0.290949, 0.724284, 0.562128]])

    data_i_test = 0.0685
    diff_test = 0.0975 #out from MATLAB function
    w_test = np.array([[0.39243, 0.17813, 0.59069, 0.52301, 0.60476, 0.10548, 0.37286,\
        0.29994, 0.72463, 0.49051]]) #out from MATLAB function

    diff = diff_calc(w, data, mk, i)
    wi = w_calc(w, data, mk, i, diff, lrate)

    # tests
    print('arima_ogd test:')
    if np.round(data[i], 4) == data_i_test:
        print('test data[i] - OK')
    else:
        print('ERROR: arima_ogd - data[i]')

    if np.round(diff, 4) == diff_test:
        print('test diff - OK')
    else:
        print('ERROR: arima_ogd - diff')

    if (np.round(wi, 4) == np.round(w_test, 4)).all():
        print('test w - OK')
    else:
        print('ERROR: arima_ogd - w')


def test_arima_ons(i, mk, lrate, data, A_trans_in):
    '''
    to test arima_ons function
    the test casees are based on MATLAB:
    the test numbers were taken from the output of MATLAB function
    the random array w is fixed
    :param i:
    :param mk:
    :param lrate:
    :param data:
    :return:
    '''
    #the random array w is fixed here
    w = np.array([[0.255152,   0.104954,   0.162370,   0.025481,   0.899503,   0.276570,   0.327885,   0.302031,   0.158955,  0.199591]])

    # out from MATLAB function
    data_i_test = 0.0685
    diff_test = 0.2718
    grad_test = np.array([-3.2207e-01,  -4.3127e-01,  -3.0633e-01,  -4.4533e-02,   2.9586e-01,   2.3102e-01,   3.5499e-03,  -2.5058e-02,  -9.6224e-04,   1.9970e-01])
    w_test = np.array([[0.406285,   0.307332,   0.306120,   0.046378,   0.760664,   0.168159,   0.326219,   0.313790,   0.159407,   0.105881]]) #out from MATLAB function

    diff = diff_calc(w, data, mk, i)
    grad = grad_calc(data, i, mk, diff)

    # tests
    print('arima_ons test:')
    if np.round(data[i], 4) == data_i_test:
        print('test data[i] - OK')
    else:
        print('ERROR: arima_ons - data[i]')

    if np.round(diff, 4) == diff_test:
        print('test diff - OK')
    else:
        print('ERROR: arima_ons - diff')

    if (np.round(grad, 4) == np.round(grad_test, 4)).all():
        print('test grad - OK')
    else:
        print('ERROR: arima_ogd - grad')

def arima_ogd(data, options):
    """
    from arima_ogd.m
    Used by example.py. ARIMA Online Newton Step algorithm.
    The function was originally written in MATLAB by Liu, C.; Hoi, S. C. H.; Zhao, P.; and Sun, J.
    It's described in their work "Online arima algorithms for time series prediction." 

    Args:
        data: Array of 10000 elements.
        options:Instance of ClassOptions class.
    
    Returns:

    """
    # MATLAB:
    # mk = options.mk;
    # lrate = options.lrate;
    # w = options.init_w;
    mk = options.mk    #10
    lrate = options.lrate #learning rate. Assigned 1.
    w = options.init_w  #Uniform distribution array with options.mk number of columns.

    # MATLAB:
    # list = [];
    # SE = 0;
    list = np.array([])
    SE = 0

    # MATLAB:
    # for i = mk+1:size(data,2)
    for i in range(mk, len(data)):#from 10 till 9999

        #MATLAB:
        # diff = w*data(i-mk:i-1)'-data(i);%'
        #w = w - data(i-mk:i-1)*2*diff/sqrt(i-mk)*lrate;
        diff = diff_calc(w, data, mk, i)
        w = w_calc(w, data, mk, i, diff, lrate)

        # MATLAB:
        #SE = SE + diff ^ 2;
        SE += diff**2

        # MATLAB:
        #if mod(i,options.t_tick)==0
        #   list = [list; sqrt(SE/i)];
        #end
        if (i % options.t_tick) == 0:
            list = np.append(list, np.sqrt(SE / i))

        #make column from row
        list = list.reshape(list.size, -1)

        # test for i == 10
        if i == 10:
            test_arima_ogd(i, mk, lrate, data)

    return list, w


def diff_calc(w, data, mk, i):
    """
    Auxiliary function to implement ARIMA in python. Others functions use it in their
    iterations.
    MATLAB: diff = w*data(i-mk:i-1)'-data(i);
    remember! MATLAB_data(1) == Python_data[0]
    we have to convert data[] from 1D vector to a numpy matrix (2D) to apply the transpose
    OR data[].reshape(-1,1) can be also used to mimick the transpose

    Args:
        w: Uniform distribution array with options.mk number of columns. 
        data: Array of 10000 elements.
        mk: Integer number. Here we used 10.
        i: Iterative number. In range from mk till data - 1.
    
    Returns:

    """

    return np.asscalar(np.dot(w, np.matrix(data[i - mk:i]).T) - data[i])


def w_calc(w, data, mk, i, diff, lrate):
    """
    Auxiliary function to implement ARIMA in python. Others functions use it in their
    iterations.
    MATLAB: w = w - data(i-mk:i-1)*2*diff/sqrt(i-mk)*lrate;
    
    Args:
        w: Uniform distribution array with options.mk number of columns. 
        data: Array of 10000 elements.
        mk: Integer number. Here we used 10.
        i: Iterative number. In range from mk till data - 1.
        diff: Result of diff_calc function
        lrate: Learning rate. Assigned 1 in example.py.
    
    Returns:
    """

    return w - data[i - mk:i] * 2 * diff / np.sqrt(i - mk + 1) * lrate

def grad_calc(data, i, mk, diff):
    """
    MATLAB: grad = 2*data(i-mk:i-1)*diff
    Used by function arima_ons.

    Args:
        data: Array of 10000 elements.
        i: Iterative number. In range from mk till data - 1.
        mk: Integer number. Here we used 10.
        diff: Result of diff_calc function

    Returns:
        Gradient.
    """
    return 2 * data[i - mk:i] * diff

def A_trans_calc(A_trans, grad):
    """
    MATLAB:
    A_trans = A_trans - A_trans * grad' * grad * A_trans/(1 + grad * A_trans * grad');
    we have to convert data[] from 1D vector to a numpy matrix (2D) to apply the transpose
    OR data[].reshape(-1,1) can be also used to mimick the transpose.

    Args:
        A_trans: np.eye(mk) * epsilon
        grad: Gradient, the return of the function grad_calc.

    Returns:
    
    """
    #@ is matrix multiply symbol
    A_trans = A_trans - A_trans @ np.matrix(grad).T @ np.matrix(grad) @ A_trans / (1 +\
        grad @ A_trans @ grad.T)

    return A_trans

def w_calc_arima_ons(w, lrate, grad, A_trans):
    """
    MATLAB: w = w - lrate * grad * A_trans
    Calculation of the weight with Gradient Descent algorithm.

    Args:
        w: Uniform distribution array with options.mk number of columns. 
        lrate: Learning rate. Assigned 1 in example.py.
        grad: Gradient, the return of the function grad_calc.
        A_trans: Return of the function A_trans_calc.

    Returns:
        Weight after an iteration of the gradient descent algorithm.
    """

    w = w - lrate * grad @ A_trans
    return w

def arima_ons(data, options):
    """
    from arima_ons.m. ARIMA Online Newton Step algorithm
    Used by example.py.
    The function was originally written in MATLAB by Liu, C.; Hoi, S. C. H.; Zhao, P.; and Sun, J.
    It's described in their work "Online arima algorithms for time series prediction." 

    Args:
        data: Array of 10000 elements.
        options:Instance of ClassOptions class.
    
    Returns:
    """

    #MATLAB:
    #mk = options.mk;
    #lrate = options.lrate;
    #w = options.init_w;
    #epsilon = options.epsilon;
    mk = options.mk
    lrate = np.array([[options.lrate]])
    w = options.init_w
    epsilon = options.epsilon

    #MATLAB:
    #list = [];
    #SE = 0;
    #A_trans = eye(mk) * epsilon;
    list = np.array([])
    SE = 0
    A_trans = np.eye(mk) * epsilon

    # MATLAB:
    # for i = mk+1:size(data,2)
    for i in range(mk, len(data)):  #from 10 till 9999

        #MATLAB: diff = w*data(i-mk:i-1)'-data(i);
        diff = diff_calc(w, data, mk, i)

        #MATLAB: grad = 2*data(i-mk:i-1)*diff;
        grad = grad_calc(data, i, mk, diff)

        # MATLAB: A_trans = A_trans - A_trans * grad' * grad * A_trans/(1 + grad *\
        # A_trans * grad');
        A_trans = A_trans_calc(A_trans, grad)

        # MATLAB: w = w - lrate * grad * A_trans ;
        w = w_calc_arima_ons(w, lrate, grad, A_trans) #weight modified by gradient descent

        # MATLAB:
        #SE = SE + diff ^ 2;
        SE = SE + diff ** 2

        # MATLAB:
        #if mod(i,options.t_tick)==0
        #   list = [list; sqrt(SE/i)];
        #end
        if (i%options.t_tick) == 0:
            list = np.append(list, np.sqrt(SE / i))

        #make column from row
        list = list.reshape(list.size, -1)

        # test for i == 10
        if i == 10:
            test_arima_ons(i, mk, lrate, data, A_trans)

    return list, w

### New funct ####

def lab(s, eta_zero):
    """
    Gives a label to auto-regression outputs and labels in seq0,seq1,seq2 pdfs.
    Returns:
        lab1: auto-regression label. Example: "AR(2), c = 2500".
    """
    # lab1 = 'AR(3) / OGD, c_0 = ' + str(eta_zero)
    lab1 = "AR(" + str(s) + "), c = " + str(int(eta_zero))
    return lab1


def plot_p1(ymin, ymax, inputs, sequence_label, have_spectral_persistent,
            predicted_spectral, predicted_ar,
            sys, p_p):
    """
    Plots seq0, seq1, seq2, logratio pdf files.

    Args:
        ymin:                       Minimal value of y-axis.
        ymax:                       Maximal value of y-axis.
        inputs:                     Input to the system matrix.
        sequence_label:             Plot's label.
        have_spectral_persistent:   True if we want to build spectral and persistent filters.
        predicted_spectral:         Predicted values of spectral filter. If
                                    have_spectral_persistent is False, it's equal to an empty list.
        predicted_ar:               Predicted values of auto-regression.
        sys:                        Linear Dynamical System created with DynamicalSystem class.
        p_p:                        PDF file, to which are export the plots.
    """

    p1 = plt.figure()
    if ymax and ymin: plt.ylim(ymin, ymax)
    if sum(inputs[1:]) > 0: plt.plot(inputs[1:], label='Input')
    if sequence_label:
        plt.plot([float(i) for i in sys.outputs][1:], label=sequence_label, color='#000000',\
        linewidth=2, antialiased=True)
    else:
        plt.plot([float(i) for i in sys.outputs][1:], label='Output', color='#000000',\
        linewidth=2, antialiased=True)
    # plt.plot([-i for i in predicted_output], label='Predicted output') #for some reason,\
    # usual way produces -ve estimate
    if have_spectral_persistent:
        plt.plot([i for i in predicted_spectral], label='Spectral')

    plt.plot(predicted_ar, label=lab)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Output')
    p1.show()
    p1.savefig(p_p, format='pdf')


def plot_p2(have_spectral_persistent, error_spec, error_persist, error_ar, lab, p_p):
    """
    Plots seq0, seq1, seq2, logratio pdf files.

    Args:
        have_spectral_persistent:      True if we want to build spectral and persistent filters.
        error_spec:                    Spectral filter error.
        error_persist:                 Persistent filter error.
        error_ar:                      Auto-regression error.
        lab:                           Auto-regression plot label.
        p_p:                           PDF file, to which are export the plots.
    """

    p2 = plt.figure()
    plt.ylim(0, 20)
    if have_spectral_persistent:
        plt.plot(error_spec, label='Spectral')
        plt.plot(error_persist, label='Persistence')
    plt.plot(error_ar, label=lab)
    plt.legend()
    p2.show()
    plt.xlabel('Time')
    plt.ylabel('Error')
    p2.savefig(p_p, format='pdf')


def plot_p3(ymin, ymax, have_spectral_persistent, error_spec_mean, error_spec_std,
            error_persist_mean, error_persist_std,
            error_ar_mean, error_ar_std,
            t_t, p_p):
    """
    Plots seq0, seq1, seq2, logratio pdf files.

    Args:
        ymin:                          Minimal value of y-axis.
        ymax:                          Maximal value of y-axis.
        have_spectral_persistent:      True if we want to build spectral and persistent filters.
        error_spec_mean:               Mean error of spectral filtering.
        error_spec_std:                Std of spectral filtering error.
        error_persist_mean:            Mean error of last-value prediction.
        error_persist_std:             Std of last-value prediction error.
        error_ar_mean:                 Mean error of auto-regression.
        error_ar_std:                  Std of auto-regression error.
        p_p:                           PDF file, to which are export the plots.
    """

    p3 = plt.figure()
    if ymax and ymin: plt.ylim(ymin, ymax)
    if have_spectral_persistent:
        plt.plot(error_spec_mean, label='Spectral', color='#1B2ACC', linewidth=2, antialiased=True)
        plt.fill_between(range(0, t_t - 1), error_spec_mean - error_spec_std,\
            error_spec_mean + error_spec_std, alpha=0.2,edgecolor='#1B2ACC', facecolor='#089FFF',\
                linewidth=1, antialiased=True)
        plt.plot(error_persist_mean, label='Persistence', color='#CC1B2A', linewidth=2,\
            antialiased=True)
        plt.fill_between(range(0, t_t - 1), error_persist_mean - error_persist_std,\
            error_persist_mean + error_persist_std, alpha=0.2, edgecolor='#CC1B2A',\
                facecolor='#FF0800', linewidth=1, antialiased=True)

    cAR1 = (42.0 / 255, 204.0 / 255.0, 1.0 / 255) #plot color
    bAR1 = (1.0, 204.0 / 255.0, 0.0)  # , alphaValue
    plt.ylim(0, 20)
    plt.plot(error_ar_mean, label='AR(3)', color=cAR1, linewidth=2, antialiased=True)
    plt.fill_between(range(0, t_t), error_ar_mean - error_ar_std, error_ar_mean + error_ar_std,\
        alpha=0.2, edgecolor=cAR1, facecolor=bAR1, linewidth=1, antialiased=True)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')
    p3.savefig(p_p, format='pdf')


def error_stat(error_spec_data, error_persist_data):
    """
    if have_spectral_persistent:
    Returns:
        error_spec_mean:    Mean error of spectral filtering
        error_spec_std:     Std of spectral filtering error
        error_persist_mean: Mean error of last-value prediction
        error_persist_std:  Std of last-value prediction error
    """

    error_spec_mean = np.mean(error_spec_data, 0)
    error_spec_std = np.std(error_spec_data, 0)
    error_persist_mean = np.mean(error_persist_data, 0)
    error_persist_std = np.std(error_persist_data, 0)

    return error_spec_mean, error_spec_std, error_persist_mean, error_persist_std

def pre_comp_filter_params(G, f_dash, proc_noise_std, obs_noise_std, t_t):
    """
    Kalman filter auxiliary recursive parameters calculation.
    """

    n = G.shape[0]   #input vector
    m = f_dash.shape[0] #observation vector

    W = proc_noise_std ** 2 * np.matrix(np.eye(n))  #covariance matrix of process noise
    V = obs_noise_std ** 2 * np.matrix(np.eye(m))   #observation noise covariance

    # m_t = [np.matrix([[0],[0]])]
    matrix_c = [np.matrix(np.eye(2))]
    R = []
    Q = []
    matrix_a = []
    Z = []

    for t in range(t_t):
        R.append(G * matrix_c[-1] * G.transpose() + W)
        # if t == 1:
        #     print('d')
        #     print(R)
        #     print('f')
        Q.append(f_dash * R[-1] * f_dash.transpose() + V)

        #LaTeX A_t &=& R_t F  / Q_t
        matrix_a.append(R[-1] * f_dash.transpose() * np.linalg.inv(Q[-1]))

         #C_t &=& R_t - A_t Q_t A'_t
        matrix_c.append(R[-1] - matrix_a[-1] * Q[-1] * matrix_a[-1].transpose())

        #In general, set $Z_t = G(I-F\otimes A_t)$ and $Z = G(I-F \otimes A)$.
        Z.append(G * (np.eye(2) - matrix_a[-1] * f_dash))

    return n, m, W, V, matrix_c, R, Q, matrix_a, Z

def p3_for_test_identification2(ylim, have_spectral_persistent, Tlim, error_spec, sequence_label,
                               error_spec_mean, error_spec_std, alphaValue,
                               error_persist, error_persist_mean, error_persist_std,
                               error_AR1_mean, error_AR1_std,
                               have_kalman, error_Kalman_mean, error_Kalman_std, p_p):

    """
    Plots Figure 2,5 after getting all the errors data.
    LaTeX
    In Figure 2, we compare the prediction error for 4 methods: 
    the standard baseline last-value prediction $\hat{y}_{t+1} := y_t$, also 
    known as persistence prediction, the spectral filtering of 
    \cite{hazan2017online}, Kalman filter, and AR(2).

    We first continue the Example \ref{HazanEx} form the main body of the 
    paper, with a system given by (\ref{eq:experem1_system_hazan}) and 
    $v=w=0.5$. Figure \ref{fig1}(right) shows a sample observations 
    trajectory of the system, together with forecast for the four methods. 
    Figure \ref{fig1}(left) show the mean and standard deviations of the 
    errors for the first 500 time steps. Figure \ref{fig1brief} in the main 
    text is the restriction of this Figure \ref{fig1}(left) to the first 20 
    steps. Similarly to Figure \ref{fig1brief}, we observe that the AR(2) 
    predictions are better than the spectral and persistence methods, and 
    worse than the Kalman filter, since only two first terms are considered. 
    """
    # p3 = plt.figure()
    p3, ax = plt.subplots()
    plt.ylim(ylim)
    if have_spectral_persistent:
        plt.plot(range(0, Tlim), error_spec[:Tlim], label='Spectral' + sequence_label,\
            color='#1B2ACC', linewidth=2, antialiased=True)
        plt.fill_between(range(0, Tlim), (error_spec_mean - error_spec_std)[:Tlim],\
                         (error_spec_mean + error_spec_std)[:Tlim], alpha=alphaValue,\
                        edgecolor='#1B2ACC',facecolor='#089FFF', linewidth=1, antialiased=True)
        plt.plot(range(0, Tlim), error_persist[:Tlim], label='Persistence' + sequence_label,\
            color='#CC1B2A', linewidth=2, antialiased=True)
        plt.fill_between(range(0, Tlim), (error_persist_mean - error_persist_std)[:Tlim],\
                        (error_persist_mean + error_persist_std)[:Tlim], alpha=alphaValue,\
                        edgecolor='#CC1B2A',facecolor='#FF0800', linewidth=1, antialiased=True)

    # import matplotlib.transforms as mtransforms
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transData)
    # trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    cAR1 = (42.0 / 255, 204.0 / 255.0, 1.0 / 255)
    bAR1 = (1.0, 204.0 / 255.0, 0.0)  # , alphaValue
    print(cAR1)
    print(bAR1)
    # print(error_AR1_data)
    # print(error_AR1_mean)
    # print(Tlim)
    plt.plot(error_AR1_mean[:Tlim], label='AR(2)' + sequence_label, color=cAR1, linewidth=2,\
        antialiased=True)
    plt.fill_between(range(0, Tlim), (error_AR1_mean - error_AR1_std)[:Tlim],\
        (error_AR1_mean + error_AR1_std)[:Tlim],\
            alpha=alphaValue, edgecolor=cAR1, facecolor=bAR1, linewidth=1,\
                antialiased=True)  # transform=trans) #offset_position="data") alpha=alphaValue,

    if have_kalman:
        cK = (42.0 / 255.0, 204.0 / 255.0, 200.0 / 255.0)
        bK = (1.0, 204.0 / 255.0, 200.0 / 255.0)  # alphaValue
        print(cK)
        print(bK)
        plt.plot(error_Kalman_mean[:Tlim], label='Kalman' + sequence_label, color=cK,\
            linewidth=2, antialiased=True)
        plt.fill_between(range(0, Tlim), (error_Kalman_mean - error_Kalman_std)[:Tlim],\
                         (error_Kalman_mean + error_Kalman_std)[:Tlim], alpha=alphaValue,\
                            facecolor=bK, edgecolor=cK, linewidth=1,\
                                antialiased=True)  # transform = trans) #offset_position="data")

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')
    # p3.show()
    p3.savefig(p_p, format='pdf')


def prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y):

    """
    Auto-regression prediction values.
    Finds the formula for Figure 1(AR(s+1)):
    The unrolling of the forecast $f_{t+1}$.
    The remainder term goes to zero exponentially fast with $s$, by Lemma
    """

    Y_pred = []
    for t in range(t_t):
        Y_pred_term1 = f_dash * G * matrix_a[t] * sys.outputs[t]
        if t == 0:  #can deal with it to decrease time consumption
            Y_pred.append(Y_pred_term1)
            continue
        acc = 0
        for j in range(min(t, s) + 1):
            for i in range(j + 1):
                if i == 0:
                    ZZ = Z[t - i]
                    continue
                ZZ = ZZ * Z[t - i]
            acc += ZZ * G * matrix_a[t - j - 1] * Y[t - j - 1]
        Y_pred.append(Y_pred_term1 + f_dash * acc) #Why didn't we add remainder term?

    return Y_pred


def prediction_kalman(t_t, f_dash, G, matrix_a, sys, Z, Y):
    """
    Kalman filter prediction values
    """
    Y_kalman = []
    for t in range(t_t):
        Y_pred_term1 = f_dash * G * matrix_a[t] * sys.outputs[t]
        if t == 0:
            Y_kalman.append(Y_pred_term1)
            continue

        accKalman = 0
        #We don't have range(min(t,s)+1) as we do for prediction function
        for j in range(t + 1):
            for i in range(j + 1):
                if i == 0:
                    ZZ = Z[t - i]
                    continue
                ZZ = ZZ * Z[t - i]
            accKalman += ZZ * G * matrix_a[t - j - 1] * Y[t - j - 1]
        Y_kalman.append(Y_pred_term1 + f_dash * accKalman)

    return Y_kalman



########## NEW ###############




