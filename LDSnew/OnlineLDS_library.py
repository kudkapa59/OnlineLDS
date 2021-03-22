#This library is compilation of python scripts (experiments.py, onlinelds.py, inputlds.py)
#and matlab scripts (example.m, arima_ogd.m, arima_ons.m)
# source for each function is given in comments (experiments.py, onlinelds.py, inputlds.py, etc)
###


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
from AddClasses.ds.dynamical_system import DynamicalSystem
from AddClasses.ts.time_series import TimeSeries
#from AddClasses.filters.real_filters.wave_filtering_siso import WaveFilteringSISO
from AddClasses.filters.real_filters.wave_filtering_siso_ftl import WaveFilteringSisoFtl
#from AddClasses.filters.real_filters.kalman_filtering_siso import KalmanFilteringSISO
from AddClasses.matlab_options.matlab_class_options import ClassOptions

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# debugging
pdb.Pdb.complete=rlcompleter.Completer(locals()).complete

#from def4lib import *

VERBOSE = False

def close_all_figs():
    '''
    from experiments.py
    '''
    plt.close('all')

def test_identification(sys, filename_stub = "test", no_runs = 2,
                       t_t = 100, k = 5, eta_zeros = None, ymin = None, ymax = None,
                       sequence_label = None, have_spectral = True):
    '''
    from experiments.py
    no_runs is the number of runs, t_t is the time horizon, k is the number of filters,
    '''


    if k>t_t:
        print("Number of filters (k) must be less than or equal to the number of time-steps (t_t).")
        exit()
    if not eta_zeros:
        eta_zeros = [1.0, 2500.0]
    print("eta_zeros:")
    print(eta_zeros)

    filename = './outputs/' + filename_stub+'.pdf'
    p_p = PdfPages(filename)

    error_ar_data = None
    error_spec_data = None
    error_persist_data = None

    for i in range(no_runs):
        print("run %i" % i)
        inputs = np.zeros(t_t)
        sys.solve([[1],[0]],inputs,t_t)

        if have_spectral:
            #using class WaveFilteringSisoFtl instead fubction WaveFilteringSisoFtl
            #predicted_spectral, M, error_spec, error_persist = WaveFilteringSisoFtl(sys, t_t, k)
            wf_siso_ftl = WaveFilteringSisoFtl(sys, t_t, k, VERBOSE)
            predicted_spectral, M, error_spec, error_persist = \
                wf_siso_ftl.y_pred_full, wf_siso_ftl.M,\
                    wf_siso_ftl.pred_error, wf_siso_ftl.pred_error_persistent

            if error_spec_data is None:
                error_spec_data = error_spec
            else:
                error_spec_data = np.vstack((error_spec_data, error_spec))
            if error_persist_data is None:
                error_persist_data = error_persist
            else:
                error_persist_data = np.vstack((error_persist_data, error_persist))

        for eta_zero in eta_zeros:
            error_ar = np.zeros(t_t)
            predicted_ar = np.zeros(t_t)
            s=2
            matrix_d=1.
            theta = [0 for i in range(s)]
            for t in range(s,t_t):
                eta = pow(float(t),-0.5) / eta_zero
                Y = sys.outputs[t]
                loss = cost_ar(theta, Y, list(reversed(sys.outputs[t-s:t])))
                error_ar[t] = pow(loss, 0.5)
                grad = gradient_ar(theta, Y, list(reversed(sys.outputs[t-s:t])))
                #print("Loss: at time step %d :" % (t), loss)
                theta = [theta[i] -eta*grad[i] for i in range(len(theta))] #gradient step
                norm_theta = np.linalg.norm(theta)
                if norm_theta>matrix_d:
                    theta = [matrix_d*i/norm_theta for i in theta] #projection step
                predicted_ar[t] = np.dot(list(reversed(sys.outputs[t-s:t])),theta)

            if error_ar_data is None:
                error_ar_data = error_ar
            else:
                error_ar_data = np.vstack((error_ar_data, error_ar))

            if not have_spectral:
                predicted_spectral = []
            plot_p1(ymin, ymax, inputs, sequence_label, have_spectral,
                    predicted_spectral, predicted_ar,
                    sys, p_p)

            if not have_spectral:
                error_spec, error_persist = [], []
            plot_p2(have_spectral, error_spec, error_persist, error_ar, lab, p_p)

    error_ar_mean = np.mean(error_ar_data, 0)
    error_ar_std = np.std(error_ar_data, 0)
    if have_spectral:
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = \
            error_stat(error_spec_data, error_persist_data)
    else:
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = [], [], [], []

    plot_p3(ymin, ymax, have_spectral, error_spec_mean, error_spec_std,
            error_persist_mean, error_persist_std,
            error_ar_mean, error_ar_std,
            t_t, p_p)


    p_p.close()
    print("See the output in " + filename)



def test_identification2(t_t = 100, no_runs = 10, s_choices = [15,3,1],
                        have_kalman = False, have_spectral = True,
                        G = np.matrix([[0.999,0],[0,0.5]]),
                        f_dash = np.matrix([[1,1]]), sequence_label = ""):
    '''
    from experiments.py
    '''
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
    proc_noise_std = 0.5
    obs_noise_std  = 0.5

    error_spec_data = None
    error_persist_data = None
    error_AR1_data = None
    error_Kalman_data = None

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
        n, m, W, V, matrix_c, R, Q, matrix_a, Z = pre_comp_filter_params(G, f_dash, proc_noise_std,\
             obs_noise_std, t_t)

        #PREDICTION
        plt.plot(Y, label='Output', color='#000000', linewidth=2, antialiased = True)

        for s in s_choices:
            Y_pred = prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y)

            #print(np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))]))

            #print(lab)
            if s == 1:
                if error_AR1_data is None:
                    error_AR1_data = np.array([pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]),\
                         2) for i in range(len(Y))])
                else:
                    #print(error_AR1_data.shape)
                    error_AR1_data = np.vstack((error_AR1_data,\
                         [pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]), 2) for i in range(len(Y))]))
            if s == t_t:
                # For the spectral filtering etc, we use:
                # loss = pow(np.linalg.norm(sys.outputs[t] - y_pred), 2)
                if error_Kalman_data is None:
                    error_Kalman_data = np.array([pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]),\
                         2) for i in range(len(Y))])
                else:
                    error_Kalman_data = np.vstack((error_Kalman_data,\
                         [pow(np.linalg.norm(Y_pred[i][0,0] - Y[i]), 2) for i in range(len(Y))]))
                plt.plot([i[0,0] for i in Y_pred], label="Kalman" + sequence_label,\
                     color=(42.0/255.0, 204.0 / 255.0, 200.0/255.0),\
                          linewidth=2, antialiased = True)
            else:
                plt.plot([i[0,0] for i in Y_pred], label='AR(%i)' % (s+1)  + sequence_label,\
                     color=(42.0/255.0, 204.0 / 255.0, float(min(255.0,s))/255.0),\
                          linewidth=2, antialiased = True)

            plt.xlabel('Time')
            plt.ylabel('Prediction')


        if have_spectral:
            #using class WaveFilteringSisoFtl instead fubction WaveFilteringSisoFtl
            #predicted_output, M, error_spec, error_persist = WaveFilteringSisoFtl(sys, t_t, 5)
            wf_siso_ftl = WaveFilteringSisoFtl(sys, t_t, 5, VERBOSE)
            predicted_output, M, error_spec, error_persist = \
                wf_siso_ftl.y_pred_full, wf_siso_ftl.M,\
                     wf_siso_ftl.pred_error, wf_siso_ftl.pred_error_persistent

            plt.plot(predicted_output, label='Spectral' + sequence_label,\
                 color='#1B2ACC', linewidth=2, antialiased = True)
            if error_spec_data is None: error_spec_data = error_spec
            else: error_spec_data = np.vstack((error_spec_data, error_spec))
            if error_persist_data is None: error_persist_data = error_persist
            else: error_persist_data = np.vstack((error_persist_data, error_persist))

        plt.legend()
        plt.savefig(p_p, format='pdf')
        plt.close('all')
        #plt.show()

    if have_spectral:
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = \
            error_stat(error_spec_data, error_persist_data)
    else:
        error_spec_mean, error_spec_std, error_persist_mean, error_persist_std = [], [], [], []


    error_AR1_mean = np.mean(error_AR1_data, 0)
    error_AR1_std = np.std(error_AR1_data, 0)
    if have_kalman:
        error_Kalman_mean = np.mean(error_Kalman_data, 0)
        error_Kalman_std = np.std(error_Kalman_data, 0)
    else:
        error_Kalman_mean, error_Kalman_std = [], []

    if error_spec is None: error_spec = []
    if error_persist is None: error_persist = []
    for (ylim, alphaValue) in [((0, 100.0), 0.2), ((0.0, 1.0), 0.05)]:
        for Tlim in [t_t-1, min(t_t-1, 20)]:

            p3_for_test_identification2(ylim, have_spectral, Tlim, error_spec, sequence_label,
                               error_spec_mean, error_spec_std, alphaValue,
                               error_persist, error_persist_mean, error_persist_std,
                               error_AR1_mean, error_AR1_std,
                               have_kalman, error_Kalman_mean, error_Kalman_std, p_p)


    p_p.close()


# This is taken from pyplot documentation
def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    '''
    from experiments.py

    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
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
    '''

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
    '''
    from experiments.py 
    '''
    
    filename = './outputs/noise.pdf'
    p_p = PdfPages(filename)

    for s in [1, 2, 3, 7]:
        data = np.zeros((discretisation, discretisation))
        diff = np.zeros((discretisation, discretisation))
        ratio = np.zeros((discretisation, discretisation))
        errKalman = np.zeros((discretisation, discretisation))
        errAR = np.zeros((discretisation, discretisation))
        ################# SYSTEM ###################
        G = np.matrix([[0.999,0],[0,0.5]])
        f_dash = np.matrix([[1,1]])
        for proc_noise_i in range(discretisation):
            proc_noise_std = float(proc_noise_i + 1) / (discretisation - 1)
            for obs_noise_i in range(discretisation):
                obs_noise_std  = float(obs_noise_i + 1) / (discretisation - 1)

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
                    n, m, W, V, matrix_c, R, Q, matrix_a, Z = pre_comp_filter_params(G, f_dash, proc_noise_std, obs_noise_std, t_t)

                    #PREDICTION
                    Y_pred = prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y)
                    Y_kalman = prediction_kalman(t_t, f_dash, G, matrix_a, sys, Z, Y)


                    data[proc_noise_i][obs_noise_i] += np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))])
                    diffHere = np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))])
                    #print(Y_kalman[0][0,0])
                    diffHere -= np.linalg.norm([Y_kalman[i][0,0] - Y[i] for i in range(min(len(Y),len(Y_kalman)))])
                    #print(diffHere)
                    diff[proc_noise_i][obs_noise_i] += diffHere
                    #print(len(Y))
                    #print(len(Y_kalman))
                    errKalman[proc_noise_i][obs_noise_i] += pow(np.linalg.norm([Y_kalman[i][0,0] - Y[i] for i in range(min(len(Y),len(Y_kalman)))]), 2)
                    errAR[proc_noise_i][obs_noise_i] += pow(np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(len(Y))]), 2)

        data = data / no_runs
        fig, ax = plt.subplots()
        tickLabels = [str(float(i+1) / 10) for i in range(11)]
        im, cbar = heatmap(data, tickLabels, tickLabels, ax=ax, cmap="YlGn", cbarlabel="Avg. RMSE of AR(%i), %s runs" % (s+1, no_runs))
        plt.ylabel('Variance of process noise')
        plt.xlabel('Variance of observation noise')
        fig.tight_layout()
        plt.savefig(p_p, format='pdf')
        #plt.show()

        diff = diff / no_runs
        fig, ax = plt.subplots()
        tickLabels = [str(float(i+1) / 10) for i in range(11)]
        im, cbar = heatmap(diff, tickLabels, tickLabels, ax=ax, cmap="YlOrRd", cbarlabel="Avg. diff. in RMSEs of AR(%i) and Kalman filter, %s runs" % (s+1, no_runs))
        plt.ylabel('Variance of process noise')
        plt.xlabel('Variance of observation noise')
        fig.tight_layout()
        plt.savefig(p_p, format='pdf')
        #plt.show()

        ratio = pow(errKalman / errAR, 2)
        fig, ax = plt.subplots()
        tickLabels = [str(float(i+1) / 10) for i in range(11)]
        im, cbar = heatmap(ratio, tickLabels, tickLabels, ax=ax, cmap="PuBu", cbarlabel="Ratios of agg. errors of Kalman and AR(%i), %s runs" % (s+1, no_runs))
        plt.ylabel('Variance of process noise')
        plt.xlabel('Variance of observation noise')
        fig.tight_layout()
        plt.savefig(p_p, format='pdf')

    p_p.close()


def testImpactOfS(t_t = 200, no_runs = 100, sMax = 15):
    '''
    from experiments.py
    '''

    if sMax > t_t:
        print("The number of s to test must be less than the horizon t_t.")
        exit()

    filename = './outputs/impacts.pdf'
    p_p = PdfPages(filename)

    for (proc_noise_std, obs_noise_std, linestyle) in [ (0.1, 0.1, "dotted"), (0.1, 1.0, "dashdot"),  (1.0, 0.1, "dashed"), (1.0, 1.0, "solid") ]:
        errAR = np.zeros((sMax+1, no_runs))
        ################# SYSTEM ###################
        G = np.matrix([[0.999,0],[0,0.5]])
        f_dash = np.matrix([[1,1]])
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
                n, m, W, V, matrix_c, R, Q, matrix_a, Z = pre_comp_filter_params(G, f_dash, proc_noise_std, obs_noise_std, t_t)

                #PREDICTION
                Y_pred = prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y)


                errAR[s][runNo] = pow(np.linalg.norm([Y_pred[i][0,0] - Y[i] for i in range(min(len(Y), len(Y_pred)))]), 2) / t_t


        error_AR1_mean = np.mean(errAR, 1)
        error_AR1_std = np.std(errAR, 1)
        print(len(error_AR1_mean))
        alphaValue = 0.2
        cAR1 = (proc_noise_std, obs_noise_std, 1.0/255)
        #plt.plot(range(1, sMax), error_AR1_mean[1:], label='AR(2)', color=cAR1, linewidth=2, antialiased = True)
        #plt.fill_between(range(1, sMax), (error_AR1_mean-error_AR1_std)[1:], (error_AR1_mean+error_AR1_std)[1:], alpha=alphaValue, edgecolor=cAR1, linewidth=2, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue,
        lab = "W = %.2f, V = %.2f" % (proc_noise_std, obs_noise_std)
        plt.plot(range(sMax+1)[1:-1], error_AR1_mean[1:-1], color=cAR1, linewidth=2, antialiased = True, label = lab, linestyle= linestyle)
        plt.fill_between(range(sMax+1)[1:-1], (error_AR1_mean-error_AR1_std)[1:-1], (error_AR1_mean+error_AR1_std)[1:-1], alpha=alphaValue, facecolor = cAR1, edgecolor=cAR1, linewidth=2, antialiased=True) #transform=trans) #offset_position="data") alpha=alphaValue,
        plt.xlabel('Number s of auto-regressive terms, past the first one')
        plt.ylabel('Avg. error of AR(s), %i runs' % no_runs )
        plt.ylim(0, 1.5)
        plt.legend()
        plt.savefig(p_p, format='pdf')

    p_p.close()


def testSeqD0(no_runs = 100):
    '''
    from experiments.py
    '''

    plain = False
    lr = True

    matlabfile_in = './OARIMA_code_data/data/setting6.mat'

    if plain:
        varname_in = "seq_d0"
        ts = TimeSeries(matlabfile = matlabfile_in, varname="seq_d0")
        t_t = len(ts.outputs)
        test_identification(ts, "%s-complete"%varname_in, no_runs, t_t, 5, sequence_label = varname_in, have_spectral = False)
        t_t = min(20000, len(ts.outputs))
        test_identification(ts, "%s-20000"%varname_in, no_runs, t_t, 5, sequence_label = varname_in, have_spectral = False)
        t_t = min(2000, len(ts.outputs))
        test_identification(ts, "%s-2000"%varname_in, no_runs, t_t, 5, sequence_label = varname_in, have_spectral = False)
        t_t = min(200, len(ts.outputs))
        test_identification(ts, "%s-200"%varname_in, no_runs, t_t, 5, sequence_label = varname_in, have_spectral = False)
        t_t = min(100, len(ts.outputs))
        test_identification(ts, "%s-short-k5"%varname_in, 1, t_t, 5, sequence_label = varname_in)
        #test_identification(ts, "seq0-short-k50", 1, t_t, 50, 27, 37, sequence_label = "seq_d0")
        #test_identification(ts, "seq0-short-k5", 1, t_t, 5, sequence_label = "seq_d0")
        #test_identification(ts, "seq0-short-k50", 1, t_t, 50, sequence_label = "seq_d0")
    if lr:
        varname_in = "lr_d0"
        ts = TimeSeries(matlabfile = matlabfile_in, varname="seq_d0")
        ts.logratio()
        t_t = len(ts.outputs) # has to go after the log-ratio truncation by one
        test_identification(ts, "logratio-complete", no_runs, t_t, 5, sequence_label = varname_in, have_spectral = False)
        t_t = min(20000, len(ts.outputs))
        test_identification(ts, "logratio-20000", no_runs, t_t, 5,  sequence_label = varname_in, have_spectral = False)
        t_t = min(2000, len(ts.outputs))
        test_identification(ts, "logratio-2000", no_runs, t_t, 5, sequence_label = varname_in, have_spectral = False)
        t_t = min(200, len(ts.outputs))
        test_identification(ts, "logratio-200", no_runs, t_t, 5, sequence_label = varname_in, have_spectral = False)
        t_t = min(100, len(ts.outputs))
        test_identification(ts, "logratio-short-k5", no_runs, t_t, 5, sequence_label = varname_in)

def test_AR():
    '''
    from experiments.py
    '''
    matlabfile_in = './OARIMA_code_data/data/setting6.mat'
    varname_in = "seq_d0"

    ts = TimeSeries(matlabfile = matlabfile_in, varname=varname_in)
    t_t = min(100, len(ts.outputs))
    s=10
    matrix_d=10.
    theta = [0 for i in range(s)]

    for t in range(s,t_t):
        eta = pow(float(t),-0.5)

        Y = ts.outputs[t]

        loss = cost_ar(theta, Y, list(reversed(ts.outputs[t-s:t])))
        grad = gradient_ar(theta, Y, list(reversed(ts.outputs[t-s:t])))

        print("Loss: at time step %d :" % (t), loss)
        theta = [theta[i] -eta*grad[i] for i in range(len(theta))] #gradient step
        norm_theta = np.linalg.norm(theta)

        if norm_theta>matrix_d: theta = [matrix_d*i/norm_theta for i in theta] #projection step

########################  inputlds ################



############# onlinelds ##############

# def do_filter_step(G,F,V,W, Id, Y_curr, m_prev,c_prev):


def cost_ar(theta, *args):
    '''
    from onlinelds.py

    theta: s parameters
    args[0]: observation at time t
    args[1]: past s observations (most most to least recent: t-1 to t-1-s)
    '''

    return pow(float(args[0]) - np.dot(args[1], theta), 2)


def gradient_ar(theta, *args):
    '''
     from onlinelds.py

    theta: s parameters
    args[0]: observation
    args[1]: past s observations
    '''

    g = [(float(args[0]) - np.dot(args[1], theta)) * i for i in args[1]]

    return np.squeeze(-2 * np.array(g).reshape(-1, 1))


###### MATLAB functions ##########


def test_arima_ogd(i, mk, lrate, data):
    '''
    to test arima_ogd function
    the test casees are based on MATLAB:
    the test numbers were taken from the output of MATLAB function
    the random array w is fixed
    :param i:
    :param mk:
    :param lrate:
    :param data:
    :return:
    '''
    # the random array w is fixed here
    w = np.array([[0.276926, 0.023466, 0.480833, 0.507039, 0.710869, 0.188331, 0.374130, 0.290949, 0.724284, 0.562128]])

    data_i_test = 0.0685
    diff_test = 0.0975 #out from MATLAB function
    w_test = np.array([[0.39243, 0.17813, 0.59069, 0.52301, 0.60476, 0.10548, 0.37286, 0.29994, 0.72463, 0.49051]]) #out from MATLAB function

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
    '''
    ARIMA Online Newton Step algorithm
    :param data:
    :param options:
    :return:
    '''
    # MATLAB:
    # mk = options.mk;
    # lrate = options.lrate;
    # w = options.init_w;
    mk = options.mk
    lrate = options.lrate
    w = options.init_w

    # MATLAB:
    # list = [];
    # SE = 0;
    list = np.array([])
    SE = 0

    # MATLAB:
    # for i = mk+1:size(data,2)
    for i in range(mk, len(data)):

        #MATLAB:
        # diff = w*data(i-mk:i-1)'-data(i);%'
        #w = w - data(i-mk:i-1)*2*diff/sqrt(i-mk)*lrate;
        diff = diff_calc(w, data, mk, i)
        w = w_calc(w, data, mk, i, diff, lrate)

        # MATLAB:
        #SE = SE + diff ^ 2;
        SE = SE + diff**2

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
    '''
    MATLAB: diff = w*data(i-mk:i-1)'-data(i);
    remember! MATLAB_data(1) == Python_data[0]
    we have to convert data[] from 1D vector to a numpy matrix (2D) to apply the transpose
    OR data[].reshape(-1,1) can be also used to mimick the transpose

    :param w:
    :param data:
    :param mk:
    :param i:
    :return:
    '''

    return np.asscalar(np.dot(w, np.matrix(data[i - mk:i]).T) - data[i])


def w_calc(w, data, mk, i, diff, lrate):
    '''
    MATLAB: w = w - data(i-mk:i-1)*2*diff/sqrt(i-mk)*lrate;
    :param w:
    :param data:
    :param mk:
    :param i:
    :param diff:
    :param lrate:
    :return:
    '''
    return w - data[i - mk:i] * 2 * diff / np.sqrt(i - mk + 1) * lrate

def grad_calc(data, i, mk, diff):
    '''
    MATLAB: grad = 2*data(i-mk:i-1)*diff;
    :param data:
    :param i:
    :param mk:
    :param diff:
    :return:
    '''
    return 2 * data[i - mk:i] * diff

def A_trans_calc(A_trans, grad):
    '''
    MATLAB:
    A_trans = A_trans - A_trans * grad' * grad * A_trans/(1 + grad * A_trans * grad');
    we have to convert data[] from 1D vector to a numpy matrix (2D) to apply the transpose
    OR data[].reshape(-1,1) can be also used to mimick the transpose
    :return:
    '''
    #@ is matrix multiply symbol
    A_trans = A_trans - A_trans @ np.matrix(grad).T @ np.matrix(grad) @ A_trans / (1 + grad @ A_trans @ grad.T)

    return A_trans

def w_calc_arima_ons(w, lrate, grad, A_trans):
    '''
    MATLAB: w = w - lrate * grad * A_trans ;
    :param w:
    :param lrate:
    :param grad:
    :param A_trans:
    :return:
    '''
    w = w - lrate * grad @ A_trans
    return w

def arima_ons(data, options):
    '''
    ARIMA Online Newton Step algorithm
    :param data:
    :param options:
    :return:
    '''

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
    for i in range(mk, len(data)):

        #MATLAB: diff = w*data(i-mk:i-1)'-data(i);
        diff = diff_calc(w, data, mk, i)

        #MATLAB: grad = 2*data(i-mk:i-1)*diff;
        grad = grad_calc(data, i, mk, diff)

        # MATLAB: A_trans = A_trans - A_trans * grad' * grad * A_trans/(1 + grad * A_trans * grad');
        A_trans = A_trans_calc(A_trans, grad)

        # MATLAB: w = w - lrate * grad * A_trans ;
        w = w_calc_arima_ons(w, lrate, grad, A_trans)

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
    # lab1 = 'AR(3) / OGD, c_0 = ' + str(eta_zero)
    lab1 = "AR(" + str(s) + "), c = " + str(int(eta_zero))
    return lab1


def plot_p1(ymin, ymax, inputs, sequence_label, have_spectral,
            predicted_spectral, predicted_ar,
            sys, p_p):
    '''

    :param ymin:
    :param ymax:
    :param inputs:
    :param sequence_label:
    :param have_spectral:
    :param predicted_spectral:
    :param eta_zero:
    :param predicted_ar:
    :param s:
    :param sys:
    :param p_p:
    :return:
    '''

    p1 = plt.figure()
    if ymax and ymin: plt.ylim(ymin, ymax)
    if sum(inputs[1:]) > 0: plt.plot(inputs[1:], label='Input')
    if sequence_label:
        plt.plot([float(i) for i in sys.outputs][1:], label=sequence_label, color='#000000', linewidth=2,
                 antialiased=True)
    else:
        plt.plot([float(i) for i in sys.outputs][1:], label='Output', color='#000000', linewidth=2, antialiased=True)
    # plt.plot([-i for i in predicted_output], label='Predicted output') #for some reason, usual way produces -ve estimate
    if have_spectral:
        plt.plot([i for i in predicted_spectral], label='Spectral')

    plt.plot(predicted_ar, label=lab)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Output')
    p1.show()
    p1.savefig(p_p, format='pdf')


def plot_p2(have_spectral, error_spec, error_persist, error_ar, lab, p_p):
    '''

    :return:
    '''
    p2 = plt.figure()
    plt.ylim(0, 20)
    if have_spectral:
        plt.plot(error_spec, label='Spectral')
        plt.plot(error_persist, label='Persistence')
    plt.plot(error_ar, label=lab)
    plt.legend()
    p2.show()
    plt.xlabel('Time')
    plt.ylabel('Error')
    p2.savefig(p_p, format='pdf')


def plot_p3(ymin, ymax, have_spectral, error_spec_mean, error_spec_std,
            error_persist_mean, error_persist_std,
            error_ar_mean, error_ar_std,
            t_t, p_p):
    p3 = plt.figure()
    if ymax and ymin: plt.ylim(ymin, ymax)
    if have_spectral:
        plt.plot(error_spec_mean, label='Spectral', color='#1B2ACC', linewidth=2, antialiased=True)
        plt.fill_between(range(0, t_t - 1), error_spec_mean - error_spec_std, error_spec_mean + error_spec_std, alpha=0.2,
                         edgecolor='#1B2ACC', facecolor='#089FFF',
                         linewidth=1, antialiased=True)
        plt.plot(error_persist_mean, label='Persistence', color='#CC1B2A', linewidth=2, antialiased=True)
        plt.fill_between(range(0, t_t - 1), error_persist_mean - error_persist_std,
                         error_persist_mean + error_persist_std, alpha=0.2, edgecolor='#CC1B2A', facecolor='#FF0800',
                         linewidth=1, antialiased=True)

    cAR1 = (42.0 / 255, 204.0 / 255.0, 1.0 / 255)
    bAR1 = (1.0, 204.0 / 255.0, 0.0)  # , alphaValue
    plt.ylim(0, 20)
    plt.plot(error_ar_mean, label='AR(3)', color=cAR1, linewidth=2, antialiased=True)
    plt.fill_between(range(0, t_t), error_ar_mean - error_ar_std, error_ar_mean + error_ar_std, alpha=0.2, edgecolor=cAR1,
                     facecolor=bAR1,
                     linewidth=1, antialiased=True)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')
    p3.savefig(p_p, format='pdf')


def error_stat(error_spec_data, error_persist_data):
    '''
    if have_spectral:
    :return:
    '''

    error_spec_mean = np.mean(error_spec_data, 0)
    error_spec_std = np.std(error_spec_data, 0)
    error_persist_mean = np.mean(error_persist_data, 0)
    error_persist_std = np.std(error_persist_data, 0)

    return error_spec_mean, error_spec_std, error_persist_mean, error_persist_std

def pre_comp_filter_params(G, f_dash, proc_noise_std, obs_noise_std, t_t):
    n = G.shape[0]
    m = f_dash.shape[0]

    W = proc_noise_std ** 2 * np.matrix(np.eye(n))
    V = obs_noise_std ** 2 * np.matrix(np.eye(m))

    # m_t = [np.matrix([[0],[0]])]
    matrix_c = [np.matrix(np.eye(2))]
    R = []
    Q = []
    matrix_a = []
    Z = []

    for t in range(t_t):
        R.append(G * matrix_c[-1] * G.transpose() + W)
        Q.append(f_dash * R[-1] * f_dash.transpose() + V)
        matrix_a.append(R[-1] * f_dash.transpose() * np.linalg.inv(Q[-1]))
        matrix_c.append(R[-1] - matrix_a[-1] * Q[-1] * matrix_a[-1].transpose())
        Z.append(G * (np.eye(2) - matrix_a[-1] * f_dash))

    return n, m, W, V, matrix_c, R, Q, matrix_a, Z

def p3_for_test_identification2(ylim, have_spectral, Tlim, error_spec, sequence_label,
                               error_spec_mean, error_spec_std, alphaValue,
                               error_persist, error_persist_mean, error_persist_std,
                               error_AR1_mean, error_AR1_std,
                               have_kalman, error_Kalman_mean, error_Kalman_std, pp):

    # p3 = plt.figure()
    p3, ax = plt.subplots()
    plt.ylim(ylim)
    if have_spectral:
        plt.plot(range(0, Tlim), error_spec[:Tlim], label='Spectral' + sequence_label, color='#1B2ACC', linewidth=2,
                 antialiased=True)
        plt.fill_between(range(0, Tlim), (error_spec_mean - error_spec_std)[:Tlim],
                         (error_spec_mean + error_spec_std)[:Tlim], alpha=alphaValue, edgecolor='#1B2ACC',
                         facecolor='#089FFF', linewidth=1, antialiased=True)
        plt.plot(range(0, Tlim), error_persist[:Tlim], label='Persistence' + sequence_label, color='#CC1B2A',
                 linewidth=2, antialiased=True)
        plt.fill_between(range(0, Tlim), (error_persist_mean - error_persist_std)[:Tlim],
                         (error_persist_mean + error_persist_std)[:Tlim], alpha=alphaValue, edgecolor='#CC1B2A',
                         facecolor='#FF0800', linewidth=1, antialiased=True)

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
    plt.plot(error_AR1_mean[:Tlim], label='AR(2)' + sequence_label, color=cAR1, linewidth=2, antialiased=True)
    plt.fill_between(range(0, Tlim), (error_AR1_mean - error_AR1_std)[:Tlim], (error_AR1_mean + error_AR1_std)[:Tlim],
                     alpha=alphaValue, edgecolor=cAR1, facecolor=bAR1, linewidth=1,
                     antialiased=True)  # transform=trans) #offset_position="data") alpha=alphaValue,

    if have_kalman:
        cK = (42.0 / 255.0, 204.0 / 255.0, 200.0 / 255.0)
        bK = (1.0, 204.0 / 255.0, 200.0 / 255.0)  # alphaValue
        print(cK)
        print(bK)
        plt.plot(error_Kalman_mean[:Tlim], label='Kalman' + sequence_label, color=cK, linewidth=2, antialiased=True)
        plt.fill_between(range(0, Tlim), (error_Kalman_mean - error_Kalman_std)[:Tlim],
                         (error_Kalman_mean + error_Kalman_std)[:Tlim], alpha=alphaValue, facecolor=bK, edgecolor=cK,
                         linewidth=1, antialiased=True)  # transform = trans) #offset_position="data")

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Error')
    # p3.show()
    p3.savefig(p_p, format='pdf')


def prediction(t_t, f_dash, G, matrix_a, sys, s, Z, Y):

    Y_pred = []
    for t in range(t_t):
        Y_pred_term1 = f_dash * G * matrix_a[t] * sys.outputs[t]
        if t == 0:
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
        Y_pred.append(Y_pred_term1 + f_dash * acc)

    return Y_pred


def prediction_kalman(t_t, f_dash, G, matrix_a, sys, Z, Y):
    Y_kalman = []
    for t in range(t_t):
        Y_pred_term1 = f_dash * G * matrix_a[t] * sys.outputs[t]
        if t == 0:
            Y_kalman.append(Y_pred_term1)
            continue

        accKalman = 0
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




