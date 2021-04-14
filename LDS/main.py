# To run the main program like experiments.py
#the paths to input and output files are the same as in the original scripts (experiments.py, onlinelds.py, inputlds.py)
#so, to use this library the structute of a progect must be the same
###

from OnlineLDS_library import *

#input settings
#version = "FinalAAAI"
version = "Working"
#version = "Extended"

if __name__ == '__main__':
    try:
        close_all_figs()

        #!!! version == "Extended"
        if version == "Extended":
            # The following calls adds the plots for the extended version
            #Plots seq0, seq1, seq2, logratio pdfs.
            testSeqD0()
        if version == "FinalAAAI":
            '''These calls produce the AAAI 2019 figures (8-page version)'''
            
            '''
            Finds all the filters' errors and
            uses function p3_for_test_identification2 for plotting them.
            '''
            test_identification2(500, no_runs = 10, s_choices = [1],\
                have_kalman = True, have_spectral = True)
            '''Plots heatmap of process and observation noises' variance'''
            testNoiseImpact()
            '''Creates file './outputs/impacts.pdf', which stores
            plots of average error of auto-regression as a function of
            regression depth s.'''
            testImpactOfS()
        if version == "Working":
            # These calls produce illuminating plots, which did not make it into the final 
            # 8-page version of the paper.
            #None
            '''
            Finds all the filters' errors and
            uses function p3_for_test_identification2 for plotting them.Plots without Kalman.
            '''
            #Gives the error
            test_identification2(t_t = 100, no_runs = 10, have_spectral = True)
            '''Plots only auto-regression'''
            test_identification2(200, 10, have_spectral = False)
            #This function doesn't exist.
            timeSeqD0()
            #testSisoInvariantShort(100)
            '''Plots without Kalman'''
            test_identification2(100)
            #Plots seq0, seq1, seq2, logratio pdfs.
            testSeqD0()

            #These functions don't exist.
            timeSeqD0()
            testSeqD1()
            testSeqD2()
            #testSisoInvariantLong()
            #testSYSID()
            gradient_AR_test(0)

            #Implements Algorithm 1.
            test_AR()

            transition = np.matrix([[1.,-0.8],[-.6,.3]])
            observation = np.matrix([[1.0,1.0]])
            test_identification2(20, no_runs = 100, s_choices = [1], have_kalman = True,\
                have_spectral = True, G = transition, f_dash = observation)

    except (KeyboardInterrupt, SystemExit):
        raise
    except:
        print(" Error: ")
        print(traceback.format_exc())
