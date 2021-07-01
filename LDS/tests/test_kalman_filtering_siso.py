import unittest
import numpy as np
from LDS.filters.kalman_filtering_siso import KalmanFilteringSISO
from LDS.ds.dynamical_system import DynamicalSystem




'''def test_identification2(t_t = 100, no_runs = 10, s_choices = [15,3,1],
                        have_kalman = False, have_spectral = True,
                        G = np.matrix([[0.999,0],[0,0.5]]),
                        f_dash = np.matrix([[1,1]]), sequence_label = ""):'''
#Need to use these parameters to check if this Kalman works the same way.

sys = DynamicalSystem(G = np.matrix([[0.999,0],[0,0.5]]),np.zeros((2,1)),
                               f_dash,np.zeros((1,1)),
                               process_noise='gaussian',
                               observation_noise='gaussian',
                               process_noise_std=proc_noise_std,
                               observation_noise_std=obs_noise_std,
                               timevarying_multiplier_b = None)
sys.solve([[1],[1]],np.zeros(100),100)

class TestKalmanFilteringSISO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        self.sys = sys
        self.kalman = KalmanFilteringSISO(self.sys,100)

    def tearDown(self):
        print('tearDown\n')

    def test_kalman(self):
        self.assertEqual(self.kalman.t_t,100)

    def test_predict(self):
        self.assertEqual(self.)

if __name__ == '__main__':
    unittest.main()
