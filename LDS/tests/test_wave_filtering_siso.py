import unittest
import numpy as np
from LDS.ds.dynamical_system import DynamicalSystem
from LDS.filters.wave_filtering_siso import WaveFilteringSISO

sys = DynamicalSystem(np.matrix([[0.999,0],[0,0.5]]),np.zeros((2,1)),np.matrix([[1,1]]),np.zeros((1,1)),
                               process_noise='gaussian',
                               observation_noise='gaussian',
                               process_noise_std=0.5,
                               observation_noise_std=0.5,
                               timevarying_multiplier_b = None)
sys.solve([[1],[1]],np.zeros(100),100)

class TestWaveFilteringSISO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        self.sys = sys
        self.wave_filter = WaveFilteringSISO(self.sys,100,5,1,1)

    def tearDown(self):
        print('tearDown\n')

    def test(self):
        self.assertEqual(self.wave_filter.y_pred_full[0],[0.+0.j])

if __name__ == '__main__':
    unittest.main()