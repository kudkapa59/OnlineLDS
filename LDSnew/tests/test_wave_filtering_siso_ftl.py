import unittest
import numpy as np
from AddClasses.ds.dynamical_system import DynamicalSystem
from AddClasses.filters.real_filters.wave_filtering_siso_ftl import WaveFilteringSisoFtl

sys = DynamicalSystem(np.matrix([[0.999,0],[0,0.5]]),np.zeros((2,1)),np.matrix([[1,1]]),np.zeros((1,1)),
                               process_noise='gaussian',
                               observation_noise='gaussian',
                               process_noise_std=0.5,
                               observation_noise_std=0.5,
                               timevarying_multiplier_b = None)
sys.solve([[1],[1]],np.zeros(100),100)

class TestWaveFilteringSisoFtl(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        self.sys = sys
        self.VERBOSE = True
        self.wave_filter = WaveFilteringSisoFtl(self.sys,100,5,self.VERBOSE)

    def tearDown(self):
        print('tearDown\n')

    def test(self):
        self.wave.var_calc()
        self.assertEqual(self.wave.k_dash,8)

if __name__ == '__main__':
    unittest.main()