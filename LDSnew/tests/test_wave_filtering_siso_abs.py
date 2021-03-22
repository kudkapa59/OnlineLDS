import unittest
import numpy as np
from unittest.mock import patch
from AddClasses.ds.dynamical_system import DynamicalSystem
from AddClasses.filters.wave_filtering_siso_abs import WaveFilteringSisoAbs

class TestWaveFilteringSisoAbs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    @patch.multiple(WaveFilteringSisoAbs, __abstractmethods__=set())
    def test(self):
        sys = DynamicalSystem(np.matrix([[0.999,0],[0,0.5]]),np.zeros((2,1)),np.matrix([[1,1]]),np.zeros((1,1)),
                               process_noise='gaussian',
                               observation_noise='gaussian',
                               process_noise_std=0.5,
                               observation_noise_std=0.5,
                               timevarying_multiplier_b = None)
        sys.solve([[1],[1]],np.zeros(100),100)

        self.wave = WaveFilteringSisoAbs(sys,100,5)
        self.wave.var_calc()
        self.assertEqual(self.wave.k_dash,8)

if __name__ == '__main__':
    unittest.main()