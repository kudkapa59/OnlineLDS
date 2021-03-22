import unittest
import numpy as np
from LDS.ds.dynamical_system import DynamicalSystem

class TestDynamicalSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        self.dyn = DynamicalSystem(np.matrix([[0.999,0],[0,0.5]]),np.zeros((2,1)),np.matrix([[1,1]]),np.zeros((1,1)),
                               process_noise='gaussian',
                               observation_noise='gaussian',
                               process_noise_std=0.5,
                               observation_noise_std=0.5,
                               timevarying_multiplier_b = None)

    def tearDown(self):
        print('tearDown\n')

    def test_check_input(self):
        return_check = self.dyn.check_input(np.zeros((2,1)))
        self.assertEqual(return_check,None)
    
    def test_solve(self):
        self.dyn.solve([[1],[1]],np.zeros(100),100)
        self.assertEqual(self.dyn.n,1)

if __name__ == '__main__':
    unittest.main()