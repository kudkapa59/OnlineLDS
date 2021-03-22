import unittest
import numpy as np
from unittest.mock import patch
from LDS.h_m.hankel import Hankel

h = Hankel(2)

class TestHankel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        self.h = h

    def tearDown(self):
        print('tearDown\n')
    
    def test_hankel(self):
        np.testing.assert_allclose(np.linalg.eig(self.h.mat)[0][0],self.h.V[0],rtol=1e-5, atol=0)

if __name__ == '__main__':
    unittest.main()
