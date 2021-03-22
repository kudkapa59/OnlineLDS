import unittest
import math
import numpy as np
from AddClasses.ts.time_series import TimeSeries


class TestTimeSeries(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        self.ts = TimeSeries(matlabfile = '/root/LDS/LDS/LDSnew/OARIMA_code_data/data/setting6.mat',\
             varname="seq_d0")

    def tearDown(self):
        print('tearDown\n')

    def test_logratio(self):
        log = math.log(self.ts.outputs[0]/self.ts.outputs[1])
        self.ts.logratio()
        np.testing.assert_allclose(log,self.ts.outputs[0],rtol=1e-5, atol=0)
    

if __name__ == '__main__':
    unittest.main()