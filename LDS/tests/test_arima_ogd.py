import unittest
import numpy as np
from OnlineLDS_library import arima_ogd
from OnlineLDS_library import diff_calc
from OnlineLDS_library import w_calc

class TestArimaOGD(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        i = 10
        mk = 10
        lrate = 1
        data = [-0.5924506,-0.79333124 -0.56350708 ... -0.0713888  -0.05520875
        -0.18946305]

        # the random array w is fixed here
        w = np.array([[0.276926, 0.023466, 0.480833, 0.507039, 0.710869, 0.188331, 0.374130,\
            0.290949, 0.724284, 0.562128]])

        data_i_test = 0.0685
        diff_test = 0.0975 #out from MATLAB function
        w_test = np.array([[0.39243, 0.17813, 0.59069, 0.52301, 0.60476, 0.10548, 0.37286,\
            0.29994, 0.72463, 0.49051]]) #out from MATLAB function        

        diff = diff_calc(w, data, mk, i)
        wi = w_calc(w, data, mk, i, diff, lrate)


    def tearDown(self):
        print('tearDown\n')
    
    def test_data_i(self):
        print('arima_ogd test:')
        if np.round(data[i], 4) == data_i_test:
            print('test data[i] - OK')
        else:
            print('ERROR: arima_ogd - data[i]')

    def test_check_input(self):
        return_check = self.dyn.check_input(np.zeros((2,1)))
        self.assertEqual(return_check,None)
    
    def test_solve(self):
        self.dyn.solve([[1],[1]],np.zeros(100),100)
        self.assertEqual(self.dyn.n,1)

if __name__ == '__main__':
    unittest.main()