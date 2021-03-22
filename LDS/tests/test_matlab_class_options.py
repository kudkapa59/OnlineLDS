import unittest
from LDS.matlab_options.matlab_class_options import ClassOptions


class TestClassOptions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    def setUp(self):
        self.options = ClassOptions()

    def tearDown(self):
        print('tearDown\n')

    def test(self):
        self.assertEqual(self.options.mk, None)
    

if __name__ == '__main__':
    unittest.main()