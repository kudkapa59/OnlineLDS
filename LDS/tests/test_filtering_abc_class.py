import unittest
from unittest.mock import patch
from LDS.filters.filtering_abc_class import Filtering

class TestFiltering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    @patch.multiple(Filtering, __abstractmethods__=set())
    def test(self):
         self.instance = Filtering(1,2)
         self.assertEqual(self.instance.t_t,2)

if __name__ == '__main__':
    unittest.main()