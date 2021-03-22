import unittest
from unittest.mock import patch
from LDS.filters.filtering_siso import FilteringSiso

class TestFilteringSiso(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print('setUpClass')
    
    @classmethod
    def tearDownClass(cls):
        print('tearDownClass')

    @patch.multiple(FilteringSiso, __abstractmethods__=set())
    def test(self):
        self.instance = FilteringSiso(1,2)
        self.assertEqual(self.instance.t_t,2)

if __name__ == '__main__':
    unittest.main()