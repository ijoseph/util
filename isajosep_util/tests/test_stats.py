import unittest
import isajosep_util

class TestStats(unittest.TestCase):

    def setUp(self):
        self.x1 = [1, 2]
        self.x2 = [1.5, 2, 2.5]


    def test_t_test(self):
        isajosep_util.t_test_with_power_comp(x1=self.x1, x2=self.x2)
        isajosep_util.t_test_with_power_comp(x1=self.x1, x2=self.x2, power=0.95)


