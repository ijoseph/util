import unittest
import numpy as np
import isajosep_util
import matplotlib.pyplot as plt

class TestPlots(unittest.TestCase):

    def setUp(self):
        self.x_test_linear = np.linspace(-10, 10)
        self.x_test_log= np.logspace(-10, 10)

    def test_scaled_distplot(self):
        isajosep_util.scaled_distplot(x=self.x_test_linear, xlabel="test")
        plt.show()
        isajosep_util.scaled_distplot(x=self.x_test_log, xlabel="test this")
        plt.show()





