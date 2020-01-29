import unittest
import numpy as np
import isajosep_util
import matplotlib.pyplot as plt
import pandas as pd
from isajosep_util import labeled_barplot


class TestPlots(unittest.TestCase):

    def setUp(self):
        self.x_test_linear = np.linspace(-10, 10)
        self.x_test_log = np.logspace(-10, 10)
        self.test_data = pd.DataFrame(
            {
                'vals': [1, 2, 3],
                'indicator': ['one', 'two', 'three'],
                'cis': [(0.5, 1.5), (1.5, 2.1), (0, 2)],
            }
        )

    def test_scaled_distplot(self):
        isajosep_util.scaled_distplot(x=self.x_test_linear, xlabel="test")
        plt.show()
        isajosep_util.scaled_distplot(x=self.x_test_log, xlabel="test this")
        plt.show()

    def test_labeled_barplot(self):
        labeled_barplot(data=self.test_data, x_label='indicator', y_label='vals', ci_colname='cis')

        plt.show()
