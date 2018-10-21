from unittest import TestCase
import pandas as pd
import numpy as np
import isajosep_util

class TestCheck_for_null(TestCase):



    def test_check_for_null(self):
        df = pd.DataFrame({'a': range(10)})
        df['a'][0] = np.nan
        df['a'][1] = np.inf
        df['a'][2] = np.inf

        self.assertEqual(isajosep_util.check_for_null(df)[0], 3)
        self.assertEqual(isajosep_util.check_for_null(df)[1], 2)
        self.assertEqual(isajosep_util.check_for_null(df)[2], 1)

        df = pd.DataFrame({'a': range(10)})

        self.assertEqual(isajosep_util.check_for_null(df)[:3], (0,0,0))


    def test_check_duplications(self):
        """
        Done
        :return:
        """
        df = pd.DataFrame({'a': range(10)})

        df['a'][0] = 1
        df['a'][1] = 1

        # two rows duplicated by non_index

        self.assertEqual(isajosep_util.check_for_null(df), (0, 0, 0, 1,0,0))








    def test_double_sample(self):

        df = pd.DataFrame({'a': range(10), 'b':range(10), 'c':range(10)})

        result = isajosep_util.double_sample_df(input_df=df, row_dims=2)


        self.assertEquals(result.shape, (2,2))

        self.assertEquals(isajosep_util.double_sample_df(input_df=df, row_dims=2, col_dims=1).shape, (2,1))


        with self.assertRaises(ValueError):
            isajosep_util.double_sample_df(input_df=df, row_dims=2, col_dims=4)
            isajosep_util.double_sample_df(input_df=df, row_dims=5)










