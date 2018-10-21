from unittest import TestCase
import pandas as pd
import numpy as np
import isajosep_util.data_frame_plotter
import matplotlib.pyplot as plt

class test_data_frame_plotter(TestCase):

    def test_null(self):
        pd.DataFrame({'a':[0,1], 'nope':[None, None]})


    def setUp(self):
        np.random.seed(seed=42)
        self.test_data = pd.DataFrame({'vals': np.random.randint(10, size=(10,)),
                                       'vals_ii': np.random.randint(10, size=(10,)),
                                       'vals_float': np.random.random(10),
                                       'vals_float_ii': np.random.random(10),
                                       'indicator':
                                           [['one', 'two', 'three'][i] for i in  np.random.randint(3, size=(10,))]})


    def test_plot_distribution_divided_by_test_var(self):
        print(self.test_data)
        isajosep_util.data_frame_plotter.plot_distribution_divided_by_test_var(input_dataframe=self.test_data,
                                                                               test_var='indicator' ,
                                                                               size_inches=(10,6),
                                                                               kde_kws={'cumulative':True},
                                                                               rug=True)


        isajosep_util.data_frame_plotter.plot_distribution_divided_by_test_var(input_dataframe=self.test_data,
                                                                               test_var='indicator' ,
                                                                               size_inches=(10,6),
                                                                               rug=True)

        isajosep_util.data_frame_plotter.plot_distribution_divided_by_test_var(input_dataframe=self.test_data,
                                                                               test_var='indicator' ,
                                                                               size_inches=(10,6),
                                                                               rug=True,
                                                                               kde=False,
                                                                               hist_kws={'log':True})

        plt.show()



    def test_scatterlin(self):
        isajosep_util.data_frame_plotter.scatterlin(data=self.test_data,
                                                    test_var='indicator',
                                                    plot_cols='vals',
                                                    save_prefix="/tmp/saves")
        plt.show()

    def test_ordered_scatterlin(self):
        isajosep_util.data_frame_plotter.scatterlin(data=self.test_data,
                                                    test_var='indicator',
                                                    plot_cols='vals',
                                                    order=['one', 'two', 'three', 'four'])
        plt.show()




    def test_named_scatterlin(self):
        isajosep_util.data_frame_plotter.scatterlin(data=self.test_data,
                                                    test_var='indicator',
                                                    plot_cols='vals',
                                                    plot_col_names='CHOOON',
                                                    order=['one', 'two', 'three', 'four'])
        plt.show()


    def test_mpl_scatter(self):
        isajosep_util.data_frame_plotter.mpl_scatter_density_from_df(x=range(len(self.test_data['indicator'])),
                                                                     y=self.test_data['vals'])

        # test wtih symbolic
        with self.assertRaises(AssertionError):
            isajosep_util.data_frame_plotter.mpl_scatter_density_from_df(x='vals',
                                                                         y='vals_ii')

        ax, fig = isajosep_util.data_frame_plotter.mpl_scatter_density_from_df(x='vals_float', y='vals_float_ii',
                                                                     data=self.test_data, dpi=5, vmax=5)
        plt.show()

        import seaborn as sns
        sns.regplot(x='vals_float', y='vals_float_ii', data=self.test_data)
        plt.show()

    def test_plot_percent(self):
        isajosep_util.plot_pie_percent(0.5)
        isajosep_util.plot_pie_percent(0.5, autopct="%.10f")
        isajosep_util.plot_pie_percent(51)
        with self.assertRaises(ValueError):
            isajosep_util.plot_pie_percent(110)





