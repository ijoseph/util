# -*- coding: utf-8 -*-



def labeled_barplot(data, x_label, y_label, hue=None, size_inches=(5,6), make_label=True, order=None):
    import seaborn as sns
    import matplotlib.pyplot as plt
    ax = sns.barplot(data=data, x=x_label, y=y_label, hue=hue, order=order)
    fig = plt.gcf()
    fig.set_size_inches(size_inches)
    plt.setp(ax.get_xticklabels(), rotation=90)
    if make_label:
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x()+p.get_width()/2.,
                    height,
                    '{:,.0f}'.format(height),
                    ha="center")


def check_for_null(input_df):
    """
    Checks for NaNs, Infs, and Duplication
    :param input_df:
    :return: ( total null, number of infs, number of nans, number of duplicated rows, number of duplicate rows by index, number of duplicate rows by both)
    """
    import pandas as pd

    num_inf = 0
    with pd.option_context('mode.use_inf_as_null', True):
        num_null_total = input_df.isnull().sum().sum()

    num_nan = input_df.isnull().sum().sum()

    num_duplicated_rows = input_df.duplicated().sum()
    num_duplicated_rows_by_index = input_df.index.duplicated().sum()
    num_duplicated_rows_by_both = input_df.reset_index().duplicated().sum()

    if num_inf or num_nan or num_duplicated_rows or num_duplicated_rows_by_index:
        print("✗ {:,} NaN or Inf values; {:,} inf, {:,} nan;"
              " Duplications: {:,} duplicated rows by exclusively non-index content, "
              "{:,} duplicated rows by exclusively index, "
              "{:,} duplicated by both"
              .format(num_null_total,
                      num_null_total - num_nan,
                      num_nan,
                      num_duplicated_rows,
                      num_duplicated_rows_by_index,
                      num_duplicated_rows_by_both))

        return(num_null_total, num_null_total - num_nan, num_nan, num_duplicated_rows,
               num_duplicated_rows_by_index, num_duplicated_rows_by_both)
    else:
        print("✓ No Nan or Inf values; no duplications.")
        return (0,0,0)



def csv_to_row_dict(input_csv_path, row_number):
    """
    :param input_csv_path: csv file path
    :param row_number: row number of csv file (1-based, not including the header row)
    :return:
    """
    import csv

    row_number_idx = row_number - 1

    assert row_number_idx >= 0, "must have positive row number; check indices (1-based index)"
    with open(input_csv_path) as input_csv_file_handle:
        reader = csv.DictReader(input_csv_file_handle)
        reader.fieldnames = [field.strip().lower() for field in reader.fieldnames]
        dict_row = list(reader)[row_number_idx]

    return dict_row


def assert_frame_equal_mod_sorting(dataframe_one, dataframe_two, ignore_index=False):
    import pandas.testing

    # Column ordering
    dataframe_one_mod =  dataframe_one[dataframe_two.columns.tolist()]

    if ignore_index:
        pandas.testing.assert_frame_equal(
            dataframe_one_mod.sort_values(dataframe_one_mod.columns.tolist()).reset_index(drop=True),
            dataframe_two.sort_values(dataframe_two.columns.tolist()).reset_index(drop=True))
    else:
        pandas.testing.assert_frame_equal(
            dataframe_one_mod.sort_values(dataframe_one_mod.columns.tolist()),
            dataframe_two.sort_values(dataframe_two.columns.tolist()))


def str2bool(v):
    import argparse
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def clean_variable_name(variable_strings):
    import re

    if isinstance(variable_strings, str):
        variable_strings = [variable_strings]
        list_output = False
    else:
        list_output = True


    out_list = list(map(lambda varStr: re.sub('\W|^(?=\d)', '_', varStr.lower()), variable_strings))

    if not list_output:
        return out_list[0]
    else:
        return(out_list)

def double_sample_df(input_df, row_dims, col_dims=None):
    if col_dims is None:
        col_dims_used = row_dims
    else:
        col_dims_used = col_dims


    return input_df.T.sample(col_dims_used).T.sample(row_dims)


def plot_pie_percent(pct, autopct="%.2f %%", **kwargs):

    if pct < 1:
        usd_pct = pct
    elif 1 < pct < 100:
        usd_pct =pct/100
    else:
        raise ValueError("Percent can't be > '100'; it's {pct}".format(pct=pct))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(3,3))
    plt.pie([usd_pct, 1-usd_pct], autopct=autopct, **kwargs)
    plt.show()


def t_test_with_power_comp(x1,x2, alternative='two-sided', alpha=0.05, power=0.8):
    """
    Independent (as contrasted with Paired) t-test with power calculation based on n_obs; effect size based
    on estimate from input.
    """
    import statsmodels.stats.power as smpwr
    import statsmodels.api as sm
    import numpy as np
    import matplotlib.pyplot as plt

    t_stat, pvalue, degrees_of_freedom = sm.stats.ttest_ind(x1=x1,x2=x2, alternative=alternative)

    print("T: {t_stat}, p-value: {pvalue}, degrees of freedom: {degrees_of_freedom},"
          " n_obs_1 = {n_obs_1}, n_obs_2 = {n_obs_2}"
          .format(t_stat=t_stat,
                  degrees_of_freedom=degrees_of_freedom,
                  pvalue=pvalue,
                  n_obs_1=len(x1),
                  n_obs_2=len(x2)))


    # Power calculation
    pooled_standard_dev_empirical = np.sqrt(
        np.mean(
            [np.std(x1), np.std(x2)]
        )
    )

    mean_diff_empirical = abs(np.mean(x1) - np.mean(x2))

    effect_size_empirical = mean_diff_empirical / pooled_standard_dev_empirical

    print("Empirical pooled stdev: {:.2f}".format(pooled_standard_dev_empirical))
    print(
        "Mean diff empirical: {:.2f}\neffect size empirical: {:.2f}".format(mean_diff_empirical, effect_size_empirical))


    # Empirical power needed

    nobs1 = smpwr.tt_ind_solve_power(effect_size=effect_size_empirical, nobs1=None, alpha=alpha, power=power,
                             alternative=alternative)

    print ("With alpha {alpha}, power {power}, need ≈ {nobs1:.0f} observations of each type to achieve significance"
           .format(alpha=alpha, power=power, nobs1=nobs1))


    # Power vs. nobs

    _ = smpwr.TTestIndPower().plot_power(dep_var='nobs', nobs=np.arange(2, 10), effect_size=[effect_size_empirical],
                                         alternative=alternative, alpha=alpha)

    plt.show()


def scaled_distplot(x, xlabel=None, **distplot_kwargs):
    """
    Distplot but scaled by arcsinh. Helpful for distributions that cross zero.
    :param x:
    :param xlabel:
    :param distplot_kwargs:
    :return:
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np

    ax1 = plt.figure().add_subplot(111)
    ax2 = ax1.twiny()
    sns.distplot(np.arcsinh(x), ax=ax1, **distplot_kwargs)

    ax1.set_xlabel("$\sinh^{-1}(\mathrm{" + "{xlabel}".format(xlabel=xlabel if xlabel is not None else 'x')
                   .replace(" ", "\ ") + "})$")

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticklabels(["{:,.0f}".format(v) for v in np.sinh(ax1.get_xticks())])
    ax2.set_xlabel("{xlabel}".format(xlabel=xlabel if xlabel is not None else 'x'))
    ax2.xaxis.set_tick_params(rotation=45)





























