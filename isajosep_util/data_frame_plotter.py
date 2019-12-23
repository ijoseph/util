# -*- coding: utf-8 -*-
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
import sys
import os
import astropy
import enum


def plot_distribution_divided_by_test_var(input_dataframe, test_var, kde_kws=None,
                                          hist_kws=None, size_inches=(60,40), plot_cols=None, **dist_kws):




    if plot_cols is not None:
        if not isinstance(plot_cols, list):
            plot_cols = [plot_cols]

        dataframe_to_plot = input_dataframe[plot_cols + [test_var]]
    else:
        dataframe_to_plot = input_dataframe



    test_var_possible_values = dataframe_to_plot[test_var].unique()

    numeric_features = dataframe_to_plot._get_numeric_data().select_dtypes(exclude=bool).columns

    figure, ax_array = _make_subplots(dataframe_to_plot, numeric_features, size_inches, test_var, test_var_possible_values,
                            kde_kws, hist_kws, **dist_kws)

    return(figure, ax_array)


def _make_subplots(input_dataframe, numeric_features, size_inches, test_var, test_var_possible_values, kde_kws,
                   hist_kws, **distkws):

    num_subplot_rows, num_subplot_cols = _get_num_subplots(len(numeric_features))

    figure, ax_array = plt.subplots(nrows=num_subplot_rows, ncols=num_subplot_cols, squeeze=False)

    figure.set_size_inches(size_inches)

    for (i, feature_name) in enumerate(numeric_features):

        sys.stdout.write("{0} ({1} of {2})...".format(feature_name, i+1, len(numeric_features)))

        for possible_value in test_var_possible_values:
            try:
                legend_string = _make_legend_string(feature_name, input_dataframe, possible_value, test_var)

                if not (len(input_dataframe[input_dataframe[test_var] == possible_value][feature_name]) > 1):
                    plt.axvline(x=input_dataframe[input_dataframe[test_var] == possible_value][feature_name][0],
                                c='black', linestyle='--')
                    plt.text(input_dataframe[input_dataframe[test_var] == possible_value][feature_name][0],
                             plt.ylim()[1]/2, possible_value, rotation=90, verticalalignment='center')
                    continue

                sns.distplot(input_dataframe[input_dataframe[test_var] == possible_value][feature_name],
                             kde_kws=kde_kws,
                             hist_kws=hist_kws,
                             ax=ax_array.ravel()[i],
                             label=legend_string,
                             **distkws)
            except ValueError as e:
                sys.stderr.write("Warning: {e}".format(e=e))
                pass

        ax_array.ravel()[i].legend()

    for ax in ax_array.ravel():
        ax.autoscale()



    return figure, ax_array


def _make_legend_string(feature_name, input_dataframe, possible_value, test_var):
    legend_string = "{name}: {number:,} " \
        .format(name=possible_value, number=(input_dataframe[test_var] == possible_value).sum())
    legend_string += fivenum((input_dataframe[input_dataframe[test_var] == possible_value])[feature_name])
    return legend_string


def _get_num_subplots(total_num_plots):
    """
    Roughly square configuration of subplots
    :param total_num_plots: 
    :return: 
    """
    rows = int(round(np.ceil(np.sqrt(total_num_plots))))
    columns = int(round(np.sqrt(total_num_plots)))
    return(rows,columns)


def fivenum(v):
    """Returns mean, std, plus Tukey's five number summary (minimum, lower-hinge, median, upper-hinge, maximum) f
        or the input vector, a list or array of numbers based on 1.5 times the interquartile distance"""
    try:
        v = v.values
    except:
        pass # v must already be a matrix

    try:
        np.sum(v)
    except TypeError:
        print('Error: you must provide a list or array of only numbers')

    q1 = scipy.stats.scoreatpercentile(v[~np.isnan(v)],25)
    q3 = scipy.stats.scoreatpercentile(v[~np.isnan(v)],75)
    iqd = q3-q1
    md = np.nanmedian(v)
    whisker = 1.5*iqd
    return(" mean {5:.1f}, σ_est: {6:.1f} fivenum: ({0:.0f}, {1:.1f}, median {2:.1f}, {3:.1f}, {4:.1f})"
           .format(np.nanmin(v), q1, md, q3, np.nanmax(v), np.nanmean(v), np.nanstd(v)))


def corrfunc(x, y, **kws):
    import scipy.stats
    r_spearman, _ = scipy.stats.spearmanr(x, y)
    r_pearson, _ = scipy.stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("R²:\nspearman={:.2f}\npearson={:.2f}".format(r_spearman**2, r_pearson**2),
                xy=(.1, .9), xycoords=ax.transAxes)


def mpl_scatter_density_from_df(x, y, cmap=plt.cm.GnBu, log_scale=False, vmin=0, vmax=None, figsize=(10,10),
                                y_eq_x_fn=lambda z: z, r_spearman=True, r_pearson=True, dpi=None, data=None):

    x, y, xlabel, ylabel = _parse_input(data, x, y)

    import mpl_scatter_density
    from astropy.visualization import LogStretch
    from astropy.visualization.mpl_normalize import ImageNormalize

    if log_scale:
        norm= ImageNormalize(vmin=vmin , vmax=vmax, stretch=LogStretch())
        vmin=None
        vmax=None
    else:
        norm=None

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1, projection='scatter_density')

    plot_obj = ax.scatter_density(x, y, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax, dpi=dpi)

    plot_obj.set_extent(plot_obj.get_extent())

    fig.colorbar(plot_obj, label='Number of points per pixel')

    # Add y = x line if relevant
    if y_eq_x_fn:
        x_plot = np.linspace(*ax.get_xlim(), num=int(1e6))

        y_plot = y_eq_x_fn(x_plot)
        ax.plot(x_plot, y_plot, linestyle='--', c='black', linewidth=0.25, label='y=x')

    if r_pearson:
        plt.text(0.5, 0.4, "R^2_pearson = {:.2f}".format(
            scipy.stats.pearsonr(x, y)[0] ** 2),
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # add the line
        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        abline(slope=slope,intercept=intercept)

    if r_spearman:
        plt.text(0.5, 0.5, "R^2_spearman = {:.2f}".format(
            scipy.stats.spearmanr(x, y)[0] ** 2),
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    return (ax, fig)


def add_y_eq_x(ax, y_eq_x_fn=lambda x: x):
    x_plot = np.linspace(*ax.get_xlim(), num=int(1e6))

    y_plot = y_eq_x_fn(x_plot)
    ax.plot(x_plot, y_plot, linestyle='--', c='black', linewidth=0.25, label='y=x')


def add_reg_line(x, y, ax, kind='perason'):
    if kind == 'pearson':
        plt.text(0.5, 0.4, "R^2_pearson = {:.2f}".format(
            scipy.stats.pearsonr(x, y)[0] ** 2),
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        # add the line
        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        abline(slope=slope, intercept=intercept)


    elif kind == 'spearman':
        plt.text(0.5, 0.5, "R^2_spearman = {:.2f}".format(
            scipy.stats.spearmanr(x, y)[0] ** 2),
                 horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)


def _parse_input(data, x, y):
    if isinstance(x, str) and isinstance(y, str) and data is not None:  # symbolic, seaborn-style
        xlabel = x
        ylabel = y

        x = data[x].values.ravel()
        y = data[y].values.ravel()

    else:
        try:
            if x.name is not None:
                xlabel = x.name.copy()
            else:
                xlabel='x'
        except AttributeError:
            xlabel = 'x'
        try:
            if y.name is not None:
                ylabel = y.name.copy()
            else:
                ylabel = 'y'
        except AttributeError:
            ylabel='y'

        x = np.array(x).ravel()
        y = np.array(y).ravel()
        assert np.can_cast(x, np.float64), "x must be numeric or 'data' keyword argument has to be non-None."
        assert np.can_cast(y, np.float64), "y must be numeric or 'data' keyword argument has to be non-None."

    return x, y, xlabel, ylabel


def scatterlin(data, test_var, plot_cols, figsize=(10,10), title="", order=None, save_prefix="", plot_col_names=None):
    """
    Make a violin plot for `plot_col` column of `data`, divided by `test_var`
    :param data:
    :param test_var:
    :param y_name:
    :param plot_cols:
    :param figsize:
    :param title:
    :return:
    """

    if not isinstance(plot_cols, list):
        plot_cols = [plot_cols]



    if not len(data):
        sys.stderr.write("No data to plot.")
        return
    for (i, plot_col) in enumerate(plot_cols):
        plt.figure(figsize=figsize)
        sns.violinplot(data=data, x=test_var, y=plot_col, inner="quartile", color=".8", order=order, hue_order=order)
        ax = sns.stripplot(data=data, x=test_var, y=plot_col, jitter=True, order=order, hue_order=order)
        plt.title(title)
        plt.setp(ax.get_xticklabels(), rotation=90)
        if plot_col_names is not None:
            if not isinstance(plot_col_names, list):
                plot_col_names = [plot_col_names]
            plt.ylabel(plot_col_names[i])

        if len(save_prefix):
            sys.stderr.write("Saving {}".format(plot_col))
            try:
                os.makedirs(save_prefix)
            except FileExistsError:
                pass
            plt.savefig(os.path.join(save_prefix, plot_col + ".pdf"))




def abline(slope, intercept, **kwargs):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', c='maroon', linewidth=1, label='least squares fit', **kwargs)






