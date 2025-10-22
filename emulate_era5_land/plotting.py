"""
General methods for plotting all kinds of data
"""
import numpy as np
import pickle as pkl
from datetime import datetime
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
import matplotlib.dates as mdates
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LatitudeFormatter,LongitudeFormatter

#plt.rcParams.update({ "text.usetex": True, })

def geo_quad_plot(data, flabels:list, latitude, longitude,
        plot_spec={}, show=False, fig_path=None):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders

    :@param data: list of up to 4 uniform 2d spatial grids
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4,
          "cmap":"jet_r", "title":"", "map_linewidth":2,
          "norm":"linear","figsize":None, "marker":"o", "cbar_shrink":1.,
          "xtick_freq":None, "ytick_freq":None, ## pixels btw included ticks
          "idx_ticks":False, ## if True, use tick indeces instead of lat/lon
          "gridlines":False, "show_ticks":True, "use_pcolormesh":False,
          "geo_bounds":None,
          }
    plt.clf()
    ps.update(plot_spec)
    if ps.get("text_size"):
        plt.rcParams.update({"font.size":ps["text_size"]})

    geo_bounds = ps.get("geo_bounds")
    fig,ax = plt.subplots(2, 2, subplot_kw={"projection": ccrs.PlateCarree()})
    if geo_bounds is None:
        geo_bounds = [np.nanmin(longitude), np.nanmax(longitude),
                  np.nanmin(latitude), np.nanmax(latitude)]
    else:
        lon0 = np.argmin(np.abs(np.nanmin(longitude, axis=0)-geo_bounds[0]))
        lonf = np.argmin(np.abs(np.nanmax(longitude, axis=0)-geo_bounds[1]))
        lat0 = np.argmin(np.abs(np.nanmin(latitude, axis=1)-geo_bounds[2]))
        latf = np.argmin(np.abs(np.nanmax(latitude, axis=1)-geo_bounds[3]))
        slc = (slice(latf, lat0), slice(lon0, lonf))
        print(slc)

    for n in range(4):
        i = n // 2
        j = n % 2
        ## Remove any unused axis panes
        if n == len(flabels):
            fig.delaxes(ax[i,j])
            break
        ax[i,j].set_extent(geo_bounds, crs=ccrs.PlateCarree())
        ax[i,j].add_feature(
                cfeature.LAND,
                linewidth=ps.get("map_linewidth")
                )
        ax[i,j].set_title(flabels[n])

        if not ps.get("xtick_freq") is None:
            xspace = np.linspace(*geo_bounds[:2], data[n].shape[1])
            ax[i,j].set_xticks(xspace[::ps.get("xtick_freq")])
            if ps.get("idx_ticks"):
                xidxs = list(map(str,range(data[n].shape[1])))
                ax[i,j].set_xticklabels(xidxs[::ps.get("xtick_freq")])
        if not ps.get("ytick_freq") is None:
            yspace = np.linspace(*geo_bounds[2:], data[n].shape[0])
            ax[i,j].set_yticks(yspace[::ps.get("ytick_freq")])
            if ps.get("idx_ticks"):
                yidxs = list(map(str,range(data[n].shape[0])))
                ax[i,j].set_yticklabels(yidxs[::ps.get("ytick_freq")][::-1])
        if not ps.get("show_ticks", True):
            ax[i,j].axes.get_xaxis().set_ticks([])
            ax[i,j].axes.get_yaxis().set_ticks([])

        if ps.get("use_pcolormesh"):
            contour = ax[i,j].pcolormesh(
                    longitude,
                    latitude,
                    data[n],
                    cmap=ps.get("cmap"),
                    norm=ps.get("norm", "linear"),
                    vmin=None if "vmin" not in ps.keys() \
                            else ps.get("vmin")[n],
                    vmax=None if "vmax" not in ps.keys() \
                            else ps.get("vmax")[n],
                    )
        else:
            contour = ax[i,j].contourf(
                    longitude,
                    latitude,
                    data[n],
                    norm=ps.get("norm", "linear"),
                    cmap=ps.get("cmap"),
                    vmin=ps.get("vmin")[n],
                    vmax=ps.get("vmax")[n],
                    )
        ax[i,j].add_feature(
                cfeature.BORDERS,
                linewidth=ps.get("map_linewidth"),
                zorder=120
                )
        ax[i,j].add_feature(
                cfeature.STATES,
                linewidth=ps.get("map_linewidth"),
                zorder=120
                )
        ax[i,j].coastlines()
        cb = fig.colorbar(
                contour,
                ax=ax[i,j],
                    norm=ps.get("norm", "linear"),
                shrink=ps.get("cbar_shrink"),
                orientation=ps.get("cbar_orient", "vertical"),
                pad=ps.get("cbar_pad", .02),
                format=ps.get("cbar_format"),
                label=ps.get("cbar_label"),
                )
        if plot_spec.get("tick_rotation"):
            cb.ax.tick_params(rotation=plot_spec.get("tick_rotation"))

    fig.suptitle(ps.get("title"), fontsize=ps.get("title_fontsize"))

    if ps.get("gridlines"):
        plt.grid()

    if ps.get("tight_layout", False):
        plt.tight_layout()

    if not fig_path is None:
        if not ps.get("figsize") is None:
            fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=200)
        print(f"Generated image at {fig_path.as_posix()}")
    if show:
        plt.show()
    plt.close()
    return

def plot_geo_rgb(rgb:np.ndarray, lat_range:tuple, lon_range:tuple,
        plot_spec:dict={}, fig_path=None, show=False):
    """
    """
    ps = {"title":"", "figsize":(16,12), "border_linewidth":2,
            "title_size":12 }
    ps.update(plot_spec)
    fig = plt.figure(figsize=ps.get("figsize"))

    pc = ccrs.PlateCarree()

    ax = fig.add_subplot(1, 1, 1, projection=pc)
    extent = [*lon_range, *lat_range]
    ax.set_extent(extent, crs=pc)

    ax.imshow(rgb, extent=extent, transform=pc)

    ax.coastlines(color='black', linewidth=ps.get("border_linewidth"))
    ax.add_feature( ccrs.cartopy.feature.STATES,
            linewidth=ps.get("border_linewidth"))

    plt.title(ps.get("title"), fontweight='bold',
            fontsize=ps.get("title_size"))

    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), bbox_inches="tight", dpi=80)
    if show:
        plt.show()
    plt.close()
    return

def plot_stats_1d(data_dict:dict, x_labels:list=None, fig_path:Path=None,
             show:bool=False, class_space:float=.2, bar_sigma:float=1,
             fill_sigma:float=1/3, yscale="linear", fill_between=False,
             plot_spec:dict={}):
    """
    Plot the mean and standard deviation of multiple classes on the same
    X axis. Means and standard deviations for each band in each class must be
    provided as a dictionary with class labels as keys mapping to a dictionary
    with "means" and "stdevs" keys mapping to lists each with N members for
    N bands. Labels for each of the N bands must be provided separately.

    data_dict = {
        "Class 1":{"means":[9,8,7], "stdevs":[1,2,3]}
        "Class 2":{"means":[9,8,7], "stdevs":[1,2,3]}
        "Class 3":{"means":[9,8,7], "stdevs":[1,2,3]}
        }

    :@param yscale: linear by default, but logit may be good for reflectance.
    :@param class_space: directs spacing between class data points/error bars
    :@param bar_sigma: determines the size of error bars in terms of a
        constant multiple on the class' standard deviation.
    :@param fill_sigma: the shaded region of the error bar is typically
        smaller than the bar sigma.
    """
    cat_labels = list(data_dict.keys())
    if x_labels is None:
        x_labels = list(range(len(tuple(data_dict.values())[0]["means"])))

    # Merge provided plot_spec with un-provided default values
    old_ps = {"colors":{}}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    fig, ax = plt.subplots()
    transforms = [
            Affine2D().translate(n, 0.)+ax.transData
            for n in np.linspace(
                -.5*class_space, .5*class_space, num=len(cat_labels))
            ]
    ax.set_yscale(yscale)
    if not plot_spec.get("yrange") is None:
        ax.set_ylim(plot_spec.get("yrange"))
    if not plot_spec.get("xrange") is None:
        ax.set_xlim(plot_spec.get("xrange"))
    if not plot_spec.get("x_ticks") is None:
        ax.set_xticks(plot_spec.get("x_ticks"))
    if not plot_spec.get("y_ticks") is None:
        ax.set_yticks(plot_spec.get("y_ticks"))
    for i in range(len(cat_labels)):
        cat = cat_labels[i]
        ax.errorbar(
                x_labels,
                data_dict[cat]["means"],
                yerr=data_dict[cat]["stdevs"]*bar_sigma,
                marker=plot_spec.get("marker"),
                label=cat_labels[i],
                linestyle="-",
                transform=transforms[i],
                linewidth=plot_spec.get("line_width"),
                alpha=plot_spec.get("alpha"),
                elinewidth=plot_spec.get("error_line_width"),
                errorevery=plot_spec.get("error_every", 1),
                ecolor=plot_spec["colors"].get(cat_labels[i]),
                )
        if fill_between:
            under_bars = [
                    m-s*fill_sigma
                    for m,s in zip(
                        data_dict[cat]["means"],
                        data_dict[cat]["stdevs"])]
            over_bars = [
                    m+s*fill_sigma
                    for m,s in zip(
                        data_dict[cat]["means"],
                        data_dict[cat]["stdevs"])]
            ax.fill_between(
                x=x_labels,
                y1=under_bars,
                y2=over_bars,
                alpha=plot_spec.get("fill_alpha"),
                transform=transforms[i],
                color=plot_spec["colors"].get(cat_labels[i]),
                )
        ax.grid(visible=plot_spec.get("grid"))
        ax.set_xlabel(plot_spec.get("xlabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_ylabel(plot_spec.get("ylabel"),
                      fontsize=plot_spec.get("label_size"))
        ax.set_title(plot_spec.get("title"),
                fontsize=plot_spec.get("title_size"))
        ax.legend(fontsize=plot_spec.get("legend_font_size"),
                ncol=plot_spec.get("legend_ncols", 1))

    fig.tight_layout()
    if not plot_spec.get("fig_size") is None:
        fig.set_size_inches(*plot_spec.get("fig_size"))
    if show:
        plt.show()
    if fig_path:
        fig.savefig(fig_path.as_posix())
    plt.close()
    return

def plot_heatmap(heatmap:np.ndarray, fig_path:Path=None, show=False,
                 show_ticks=True, plot_diagonal:bool=False, plot_spec:dict={}):
    """
    Plot an integer heatmap, with [0,0] indexing the lower left corner
    """
    # Merge provided plot_spec with un-provided default values
    old_ps = { "cmap":"nipy_spectral", "cb_size":1, "cb_orient":"vertical",
            "imshow_norm":"linear"}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    fig, ax = plt.subplots()

    if plot_diagonal:
        ax.plot((0,heatmap.shape[1]-1), (0,heatmap.shape[0]-1),
                linewidth=plot_spec.get("line_width"))
    im = ax.imshow(
            heatmap,
            cmap=plot_spec.get("cmap"),
            vmin=plot_spec.get("vmin"),
            vmax=plot_spec.get("vmax"),
            extent=plot_spec.get("imshow_extent"),
            norm=plot_spec.get("imshow_norm"),
            origin="lower",
            aspect=plot_spec.get("imshow_aspect")
            )
    cbar = fig.colorbar(
            im, orientation=plot_spec.get("cb_orient"),
            label=plot_spec.get("cb_label"), shrink=plot_spec.get("cb_size")
            )
    if not show_ticks:
        plt.tick_params(axis="x", which="both", bottom=False,
                        top=False, labelbottom=False)
        plt.tick_params(axis="y", which="both", bottom=False,
                        top=False, labelbottom=False)
    if plot_spec.get("imshow_extent"):
        extent = plot_spec.get("imshow_extent")
        assert len(extent)==4, extent
        plt.xlim(extent[:2])
        plt.ylim(extent[2:])

    #fig.suptitle(plot_spec.get("title"))
    ax.set_title(plot_spec.get("title"))
    ax.set_xlabel(plot_spec.get("xlabel"))
    ax.set_ylabel(plot_spec.get("ylabel"))
    if not plot_spec.get("x_ticks") is None:
        ax.set_xticks(plot_spec.get("x_ticks"))
    if not plot_spec.get("y_ticks") is None:
        ax.set_yticks(plot_spec.get("y_ticks"))
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), dpi=plot_spec.get("dpi"),
                    bbox_inches="tight")
        print(f"Generated {fig_path.name}")
    plt.close()
    return

def plot_lr_func(lr_func, num_epochs:int, init_lr:float,
        show:bool=False, fig_path:Path=None, plot_spec:dict={}):
    """ """
    def_ps = {"title":"Learning rate", "xlabel":"epoch",
            "ylabel":"learning rate"}
    ps = {**def_ps, **plot_spec}
    lr = [init_lr]
    for cur_epoch in range(num_epochs):
        lr.append(lr_func(cur_epoch, lr[-1]))

    fig,ax = plt.subplots()
    ax.plot(list(range(num_epochs+1)), lr)
    ax.set_xlabel(ps.get("xlabel"))
    ax.set_ylabel(ps.get("ylabel"))
    ax.set_title(ps.get("title"))
    ax.set_xscale(plot_spec.get("xscale", "linear"))
    ax.set_yscale(plot_spec.get("yscale", "linear"))
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path.as_posix(), dpi=plot_spec.get("dpi"),
                bbox_inches="tight")
    plt.close()
    return lr

def plot_lines(domain:list, ylines:list, fig_path:Path=None,
               labels:list=[], multi_domain=False, plot_spec={},
               show:bool=False):
    """
    Plot a list of 1-d lines that share a domain and codomain.

    :@param domain: 1-d numpy array describing the common domain
    :@param ylines: list of at least 1 1-d array of data values to plot, which
            must be the same size as the domain array.
    :@param fig_path: Path to the location to store the figure. If None,
            doesn't store an image.
    :@param labels: list of string labels to include in a legend describing
            each line. If fewer labels than lines are provided, the labels
            will apply to the first of the lines.
    :@param multi_domain: If True, domain is assumed to be a list of domains
            corresponding to each of the ylines
    :@param plot_spec: Dictionary of plot options see the geo_plot module
            for plot_spec options, but the defaults are safe.
    :@param show: if True, shows the image in the matplotlib Agg client.
    """
    plt.clf()
    # Merge provided plot_spec with un-provided default values
    old_ps = {"xscale":"linear", "legend_font_size":8, "legend_ncols":1,
            "yscale":"linear", "date_format":"%Y-%m-%d",
            "colors":None, "linestyle":None}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    if multi_domain:
        assert len(domain) == len(ylines)

    # Plot each
    fig, ax = plt.subplots(figsize=plot_spec.get("fig_size"))

    if plot_spec.get("zero_axis"):
        ax.axhline(0, color="black")
    if plot_spec.get("zero_yaxis"):
        ax.axvline(0, color="black")

    colors = plot_spec.get("colors")
    if colors:
        assert len(ylines)<=len(colors)
    linestyle = plot_spec.get("linestyle")
    if linestyle:
        assert len(ylines)<=len(linestyle)
    for i in range(len(ylines)):
        if multi_domain:
            cur_domain = domain[i]
        else:
            cur_domain = domain
        ax.plot(cur_domain, ylines[i],
                label=labels[i] if len(labels) else "",
                linewidth=plot_spec.get("line_width"),
                linestyle=None if not linestyle else linestyle[i],
                color=None if not colors else colors[i])

    ax.set_xlabel(plot_spec.get("xlabel"),
            fontsize=plot_spec.get("label_size"))
    ax.set_ylabel(plot_spec.get("ylabel"),
            fontsize=plot_spec.get("label_size"))
    ax.set_title(plot_spec.get("title"),
            fontsize=plot_spec.get("title_size"))
    if plot_spec.get("xrange"):
        ax.set_xlim(*plot_spec.get("xrange"))
    if plot_spec.get("yrange"):
        ax.set_ylim(*plot_spec.get("yrange"))
    ax.set_xscale(plot_spec.get("xscale"))
    ax.set_yscale(plot_spec.get("yscale"))

    if type(cur_domain[0])==datetime:
        ax.xaxis.set_major_formatter(
                mdates.DateFormatter(plot_spec.get("date_format")))
        if plot_spec.get("time_locator"):
            ax.xaxis.set_major_locator({
                "minute":mdates.MinuteLocator,
                "day":mdates.DayLocator,
                "weekday":mdates.WeekdayLocator,
                "month":mdates.MonthLocator,
                }[plot_spec.get("time_locator")](
                    interval=plot_spec.get("time_locator_interval")
                    ))

    if plot_spec.get("xtick_rotation"):
        plt.tick_params(axis="x", **{"labelrotation":plot_spec.get(
            "xtick_rotation")})
    if plot_spec.get("ytick_rotation"):
        plt.tick_params(axis="y", **{"labelrotation":plot_spec.get(
            "ytick_rotation")})
    if len(labels):
        plt.legend(fontsize=plot_spec.get("legend_font_size"),
                   ncol=plot_spec.get("legend_ncols"))
    if plot_spec.get("xtick_align"):
        plt.setp(ax.get_xticklabels(),
                horizontalalignment=plot_spec.get("xtick_align"))

    if plot_spec.get("grid"):
        plt.grid()
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    plt.close()
    return

def plot_lines_multiy(ylines, domain, plot_spec={},
        show=False, fig_path=None):
    """
    :@param ylines: list of lists such that the first level contains a sub-list
        for each group of lines that share a domain
    """
    ps = {"fig_size":(12,6), "dpi":80, "spine_increment":.05,
            "date_format":"%Y-%m-%d", "xtick_rotation":30}
    ps.update(plot_spec)
    if len(domain) != len(ylines[0][0]):
        raise ValueError(
                "Length of 'domain' must match length of each time series.")

    fig,host = plt.subplots(figsize=ps.get("fig_size"))
    fig.subplots_adjust(left=0.2 + ps.get("spine_increment") \
            * (len(ylines) - 1))

    nlines = sum(len(yl) for yl in ylines)

    axes = [host]
    colors = ps.get("line_colors", ["C" + str(i) for i in range(nlines)])
    line_labels = ps.get("line_labels", [""] * nlines)
    line_style = ps.get("line_style", ["-"] * nlines)
    y_labels = ps.get("y_labels", [""] * len(ylines))
    y_ranges = ps.get("y_ranges", [None] * len(ylines))
    y_scales = ps.get("y_scales", ["linear"] * len(ylines))

    assert len(colors) == nlines
    assert len(line_labels) == nlines
    assert len(line_style) == nlines
    assert len(y_labels) == len(ylines)
    assert len(y_ranges) == len(ylines)
    assert len(y_scales) == len(ylines)

    ## Create additional y-axes on the left, offset horizontally
    for i in range(1, len(ylines)):
        ax = host.twinx()
        #ax.spines["left"] = ax.spines["right"]
        ax.yaxis.set_label_position("left")
        ax.yaxis.set_ticks_position("left")
        ax.spines["left"].set_position(
                ("axes", -1*ps.get("spine_increment") * i))
        ax.set_yscale(y_scales[i])
        #ax.spines["right"].set_visible(False)
        axes.append(ax)

    ## Plot each series
    lc = 0
    lines = []
    for i, (ax, series) in enumerate(zip(axes, ylines)):
        for s in series:
            lines.append(ax.plot(
                domain, s, color=colors[lc], label=line_labels[lc],
                linestyle=line_style[lc]))
            lc += 1
        ax.set_ylabel(y_labels[i], color=colors[lc-1],
                fontsize=ps.get("label_fontsize"))
        ax.tick_params(axis="y", colors=colors[lc-1])
        if y_ranges[i] is not None:
            ax.set_ylim(y_ranges[i])

    host.grid(ps.get("grid", False), **ps.get("grid_kwargs", {}))
    host.set_xlabel(ps.get("x_label", "Time"),
            fontsize=ps.get("label_fontsize"))
    host.xaxis.set_major_formatter(mdates.DateFormatter(ps.get("date_format")))
    host.tick_params(axis="x", rotation=ps.get("xtick_rotation"))
    host.legend(
            handles=[l[0] for l in lines],
            labels=line_labels,
            ncol=ps.get("legend_ncols", 1),
            fontsize=ps.get("legend_fontsize"),
            )
    if plot_spec.get("time_locator"):
        host.xaxis.set_major_locator({
            "minute":mdates.MinuteLocator,
            "day":mdates.DayLocator,
            "weekday":mdates.WeekdayLocator,
            "month":mdates.MonthLocator,
            }[plot_spec.get("time_locator")](
                interval=plot_spec.get("time_locator_interval")
                ))
    if plot_spec.get("xtick_align"):
        plt.setp(host.get_xticklabels(),
                horizontalalignment=plot_spec.get("xtick_align"))

    if ps.get("zero_axis"):
        host.axhline(0, color="black")

    plt.title(ps.get("title", ""),
            fontdict={"fontsize":ps.get("title_fontsize")})
    plt.tight_layout()
    if show:
        plt.show()
    if not fig_path is None:
        fig.savefig(fig_path, bbox_inches="tight", dpi=plot_spec.get("dpi"))
    plt.close()

def plot_combo_matrix(matrix, fig_path:Path, plot_spec={}):
    """ """
    old_ps = {"cmap":"magma", "norm":"linear", "xlabel":"Soil type",
            "ylabel":"Surface type"}
    old_ps.update(plot_spec)
    plot_spec = old_ps

    fig,ax = plt.subplots()
    cb = ax.imshow(matrix, cmap=plot_spec.get("cmap"),
            vmax=plot_spec.get("vmax"), norm=plot_spec.get("norm"))
    fig.colorbar(cb)
    ax.set_xlabel(plot_spec.get("xlabel"),
                  fontsize=plot_spec.get("label_size"))
    ax.set_ylabel(plot_spec.get("ylabel"),
                  fontsize=plot_spec.get("label_size"))
    ax.set_title(plot_spec.get("title"),
            fontsize=plot_spec.get("title_size"))

    # Adding labels to the matrix
    if not plot_spec.get("yticks") is None:
        assert len(plot_spec.get("yticks"))==matrix.shape[0]
        ax.set_yticks(range(matrix.shape[0]), plot_spec.get("yticks"),
                **plot_spec.get("ytick_params", {}))
    if not plot_spec.get("xticks") is None:
        assert len(plot_spec.get("xticks"))==matrix.shape[1]
        ax.set_xticks(range(matrix.shape[1]), plot_spec.get("xticks"),
                **plot_spec.get("xtick_params", {"rotation":45, "ha":"right"}))
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close()
    return

def plot_quad_sequence(
        pred_array, fig_path=None, true_array=None, pred_coarseness=1,
        plot_spec={}, show=False):
    """
    Plot a series of true and predicted sequences in a 4-panel plot,
    each panel containing the data from a single feature

    :@param true_array:(N,S,4) shaped array with N sequence samples, each of
        length S and having 4 features corresponding to feat_labels
    :@param pred_array:(N,S,4) shaped array with N sequence samples, each of
        length S and having 4 features corresponding to feat_labels
    """
    if not true_array is None:
        assert true_array.shape==pred_array.shape, \
                (true_array.shape, pred_array.shape)
    ps = {
            "true_linewidth":1, "pred_linewidth":1,
            "true_linestyle":"-", "pred_linestyle":"-", "main_title":"",
            "quad_titles":["", "", "", ""], "figsize":(12,12), "dpi":100,
            "yscale":"linear", "lines_rgb":None, "grid":False,
            }
    ps.update(plot_spec)
    seq_range = np.arange(pred_array.shape[1]) * pred_coarseness
    plt.clf()
    fig,ax = plt.subplots(2, 2)
    num_px = pred_array.shape[0]
    cm = matplotlib.cm.get_cmap("hsv", num_px)

    has_legend = False
    for n in range(4):
        i = n // 2
        j = n % 2
        ## If fewer than 4 features are included, stop plotting
        if n == pred_array.shape[-1]:
            fig.delaxes(ax[i,j])
            break
        pixel_plots = []
        for px in range(pred_array.shape[0]):
            if not ps.get("lines_rgb") is None:
                color_true = ps["lines_rgb"][px]
                color_pred = ps["lines_rgb"][px]
            else:
                color_true = ps.get("true_color", cm(px))
                color_pred = ps.get("pred_color", cm(px))

            if not true_array is None:
                tmp_ax_true, = ax[i,j].plot(
                        seq_range,
                        true_array[px,:,n],
                        color=color_true,
                        linewidth=ps.get("true_linewidth"),
                        alpha=ps.get("true_line_opacity"),
                        linestyle=ps.get("true_linestyle", "-")
                        )
            tmp_ax_pred, = ax[i,j].plot(
                    seq_range,
                    pred_array[px,:,n],
                    color=color_pred,
                    linewidth=ps.get("pred_linewidth"),
                    alpha=ps.get("pred_line_opacity"),
                    linestyle=ps.get("pred_linestyle", "-")
                    )
            pixel_plots.append(tmp_ax_pred)

        ax[i,j].set_title(ps["quad_titles"][n],
                fontsize=ps.get("quad_title_size",12))
        if ps.get("hlines"):
            ax[i,j].axhline(ps["hlines"][n], color="black")
        if ps.get("xticks"):
            ax[i,j].set_xticks(
                    range(len(ps.get("xticks"))),
                    labels=ps.get("xticks"),
                    rotation=ps.get("xticks_rotation", 0))
        if not ps.get("subplot_xlabels") is None:
            ax[i,j].set_xlabel(ps.get("subplot_xlabels")[n])
        if not ps.get("subplot_ylabels") is None:
            ax[i,j].set_ylabel(ps.get("subplot_ylabels")[n])
        if ps.get("yrange"):
            ax[i,j].set_ylim(ps.get("yrange"))
        if ps.get("xrange"):
            ax[i,j].set_xlim(ps.get("xrange"))
        ax[i,j].set_xscale(ps.get("xscale", "linear"))
        ax[i,j].set_yscale(ps.get("yscale", "linear"))

    if ps.get("legend_linestyle"):
        for k,ln in enumerate(pixel_plots):
            pixel_plots[k] = Line2D(
                    [0,1],[0,1],
                    linestyle=ps.get("legend_linestyle"),
                    color=ln.get_color(),
                    )

    ## Add a legend if it is requested but hasn't been added yet
    if not ps.get("per_pixel_legend") is None and not has_legend:
        fig_legend = fig.legend(
                pixel_plots,
                ps["per_pixel_legend"],
                loc=ps.get("legend_location", "upper left"),
                prop={"size": ps.get("legend_size",12)},
                ncol=ps.get("legend_ncols", 1),
                bbox_to_anchor=ps.get(
                    "legend_bbox_to_anchor", (0,0,1,1)),
                )
        has_legend = True
    elif not ps.get("pred_legend_label") is None and not has_legend:
        if true_array is None:
            fig_legend = fig.legend(
                    (tmp_ax_pred,),
                    (ps.get("pred_legend_label"),),
                    loc=ps.get("legend_location", "upper left"),
                    bbox_to_anchor=ps.get(
                        "legend_bbox_to_anchor", (0,0,1,1)),
                    )
        else:
            fig_legend = fig.legend(
                    (tmp_ax_pred, tmp_ax_true),
                    (ps.get("pred_legend_label"),
                        ps.get("true_legend_label")),
                    loc=ps.get("legend_location", "upper left"),
                    prop={"size": ps.get("legend_size",12)},
                    ncol=ps.get("legend_ncols", 1),
                    bbox_to_anchor=ps.get(
                        "legend_bbox_to_anchor", (0,0,1,1)),
                    )
        has_legend = True

    fig.supxlabel(ps.get("xlabel"), fontsize=ps.get("xlabel_size", 16))
    fig.supylabel(ps.get("ylabel"), fontsize=ps.get("ylabel_size", 16))

    if ps.get("grid"):
        plt.grid()
    if ps["main_title"] != "":
        fig.suptitle(ps["main_title"], fontsize=ps.get("main_title_size", 16))
    plt.tight_layout()
    if not fig_path is None:
        print(f"Saving {fig_path.as_posix()}")
        if ps.get("figsize"):
            fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",
                dpi=ps.get("dpi"))
    if show:
        plt.show()
    plt.close()
    return

def plot_nested_bars(data_dict:dict, labels:dict={}, plot_error_bars=False,
        bar_colors:list=None, plot_spec:dict={}, show=False, fig_path=None,
        group_order:list=None, bar_order:list=None):
    """
    Plot a bar graph of metrics nested 2 levels deep, with optional error bars.

    :@param data_dict: Dict nested 2 layers deep, where the first layer
        identifies the bar grouping, and the second layer identifies the
        subcategory of a data point within each bar grouping. The second layer
        should map to a number if plot_error_bars is False, or a 2-tuple of
        [data, error_bar_magnitude] if plot_error_bars is True.
    :@param labels: dict of optional labels to replace data_dict keys in the
        legend or x-axis, if the data_dict key matches a labels key
    :@param plot_error_bars: Determines whether to expect a 2-tuple including
        error bar data per bar, as specified above
    :@param plot_spec: Dict of configuration options for the plot
    """
    ps = {"xlabel":"", "ylabel":"", "text_size":12, "title":"", "dpi":80,
            "figsize":(12,12), "legend_ncols":1, "line_opacity":1,
            "cmap":"hsv", "label_fontsize":14, "title_fontsize":20,
            "legend_fontsize":14, "bar_spacing":1}
    ps.update(plot_spec)
    fig,ax = plt.subplots()

    ## group keys
    gkeys = list(data_dict.keys()) if group_order is None else group_order
    ngroups = len(gkeys)
    group_starts = np.arange(ngroups)
    assert all(set(data_dict[k])==set(data_dict[gkeys[0]]) for k in gkeys[1:])
    ## bar keys
    bkeys = list(data_dict[gkeys[0]]) if bar_order is None else bar_order
    cm = matplotlib.cm.get_cmap(ps.get("cmap"), len(bkeys))

    bwidth = ps.get("bar_width", 1/(len(bkeys)+ps.get("bar_spacing")))

    bar_plots = []
    err_plots = []
    offset = 0
    for bix,bk in enumerate(bkeys):
        if plot_error_bars:
            tmp_data = [data_dict[gk][bk][0] for gk in gkeys]
            tmp_err = [data_dict[gk][bk][1] for gk in gkeys]
        else:
            tmp_data = [data_dict[gk][bk] for gk in gkeys]
            tmp_err = None
        bar_plots.append(ax.bar(
                group_starts + offset,
                tmp_data,
                color=cm(bix) if bar_colors is None else bar_colors[bix],
                width=bwidth,
                label=labels.get(bk,bk),
                ))
        if plot_error_bars:
            err_plots.append(ax.errorbar(
                    group_starts + offset,
                    tmp_data,
                    yerr=tmp_err,
                    fmt=ps.get("err_fmt","o"),
                    color=ps.get("err_color","black"),
                    ))
        offset += bwidth

    ax.set_xticks(
            group_starts+bwidth/2, [labels.get(gk,gk) for gk in gkeys],
            rotation=ps.get("xtick_rotation", 0),
            fontsize=ps.get("xtick_fontsize", ps.get("label_fontsize")),
            )
    plt.yticks(fontsize=ps.get("label_fontsize"))

    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("label_fontsize"))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("label_fontsize"))
    if not ps.get("ylim") is None:
        ax.set_ylim(*ps.get("ylim"))
    ax.set_title(ps.get("title"), fontsize=ps.get("title_fontsize"))
    ax.legend(ncol=ps.get("legend_ncols"), fontsize=ps.get("legend_fontsize"))

    if show:
        plt.show()
    if fig_path:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(),bbox_inches="tight",dpi=ps.get("dpi"))
    return

def plot_hists(counts:list, labels:list, bin_coords:np.array, normalize=False,
        line_colors:list=None, plot_spec:dict={}, show=False, fig_path=None):
    """
    Plot one or more histograms on a single pane

    :@param counts: List of 1D arrays representing the binned counts
    :@param labels: List of string labels corresponding to each histogram
    :@param bin_mins: List of 2-tuple (min, max) data coordinate values for
        each histogram. The minimum should be the minimum value of the first
        bin, and the maximum should be the upper value of the last bin.
    :@param plot_spec: Dict of configuration options for the plot
    """
    ps = {"xlabel":"", "ylabel":"", "linewidth":2, "text_size":12,
            "title":"", "dpi":80, "norm":None,"figsize":(12,12),
            "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
            "label_fontsize":14, "title_fontsize":20, "legend_fontsize":14,
            "xscale":"linear", "yscale":"linear", "tick_fontsize":14,
            }
    ps.update(plot_spec)
    fig,ax = plt.subplots()
    cm = matplotlib.cm.get_cmap(ps.get("cmap"), len(counts))

    if not ps.get("hlines") is None:
        for hl in ps.get("hlines"):
            hlparams = {"color":"black", "linewidth":ps.get("linewidth")}
            if isinstance(hl, (list,tuple)) and len(hl)==2:
                hl,hlpnew = hl
                hlparams.update(hlpnew)
            ax.axhline(hl, **hlparams)
    if not ps.get("vlines") is None:
        for vl in ps.get("vlines"):
            vlparams = {"color":"black", "linewidth":ps.get("linewidth")}
            if isinstance(vl, (list,tuple)) and len(vl)==2:
                vl,vlpnew = vl
                vlparams.update(vlpnew)
            ax.axvline(vl, **vlparams)

    for i,(carr,label,bins) in enumerate(zip(counts,labels,bin_coords)):
        assert len(carr.shape) == 1, f"counts array must be 1D, {carr.shape=}"
        #bins = (np.arange(carr.size)+.5)/carr.size * (bmax-bmin) + bmin
        color = cm(i) if not line_colors else line_colors[i]
        if normalize:
            carr = carr / np.sum(carr)
        ax.plot(bins, carr, label=label, linewidth=ps.get("linewidth"),
                color=color, alpha=ps.get("line_opacity"))

    if not ps.get("ylim") is None:
        ax.set_ylim(*ps.get("ylim"))
    if not ps.get("xlim") is None:
        ax.set_xlim(*ps.get("xlim"))

    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("label_fontsize"))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("label_fontsize"))
    ax.set_title(ps.get("title"), fontsize=ps.get("title_fontsize"))
    ax.legend(ncol=ps.get("legend_ncols"), fontsize=ps.get("legend_fontsize"))
    ax.set_xscale(ps.get("xscale"))
    ax.set_yscale(ps.get("yscale"))
    ax.tick_params(axis='both',which='major',labelsize=ps.get("tick_fontsize"))
    ax.tick_params(axis='both',which='minor',labelsize=ps.get("tick_fontsize"))

    if show:
        plt.show()
    if fig_path:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(),bbox_inches="tight",dpi=ps.get("dpi"))
    return

def plot_scatter(x, y, size=None, color=None, plot_spec={},
        fig_path=None, show=False):
    """
    basic scatterplot using my plot_spec standard
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4, "dpi":200, "cmap":"jet",
            "text_size":12, "title":"", "norm":"linear", "marker":"o",
            "cbar_shrink":1., "map_linewidth":2, "title_fontsize":14,
            "legend_fontsize":14, "tick_fontsize":10, "legend_ncols":1}
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    fig, ax = plt.subplots(figsize=ps.get("fig_size"))
    ax.scatter(
            x=x, y=y, s=size, c=color,
            marker=ps.get("marker"),
            cmap=ps.get("cmap"),
            vmin=ps.get("vmin"),
            norm=ps.get("norm"),
            vmax=ps.get("vmax"),
            linewidths=ps.get("linewidths"), ## marker edges
            )
    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("label_fontsize"))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("label_fontsize"))
    ax.set_title(ps.get("title"), fontsize=ps.get("title_fontsize"))
    #ax.legend(ncol=ps.get("legend_ncols"), fontsize=ps.get("legend_fontsize"))
    ax.tick_params(axis='both',which='major',labelsize=ps.get("tick_fontsize"))
    ax.tick_params(axis='both',which='minor',labelsize=ps.get("tick_fontsize"))

    if show:
        plt.show()
    if fig_path:
        fig.savefig(fig_path.as_posix(),bbox_inches="tight",dpi=ps.get("dpi"))

def plot_geo_scalar(data, latitude, longitude, bounds=None, plot_spec={},
             latlon_ticks=False, show=False, fig_path=None,
             use_contours=False):
    """
    Plot a gridded scalar value on a geodetic domain, using cartopy for borders
    """
    ps = {"xlabel":"", "ylabel":"", "marker_size":4,
          "cmap":"jet_r", "text_size":12, "title":"",
          "norm":"linear","figsize":(12,12), "marker":"o", "cbar_shrink":1.,
          "map_linewidth":2}
    plt.clf()
    ps.update(plot_spec)
    plt.rcParams.update({"font.size":ps["text_size"]})

    ax = plt.axes(projection=ccrs.PlateCarree())
    fig = plt.gcf()
    if bounds is None:
        bounds = [np.amin(longitude), np.amax(longitude),
                  np.amin(latitude), np.amax(latitude)]
    ax.set_extent(bounds, crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.LAKES, linewidth=ps.get("map_linewidth"))
    #ax.add_feature(cfeature.RIVERS, linewidth=ps.get("map_linewidth"))

    ax.set_title(ps.get("title"), fontsize=ps.get("fontsize_title", 18))
    ax.set_xlabel(ps.get("xlabel"), fontsize=ps.get("fontsize_labels", 14))
    ax.set_ylabel(ps.get("ylabel"), fontsize=ps.get("fontsize_labels", 14))

    if use_contours:
        scat = ax.contourf(
                longitude,
                latitude,
                data,
                cmap=ps.get("cmap"),
                norm=ps.get("norm"),
                vmin=ps.get("vmin"),
                vmax=ps.get("vmax"),
                )
    else:
        scat = ax.pcolormesh(
                longitude,
                latitude,
                data,
                cmap=ps.get("cmap"),
                norm=ps.get("norm"),
                vmin=ps.get("vmin"),
                vmax=ps.get("vmax"),
                )

    if latlon_ticks:
        '''
        lonmin,lonmax,latmin,latmax = bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.arange(data.shape[0])[::frq],
                labels=np.linspace(latmin,latmax,data.shape[0])[::frq])
        ax.set_xticks(np.arange(data.shape[1])[::frq],
                labels=np.linspace(lonmin,lonmax,data.shape[1])[::frq])
        ax.tick_params(rotation=ps.get("tick_rotation", 0))
        '''
        lonmin,lonmax,latmin,latmax = bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,data.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,data.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))

    ax.add_feature(cfeature.BORDERS, linewidth=ps.get("map_linewidth"),
                   zorder=120)
    ax.add_feature(cfeature.STATES, linewidth=ps.get("map_linewidth"),
                   zorder=120)
    ax.coastlines()
    fig.colorbar(
            scat,
            ax=ax,
            shrink=ps.get("cbar_shrink"),
            label=ps.get("cbar_label"),
            orientation=ps.get("cbar_orient", "vertical"),
            pad=ps.get("cbar_pad", 0.0),
            norm=ps.get("norm"),
            )
    scat.figure.axes[0].tick_params(
            axis="both", labelsize=ps.get("fontsize_labels",14))

    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()

def plot_geo_ints(int_data, lat, lon, geo_bounds=None, latlon_ticks=True,
        int_ticks=None, int_labels=None, fig_path=None,
        colors=None, show=False, plot_spec={}):
    """
    Plots a map with pixels colored according to a 2D array of integer values.

    :@param int_data: 2D numpy array of integer values to be visualized
    :@param latitudes: 1D array of latitudes corresponding to rows in `data`
    :@param longitudes: 1D array of longitudes corresponding to columns in`data`
    :@param colors: list or dict mapping indeces present in int_data to
        matplotlib-valid colors
    """
    ps = {"xlabel":"", "ylabel":"",
            "title":"", "dpi":80, "norm":None,"figsize":(12,12),
            "legend_ncols":1, "line_opacity":1, "cmap":"hsv",
            "label_size":14, "title_size":20}
    ps.update(plot_spec)
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})

    ax.add_feature(
            cfeature.LAND,
            linestyle=ps.get("border_style", "-"),
            linewidth=ps.get("border_linewidth", 2),
            edgecolor=ps.get("border_color", "black"),
            )

    ax.add_feature(
            cfeature.BORDERS,
            linestyle=ps.get("border_style", "-"),
            linewidth=ps.get("border_linewidth", 2),
            edgecolor=ps.get("border_color", "black"),
            )
    ax.add_feature(
            cfeature.STATES,
            linestyle=ps.get("border_style", "-"),
            linewidth=ps.get("border_linewidth", 2),
            edgecolor=ps.get("border_color", "black"),
            )

    if geo_bounds is None:
        geo_bounds = [np.amin(lon), np.amax(lon), np.amin(lat), np.amax(lat)]
    ax.set_extent(geo_bounds, crs=ccrs.PlateCarree())

    m_invalid = ~np.isfinite(int_data)
    int_data[m_invalid] = int_data[~m_invalid][0]
    int_data = int_data.astype(int)

    ## assign each unique integer to an index
    unq_ints = np.unique(int_data)
    val_to_ix = {v:ix for ix,v in enumerate(unq_ints)}
    if colors is None:
        ref_cmap = plt.get_cmap(ps.get("cmap", "tab20"), unq_ints.size)
        cmap = ListedColormap([ref_cmap(i) for i in range(unq_ints.size)])
    else:
        cmap = ListedColormap([colors[v] for v in unq_ints])
    if int_labels is None:
        ix_labels = list(unq_ints)
    else:
        ix_labels = [int_labels[v] for v in unq_ints]
    ix_data = np.vectorize(val_to_ix.get)(int_data).astype(float)
    ix_data[m_invalid] = np.nan

    im = ax.imshow(
            ix_data,
            origin=ps.get("origin", "upper"),
            cmap=cmap,
            extent=geo_bounds,
            interpolation=ps.get("interpolation")
            )

    if latlon_ticks:
        lonmin,lonmax,latmin,latmax = geo_bounds
        frq = ps.get("tick_frequency", 1)
        ax.set_yticks(np.linspace(latmin,latmax,ix_data.shape[0])[::frq],
                crs=ccrs.PlateCarree())
        ax.set_xticks(np.linspace(lonmin,lonmax,ix_data.shape[1])[::frq],
                crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter(zero_direction_label=True)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        ax.tick_params(rotation=ps.get("tick_rotation", 0))
        '''
        ax.set_yticks(np.arange(ix_data.shape[0])[::frq],
                labels=np.linspace(latmin,latmax,ix_data.shape[0])[::frq])
        ax.set_xticks(np.arange(ix_data.shape[1])[::frq],
                labels=np.linspace(lonmin,lonmax,ix_data.shape[1])[::frq])
        '''

    cbar = plt.colorbar(
            im, ax=ax,
            orientation=ps.get("cbar_orient", "vertical"),
            pad=ps.get("cbar_pad", 0.0),
            shrink=ps.get("cbar_shrink", 1.)
            )
    ## make a scale that centers ticks on their color bar increments
    nunq = unq_ints.size
    ticks = np.array(list(range(nunq))) * (nunq-1)/nunq + .5
    cbar.ax.tick_params(rotation=ps.get("cbar_tick_rotation", 0))
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(ix_labels)
    cbar.ax.tick_params(labelsize=ps.get("cbar_fontsize", 14))

    cbar.set_label(ps.get("cbar_label"))
    ax.set_title(ps.get("title", ""), fontsize=ps.get("title_fontsize", 18))
    if not fig_path is None:
        fig.set_size_inches(*ps.get("figsize"))
        fig.savefig(fig_path.as_posix(), bbox_inches="tight",dpi=80)
    if show:
        plt.show()
    return

if __name__=="__main__":
    pass
