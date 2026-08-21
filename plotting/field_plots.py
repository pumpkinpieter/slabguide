#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:06:36 2026

@author: pv
"""

import numpy as np
import matplotlib.pyplot as plt
from warnings import warn


def plot_field_1d(
    Slab,
    F,
    *F_args,
    xs=None,
    zs=None,
    figsize=(11, 5),
    part="real",
    plot_regions=False,
    plot_axis=True,
    plot_Rhos=True,
    hatch="///",
    contrast=0.2,
    Rho_linewidth=1,
    fig=None,
    ax=None,
    legend_fontsize=12,
    close_others=True,
    label=None,
    axis_linewidth=0.5,
    figure_name=None,
    figure_title=None,
    figure_title_fontsize=16,
    axis_title=None,
    axis_title_fontsize=14,
    **lineargs,
):

    if close_others:
        plt.close("all")

    if fig is None and ax is None:
        fig, ax = plt.subplots(1, figsize=figsize, num=figure_name)
    else:
        fig, ax = fig, ax

    ax.set_title(axis_title, fontsize=axis_title_fontsize)
    fig.suptitle(figure_title, fontsize=figure_title_fontsize)

    # Make at least one of xs and zs contain data
    if xs is None and zs is None:  # Default to plotting xs at z = 0
        xs = Slab.all_Xs
        zs = np.array([0])

    elif xs is not None and zs is None:  # Default to plotting xs at z = 0
        # xs provided, but allow for length 1 entry
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])

        if len(xs) == 1:  # if len 1, plot in z with xs as x0
            zs = np.linspace(0, 5, 300)
        else:  # otherwise assume x plotting with z0 = 0
            zs = np.array([0])

    elif xs is None and zs is not None:
        # zs provided, but allow for length 1 entry
        try:
            len(zs)
            zs = np.array(zs)
        except TypeError:
            zs = np.array([zs])

        if len(zs) == 1:  # if len 1, plot in x with zs as z0
            xs = Slab.all_Xs
        else:  # otherwise assume z plotting with x0 = 0
            xs = np.array([0])

    else:
        try:
            len(zs)
            zs = np.array(zs)
        except TypeError:
            zs = np.array([zs])
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])

    # Check that only one of xs and zs has length one, set flags
    if len(zs) > 1 and len(xs) == 1:
        plot_Rhos = False
        plot_regions = False
        ind_vars = zs
        z_vars = True
        plot_Rhos = False
        plot_regions = False
    elif len(xs) > 1 and len(zs) == 1:
        ind_vars = xs
        z_vars = False
    elif len(zs) > 1 and len(xs) > 1:
        raise ValueError("Either xs or zs must have length 1 for 1D plot.")
    else:
        raise ValueError("At least one of xs or zs must have len > 1.")

    if callable(F):

        # Catch error if input F only has one variable
        try:
            fs = F(xs, zs, *F_args)
            # Reduce dimension if necessary
            if len(fs.shape) > 1:
                if fs.shape[0] == 1 and fs.shape[1] > 1:
                    fs = fs[0]

        except TypeError:
            if z_vars:
                warn("Plotting single variable function of x along z axis. \
This defaults to plotting function in x axis.  Consider providing xs instead.")
            fs = F(ind_vars, *F_args)
    elif type(F) in [list, np.ndarray]:
        fs = np.array(F, dtype=complex)
        if len(fs) != len(ind_vars):
            raise ValueError("Length %i of plotting data does not match \
length %i of independent variables." % (len(fs), len(ind_vars)))
    else:
        raise ValueError("User must provide either callable function or data \
of correct length for input F.")

    if part == "real":
        ys = fs.real
    elif part == "imag":
        ys = fs.imag
    elif part == "norm":
        ys = np.abs(fs)
    else:
        raise ValueError("Part must be real, imag or norm.")

    if label is not None:
        ax.plot(ind_vars, ys, label=label, **lineargs)
        ax.legend(fontsize=legend_fontsize)

    else:
        ax.plot(ind_vars, ys, **lineargs)

    if plot_axis:
        near_x_axis = (min(np.abs(ys)) / np.ptp(ys)) <= 0.1
        if near_x_axis:
            ax.axhline(0, color="lightgray", linewidth=axis_linewidth)

        near_y_axis = (min(np.abs(ind_vars)) / np.ptp(ind_vars)) <= 0.1
        if near_y_axis:
            ax.axvline(0, color="lightgray", linewidth=axis_linewidth)

    if plot_Rhos:
        Rhos = Slab.Rhos[1:-1]
        msk = np.where(
            (Rhos <= np.max(ind_vars)) * (Rhos >= np.min(ind_vars))
        )
        for Rho in Rhos[msk]:
            ax.axvline(Rho, ls=":", c="lightgray", lw=Rho_linewidth)

    if plot_regions:
        Rhos = Slab.Rhos
        ns = Slab.ns
        minx, maxx = min(ind_vars), max(ind_vars)
        for i in range(len(Rhos) - 1):
            if i == 0:
                L, R = minx, Rhos[i + 1]
            elif i == len(Rhos) - 2:
                L, R = Rhos[i], maxx
            else:
                L, R = Rhos[i], Rhos[i + 1]
            n = ns[i]
            rcolor = 1 - 2 / np.pi * np.arctan(contrast * (n - 1))
            hcolor = max(
                0.9 - 2 / np.pi * np.arctan(contrast * (n - 1)), 0
            )
            plt.axvspan(L, R, color=str(rcolor), linewidth=0)
            plt.axvspan(
                L,
                R,
                fill=None,
                linewidth=0,
                hatch=hatch,
                color=str(hcolor),
            )
    return fig, ax


def add_1d_plot(
    Slab,
    F,
    *F_args,
    xs=None,
    zs=None,
    part="real",
    ax=None,
    legend_fontsize=12,
    plot_axis=True,
    plot_Rhos=True,
    Rho_linewidth=1,
    axis_linewidth=0.5,
    label=None,
    **lineargs,
):

    if ax is None:
        ax = plt.gca()

    # Make at least one of xs and zs contain data
    if xs is None and zs is None:  # Default to plotting xs at z = 0
        xs = Slab.all_Xs
        zs = np.array([0])

    elif xs is not None and zs is None:  # Default to plotting xs at z = 0
        # xs provided, but allow for length 1 entry
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])

        if len(xs) == 1:  # if len 1, plot in z with xs as x0
            zs = np.linspace(0, 5, 300)
        else:  # otherwise assume x plotting with z0 = 0
            zs = np.array([0])

    elif xs is None and zs is not None:
        # zs provided, but allow for length 1 entry
        try:
            len(zs)
            zs = np.array(zs)
        except TypeError:
            zs = np.array([zs])

        if len(zs) == 1:  # if len 1, plot in x with zs as z0
            xs = Slab.all_Xs
        else:  # otherwise assume z plotting with x0 = 0
            xs = np.array([0])

    else:
        try:
            len(zs)
            zs = np.array(zs)
        except TypeError:
            zs = np.array([zs])
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])

    # Check that only one of xs and zs has length one, set flags
    if len(zs) > 1 and len(xs) == 1:
        ind_vars = zs
        z_vars = True
        plot_Rhos = False
    elif len(xs) > 1 and len(zs) == 1:
        ind_vars = xs
        z_vars = False
    elif len(zs) > 1 and len(xs) > 1:
        raise ValueError("Either xs or zs must have length 1 for 1D plot.")
    else:
        raise ValueError("At least one of xs or zs must have len > 1.")

    if callable(F):

        # Catch error if input F only has one variable
        try:
            fs = F(xs, zs, *F_args)
            # Reduce dimension if necessary
            if len(fs.shape) > 1:
                if fs.shape[0] == 1 and fs.shape[1] > 1:
                    fs = fs[0]

        except TypeError:
            if z_vars:
                warn("Plotting single variable function of x along z axis. \
This defaults to plotting function in x axis.  Consider providing xs instead.")
            fs = F(ind_vars, *F_args)
    elif type(F) in [list, np.ndarray]:
        fs = np.array(F, dtype=complex)
        if len(fs) != len(ind_vars):
            raise ValueError("Length %i of plotting data does not match \
length %i of independent variables." % (len(fs), len(ind_vars)))
    else:
        raise ValueError("User must provide either callable function or data \
of correct length for input F.")

    if part == "real":
        ys = fs.real
    elif part == "imag":
        ys = fs.imag
    elif part == "norm":
        ys = np.abs(fs)
    else:
        raise ValueError("Part must be real, imag or norm.")

    if plot_axis:
        near_x_axis = (min(np.abs(ys)) / np.ptp(ys)) <= 0.1
        if near_x_axis:
            ax.axhline(0, color="lightgray", linewidth=axis_linewidth)

        near_y_axis = (min(np.abs(ind_vars)) / np.ptp(ind_vars)) <= 0.1
        if near_y_axis:
            ax.axvline(0, color="lightgray", linewidth=axis_linewidth)

    if plot_Rhos:
        Rhos = Slab.Rhos[1:-1]
        msk = np.where(
            (Rhos <= np.max(ind_vars)) * (Rhos >= np.min(ind_vars))
        )
        for Rho in Rhos[msk]:
            ax.axvline(Rho, ls=":", c="lightgray", lw=Rho_linewidth)

    if label is not None:
        ax.plot(ind_vars, ys, label=label, **lineargs)
        ax.legend(fontsize=legend_fontsize)
    else:
        ax.plot(ind_vars, ys, **lineargs)


def plot_field_2d(
    Slab,
    F,
    *F_args,
    xs=None,
    zs=None,
    zmin=0,
    zmax=10,
    zref=150,
    xstride=10,
    zstride=1,
    levels=40,
    part="real",
    figsize=None,
    maxdim=9,
    cmap="viridis",
    equal=True,
    plot_Rhos=True,
    Rho_lineargs={"lw": 1, "ls": ":", "c": "k"},
    colorbar=True,
    pad=0.08,
    shrink=.95,
    colorbar_frac=0.05,
    colorbar_aspect=15,
    colorbar_nticks=7,
    anchor=(0.5, 0.5),
    colorbar_format="%.2f",
    layout='vertical',
    **contourargs,
):
    """Plot field F as filled contour plot.

    If no xs are provided, plots entire region defined by SlabExact class.

    Note that we avoid applying meshgrid to xs and zs until after calling
    function. Since fields are sums of products of single functions of xs
    and single functions of zs it would be wasteful to call it on the meshed
    arrays."""

    plt.close("all")

    if layout not in ['horizontal', 'vertical']:
        raise ValueError('Layout must be horizontal or vertical.')

    if zs is None:
        zs = np.linspace(zmin, zmax, zref)
    zmin, zmax = np.min(zs), np.max(zs)

    if xs is None:
        xs = Slab.all_Xs

    xs = xs[0: -1: xstride]
    zs = zs[0: -1: zstride]

    lenzs, lenxs = np.ptp(zs), np.ptp(xs)

    if figsize is None:
        figsize = 1 / max(lenxs, lenzs) * np.array([lenxs, lenzs]) * maxdim
        if layout == 'horizontal':
            figsize[0], figsize[1] = figsize[1], figsize[0]
    else:
        figsize = figsize

    fig, ax = plt.subplots(1, figsize=figsize)

    Fs = F(xs, zs, *F_args)
    Xs, Zs = np.meshgrid(xs, zs)

    if layout == 'horizontal':
        Fs = Fs.T
        Xs, Zs = Zs.T, Xs.T

    if part == "real":
        Ys = Fs.real
    elif part == "imag":
        Ys = Fs.imag
    elif part == "norm":
        Ys = np.abs(Fs)
    else:
        raise ValueError("Part must be real, imag or norm.")

    axmap = ax.contourf(
        Xs, Zs, Ys, levels=levels, cmap=cmap, **contourargs
    )
    if equal:
        ax.set_aspect("equal")

    if layout == 'horizontal':
        ax.set_ylim(ax.get_ylim()[::-1])

    Rhos = Slab.Rhos[1:-1]

    if plot_Rhos:
        msk = np.where((Rhos <= np.max(xs)) * (Rhos >= np.min(xs)))
        if layout == 'horizontal':
            pltfunc = ax.axhline
        else:
            pltfunc = ax.axvline
        for Rho in Rhos[msk]:
            pltfunc(
                Rho,
                linestyle=Rho_lineargs["ls"],
                linewidth=Rho_lineargs["lw"],
                color=Rho_lineargs["c"],
            )
    if colorbar:
        if lenzs >= lenxs:
            if layout == 'vertical':
                cbar_orient = 'horizontal'
                loc = 'bottom'
            else:
                cbar_orient = 'vertical'
                loc = 'left'
        else:
            if layout == 'vertical':
                cbar_orient = 'vertical'
                loc = 'right'
            else:
                cbar_orient = 'horizontal'
                loc = 'bottom'

        ticks = np.linspace(np.min(Ys), np.max(Ys), colorbar_nticks)

        plt.colorbar(
            axmap,
            pad=pad,
            shrink=shrink,
            orientation=cbar_orient,
            anchor=anchor,
            fraction=colorbar_frac,
            aspect=colorbar_aspect,
            ticks=ticks,
            location=loc,
            format=colorbar_format,
        )
    return fig, ax


def add_2d_plot(
    Slab,
    F,
    *F_args,
    ax=None,
    xs=None,
    zs=None,
    zmin=0,
    zmax=10,
    zref=100,
    xstride=10,
    zstride=1,
    part="real",
    cmap="viridis",
    levels=40,
    plot_Rhos=True,
    Rho_lineargs={"lw": 1, "ls": ":", "c": "k"},
    colorbar=False,
    colorbar_orientation=None,
    pad=0.05,
    shrink=1,
    colorbar_frac=0.1,
    colorbar_aspect=15,
    anchor=(0.5, 0.5),
    colorbar_nticks=5,
    colorbar_format="%.2f",
    **contourargs,
):

    if ax is None:
        ax = plt.gca()

    if zs is None:
        zs = np.linspace(zmin, zmax, zref)

    if xs is None:
        xs = Slab.all_Xs

    xs = xs[0: -1: xstride]
    zs = zs[0: -1: zstride]

    Fs = F(xs, zs, *F_args)

    if part == "real":
        Ys = Fs.real
    elif part == "imag":
        Ys = Fs.imag
    elif part == "norm":
        Ys = np.abs(Fs)
    else:
        raise ValueError("Part must be real, imag or norm.")

    Xs, Zs = np.meshgrid(xs, zs)
    lenzs, lenxs = np.ptp(Zs), np.ptp(Xs)

    axmap = ax.contourf(
        Xs, Zs, Ys, levels=levels, cmap=cmap, **contourargs
    )

    if plot_Rhos:
        for Rho in Slab.Rhos[1:-1]:
            ax.axvline(
                Rho,
                linestyle=Rho_lineargs["ls"],
                linewidth=Rho_lineargs["lw"],
                color=Rho_lineargs["c"],
            )
    if colorbar:
        if colorbar_orientation is None:
            if lenzs >= lenxs:
                colorbar_orientation = "horizontal"
            else:
                colorbar_orientation = "vertical"
        else:
            colorbar_orientation = colorbar_orientation
        ticks = np.linspace(np.min(Ys), np.max(Ys), colorbar_nticks)

        plt.colorbar(
            axmap,
            pad=pad,
            shrink=shrink,
            orientation=colorbar_orientation,
            anchor=anchor,
            fraction=colorbar_frac,
            aspect=colorbar_aspect,
            ax=ax,
            ticks=ticks,
            format=colorbar_format,
        )


def plot_field_2d_surface(
    Slab,
    F,
    *F_args,
    xs=None,
    zs=None,
    zmin=0,
    zmax=7,
    zref=120,
    part="real",
    product_func=True,
    show_axis=False,
    figsize=(10, 5),
    cmap="viridis",
    azim=-90,
    elev=45,
    roll=0,
    zoom=2,
    rstride=2,
    cstride=10,
    z_lim_factor=1,
    colorbar=False,
    pad=0.15,
    shrink=0.85,
    colorbar_frac=0.15,
    anchor=(0.5, 0.5),
    orient="vertical",
    **surfaceargs,
):
    """Plot field F as a surface plot in 3d."""
    plt.close("all")

    if zs is None:
        zs = np.linspace(zmin, zmax, zref)
    else:
        zmin, zmax = np.min(zs), np.max(zs)

    if xs is None:
        xs = Slab.all_Xs  # doesn't duplicate endpoints

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection="3d")

    Xs, Zs = np.meshgrid(xs, zs)

    if product_func:
        fs = F(xs, zs, *F_args)
    else:
        fs = F(Xs, Zs, *F_args)

    if part == "real":
        ys = fs.real
    elif part == "imag":
        ys = fs.imag
    elif part == "norm":
        ys = np.abs(fs)
    else:
        raise ValueError("Part must be real, imag or norm.")

    axmap = ax.plot_surface(
        Xs,
        Zs,
        ys,
        clip_on=False,
        cmap=cmap,
        rstride=rstride,
        cstride=cstride,
        **surfaceargs,
    )
    # lims = (np.ptp(Xs), np.ptp(Zs), min(np.ptp(Xs), np.ptp(Zs)))
    lenys = np.ptp(ys[~np.isnan(ys)])
    lims = (np.ptp(Xs), np.ptp(Zs), z_lim_factor * lenys)
    ax.set_box_aspect(lims, zoom=zoom)
    # ax.set_box_aspect(lims, zoom=zoom)

    if not show_axis:
        ax.set_axis_off()
    ax.view_init(elev, azim, roll)
    # ax.set_zlim(-z_lim_factor * lenys, z_lim_factor * lenys)

    if colorbar:
        plt.colorbar(
            axmap,
            pad=pad,
            shrink=shrink,
            orientation=orient,
            anchor=anchor,
            fraction=colorbar_frac,
        )
    return fig, ax


def add_2d_plot_surface(
    Slab,
    F,
    *F_args,
    ax=None,
    xs=None,
    zs=None,
    zmin=0,
    zmax=5,
    zref=100,
    part="real",
    product_func=True,
    cmap="viridis",
    rstride=2,
    cstride=10,
    **surfaceargs,
):
    """Add 3d surface plot of function F to axis."""

    if ax is None:
        ax = plt.gca()

    if zs is None:
        zs = np.linspace(zmin, zmax, zref)
    else:
        zmin, zmax = np.min(zs), np.max(zs)

    if xs is None:
        xs = Slab.all_Xs  # doesn't duplicate endpoints

    Xs, Zs = np.meshgrid(xs, zs)

    if product_func:
        fs = F(xs, zs, *F_args)
    else:
        fs = F(Xs, Zs, *F_args)

    if part == "real":
        ys = fs.real
    elif part == "imag":
        ys = fs.imag
    elif part == "norm":
        ys = np.abs(fs)
    else:
        raise ValueError("Part must be real, imag or norm.")

    ax.plot_surface(
        Xs,
        Zs,
        ys,
        clip_on=False,
        cmap=cmap,
        rstride=rstride,
        cstride=cstride,
        **surfaceargs,
    )

    return ax
