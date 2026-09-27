#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:39:20 2026

@author: pv
"""
import numpy as np
import matplotlib.pyplot as plt
from ..sdp import sdp


def plot_points(Cs, m="o", ms=5, ax=None, **kwargs):
    """Plot provided points in complex plane."""
    try:
        len(Cs)
        Cs = np.asanyarray(Cs)
    except TypeError:
        Cs = np.asanyarray([Cs])
    if ax is None:
        ax = plt.gca()
    for Z in Cs:
        ax.plot(Z.real, Z.imag, marker=m, markersize=ms, **kwargs)


def plot_refractive_index(
    Slab,
    figsize=(11, 4),
    contrast=0.2,
    color="cornflowerblue",
    label=None,
    plot_regions=False,
    hatch="///",
    plot_Rhos=True,
    ax=None,
    part="both",
    legend_fontsize=12,
    Rho_linewidth=1,
    title="Refractive Index Plot\n",
    title_fontsize=14,
    **linekwargs,
):
    """Plot refractive index profile."""

    plt.close('all')

    if ax is not None:
        fig = plt.gcf()
    else:
        fig, ax = plt.subplots(1, figsize=figsize)

    Rhos = Slab.Rhos
    if part == "real":
        ns = Slab.ns.real
    elif part == "imag":
        ns = Slab.ns.imag
    elif part == "both":
        nsr = Slab.ns.real
        nsi = Slab.ns.imag
    else:
        raise ValueError("Part must be real or imag or both.")

    if label is None:
        label = part
        if part == "both":
            label1 = "real"
            label2 = "imag"
    if part != "both":
        for i, (n, Xs) in enumerate(zip(ns, Slab.Xs)):
            if i == 0:
                ax.plot(
                    Xs,
                    n * np.ones_like(Xs),
                    color=color,
                    label=label,
                    **linekwargs,
                )
            ax.plot(Xs, n * np.ones_like(Xs), color=color, **linekwargs)
        for i in range(1, len(Rhos) - 1):
            Rho = Rhos[i]
            nl, nr = ns[i - 1], ns[i]
            ax.plot([Rho, Rho], [nl, nr], color=color, **linekwargs)
    else:
        for i, (n, Xs) in enumerate(zip(nsr, Slab.Xs)):
            if i == 0:
                ax.plot(
                    Xs,
                    n * np.ones_like(Xs),
                    color=color,
                    label=label1,
                    **linekwargs,
                )
            ax.plot(Xs, n * np.ones_like(Xs), color=color, **linekwargs)
        for i in range(1, len(Rhos) - 1):
            Rho = Rhos[i]
            nl, nr = nsr[i - 1], nsr[i]
            ax.plot([Rho, Rho], [nl, nr], color=color, **linekwargs)

        for i, (n, Xs) in enumerate(zip(nsi, Slab.Xs)):
            if i == 0:
                ax.plot(
                    Xs,
                    n * np.ones_like(Xs),
                    color="C1",
                    label=label2,
                    **linekwargs,
                )
            ax.plot(Xs, n * np.ones_like(Xs), color="C1", **linekwargs)
        for i in range(1, len(Rhos) - 1):
            Rho = Rhos[i]
            nl, nr = nsi[i - 1], nsi[i]
            ax.plot([Rho, Rho], [nl, nr], color="C1", **linekwargs)

    if plot_Rhos:
        for Rho in Rhos[1:-1]:
            ax.axvline(Rho, ls=":", c="lightgray", lw=Rho_linewidth)

    if plot_regions:
        for i in range(len(Rhos) - 1):
            n = ns[i]
            rcolor = 1 - 2 / np.pi * np.arctan(contrast * (n - 1))
            hcolor = max(
                0.9 - 2 / np.pi * np.arctan(contrast * (n - 1)), 0
            )
            plt.axvspan(
                Rhos[i], Rhos[i + 1], color=str(rcolor), linewidth=0
            )
            plt.axvspan(
                Rhos[i],
                Rhos[i + 1],
                fill=None,
                linewidth=0,
                hatch=hatch,
                color=str(hcolor),
            )
    ax.legend(fontsize=legend_fontsize)
    fig.suptitle(title, fontsize=title_fontsize)
    return fig, ax


def eigenvalue_plot(
    Slab,
    rmin,
    rmax,
    imin,
    imax,
    rref=200,
    iref=200,
    levels=70,
    plane="Z",
    field_type="TE",
    mode_type="guided",
    derivate=False,
    even=False,
    odd=False,
    log_abs=True,
    equal=False,
    grid=True,
    fig=None,
    ax=None,
    figsize=(11, 5),
    cmap="viridis",
    part="norm",
    facecolor="gray",
    plot_sdp=False,
    sdp_sign=-1,
    plot_axis=True,
    plot_Z_evanescent=True,
    axis_linewidth=0.7,
    axis_linecolor="k",
    Normalizer=None,
    sign=1,
    colorbar=True,
    pad=0.02,
    shrink=1,
    colorbar_frac=0.05,
    colorbar_aspect=30,
    anchor=(0.5, 1),
    sdp_lc="C0",
    title_fontsize=14,
    **contourargs,
):
    """Plot locations of eigenvalues in the complex plane in window given by
    rmin, rmax, imin, imax."""

    plt.close("all")
    xs = np.linspace(rmin, rmax, num=rref)
    ys = np.linspace(imin, imax, num=iref)

    if max(xs) < Slab.Z_evanescent.real:
        plot_Z_evanescent = False
    if min(xs) > Slab.Z_evanescent.real:
        plot_Z_evanescent = False

    Xs, Ys = np.meshgrid(xs, ys)
    Cs = Xs + 1j * Ys

    Fs = Slab.eigenvalues(
        Cs,
        field_type=field_type,
        mode_type=mode_type,
        plane=plane,
        Normalizer=Normalizer,
        sign=sign,
        derivate=derivate,
        even=even,
        odd=odd,
    )

    if part == "real":
        Fs = Fs.real
    elif part == "imag":
        Fs = Fs.imag
    elif part == "norm":
        Fs = np.abs(Fs)
    elif part == "phase":
        Fs = np.angle(Fs)
    else:
        raise ValueError("Part must be real, imag, norm or phase.")

    if log_abs:
        Fs = np.log(np.abs(Fs))

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot()
        ax.grid(grid)
        ax.set_facecolor(facecolor)

    if plot_axis:
        if (imin <= 0) * (imax >= 0):
            ax.axhline(0, color=axis_linecolor, linewidth=axis_linewidth)
        if (rmin <= 0) * (rmax >= 0):
            ax.axvline(0, color=axis_linecolor, linewidth=axis_linewidth)

    im = ax.contour(xs, ys, Fs, levels=levels, cmap=cmap, **contourargs)

    if plot_sdp:
        if plane == "Z":
            sdp_points = sdp(Slab, xs, sdp_sign=sdp_sign, plane=plane)
            sdp_ys = sdp_points.imag
            msk = np.where((sdp_ys < imax) * (sdp_ys > imin))
            ax.plot(xs[msk], sdp_ys[msk], c=sdp_lc)

        elif plane == "Beta":
            sdp_points = sdp(Slab, ys, sdp_sign=sdp_sign, plane=plane)
            sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
            if (rmin <= Slab.K0 * Slab.n0) * (rmax >= Slab.K0 * Slab.n0):
                ax.plot(sdp_xs, sdp_ys, c=sdp_lc)

        elif plane == "Psi":
            sdp_points = sdp(Slab, ys, sdp_sign=sdp_sign, plane=plane)
            sdp_xs, sdp_ys = sdp_points.real, sdp_points.imag
            msk = np.where((sdp_xs < rmax) * (sdp_xs > rmin))
            ax.plot(sdp_xs[msk], sdp_ys[msk], c=sdp_lc)
        else:
            raise ValueError("Plane must be 'Z', 'Beta' or 'Psi'.")

    if equal:
        ax.axis("equal")

    if colorbar:
        fig.colorbar(
            im,
            pad=pad,
            shrink=shrink,
            orientation="vertical",
            anchor=anchor,
            fraction=colorbar_frac,
            aspect=colorbar_aspect,
            ax=[ax],
        )
    fig.suptitle("Determinant Plot in %s Plane\n" % plane,
                 fontsize=title_fontsize)
    if plot_Z_evanescent:
        ax.axvline(
            Slab.Z_evanescent.real,
            color="grey",
            linestyle=":",
            linewidth=1.25,
        )
    plt.show(fig)
    return fig, ax, im


def eigenvalue_plot_1d(
    Slab,
    contour,
    plane="Z",
    derivate=False,
    log_abs=True,
    fig=None,
    ax=None,
    figsize=(11, 5),
    cmap="viridis",
    part="norm",
    field_type="TE",
    mode_type="guided",
    plot_axis=True,
    axis_linewidth=0.7,
    axis_linecolor="k",
    Normalizer=None,
    sign=1,
    title_fontsize=14,
    **lineargs
):

    plt.close("all")

    Cs, contour_type = contour["Cs"], contour["contour_type"]

    if contour_type in ["real", "horizontal", "sdp"]:
        xs = Cs.real
        if max(xs) < Slab.Z_evanescent.real:
            plot_Z_evanescent = False
    elif contour_type in ["imaginary", "vertical"]:
        plot_Z_evanescent = False
        xs = Cs.imag
    else:
        plot_Z_evanescent = False
        xs = contour["Ss"]

    Fs = Slab.eigenvalues(
        Cs,
        field_type=field_type,
        mode_type=mode_type,
        plane=plane,
        Normalizer=Normalizer,
        sign=sign,
        derivate=derivate,
    )

    if part == "real":
        Fs = Fs.real
    elif part == "imag":
        Fs = Fs.imag
    elif part == "norm":
        Fs = np.abs(Fs)
    elif part == "phase":
        Fs = np.angle(Fs)
    else:
        raise ValueError("Part must be real, imag, norm or phase.")

    if log_abs:
        Fs = np.log(np.abs(Fs))

    if fig is None:
        fig = plt.figure(figsize=figsize)
    if ax is None:
        ax = fig.add_subplot()

    if plot_axis:
        ax.axhline(0, color=axis_linecolor, linewidth=axis_linewidth)
        ax.axvline(0, color=axis_linecolor, linewidth=axis_linewidth)

    ax.plot(xs.real, Fs, **lineargs)
    fig.suptitle("Determinant Plot in %s Plane\n" % plane,
                 fontsize=title_fontsize)

    if plot_Z_evanescent:
        ax.axvline(
            Slab.Z_evanescent.real,
            color="grey",
            linestyle=":",
            linewidth=1.25,
        )
    return fig, ax


def plot_transform(
    Propagator,
    indices=None,
    figsize=(11, 4),
    part="real",
    marker="o",
    markersize=3,
    ax=None,
    plot_axis=True,
    axis_linewidth=1.25,
    legend_fontsize=12,
    color_I='C0',
    color_II='C1',
    plot_Z_evanescent=True,
    legend=True,
    plot_class_I=True,
    plot_class_II=True,
    **linekwargs,
):
    """
    Plot radiation transform of input associated with Propagator.

    Parameters
    ----------
    Propagator : TYPE
        DESCRIPTION.
    indices : TYPE, optional
        DESCRIPTION. The default is None.
    figsize : TYPE, optional
        DESCRIPTION. The default is (11, 4).
    part : TYPE, optional
        DESCRIPTION. The default is "real".
    marker : TYPE, optional
        DESCRIPTION. The default is "o".
    markersize : TYPE, optional
        DESCRIPTION. The default is 3.
    ax : TYPE, optional
        DESCRIPTION. The default is None.
    plot_axis : TYPE, optional
        DESCRIPTION. The default is True.
    axis_linewidth : TYPE, optional
        DESCRIPTION. The default is 1.25.
    legend_fontsize : TYPE, optional
        DESCRIPTION. The default is 12.
    color_I : TYPE, optional
        DESCRIPTION. The default is None.
    color_II : TYPE, optional
        DESCRIPTION. The default is None.
    plot_Z_evanescent : TYPE, optional
        DESCRIPTION. The default is True.
    **linekwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    fig : TYPE
        DESCRIPTION.
    ax : TYPE
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.
    """
    if ax is None:
        plt.close('all')
        fig, ax = plt.subplots(1, figsize=figsize)
    else:
        fig = plt.gcf()

    if indices is None:
        indices = slice(0, Propagator.N_contours)

    contours = np.array(Propagator.contours)[indices]

    for i in range(len(contours)):

        contour = contours[i]
        bools = [Propagator.compute_class_I * plot_class_I,
                 Propagator.compute_class_II * plot_class_II]

        for idx, boolean in enumerate(bools):

            if idx == 0:
                alphas = contour['alphas_I']
                color = color_I
                if i == 0:
                    label = 'class I %s' % part
                else:
                    label = None
            else:
                alphas = contour['alphas_II']
                color = color_II
                if i == 0:
                    label = 'class II %s' % part
                else:
                    label = None
            if boolean:

                if part == "real":
                    ys = alphas.real
                elif part == "imag":
                    ys = alphas.imag
                elif part == "norm":
                    ys = np.abs(alphas)
                else:
                    raise ValueError("Part must be real, imag, or norm.")

                if contour['contour_type'] in ["real",
                                               "horizontal",
                                               "sdp"]:
                    xs = contour['Cs'].real
                if contour['contour_type'] in ["imaginary",
                                               "vertical"]:
                    xs = contour['Cs'].imag
                    plot_Z_evanescent = False
                if contour['contour_type'] in ["circle",
                                               "half_circle"]:
                    xs = contour['Ss']
                    plot_Z_evanescent = False

                ax.plot(xs, ys, marker=marker, markersize=markersize,
                        color=color, label=label, **linekwargs)

                if plot_Z_evanescent:

                    flag = (np.min(xs) <= Propagator.Slab.Z_evanescent.real) *\
                        (np.max(xs) >= Propagator.Slab.Z_evanescent.real)

                    if flag:
                        plt.axvline(Propagator.Slab.Z_evanescent.real, lw=.75,
                                    ls='--', color='gray')

                if plot_axis:

                    ax.axhline(0, color="gray",
                               linewidth=axis_linewidth)

                    near_y_axis = (min(np.abs(xs)) / np.ptp(xs)) <= 0.1
                    if near_y_axis:
                        ax.axvline(0, color="gray",
                                   linewidth=axis_linewidth)

        if legend:
            ax.legend(fontsize=legend_fontsize)

    return fig, ax
