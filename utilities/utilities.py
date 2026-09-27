#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 14:16:45 2025

@author: pv
"""
import numpy as np
import matplotlib.pyplot as plt
import json


def copy_doc(copy_func):
    """Copies the doc string of the given function to another. 
    This function is intended to be used as a decorator.

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @copy_doc(foo)
        def bar():
            ...
    """

    def wrapped(func):
        func.__doc__ = copy_func.__doc__
        return func

    return wrapped


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


def plot_complex(f, rmin, rmax, imin, imax, fargs=(), fkwargs={}, rref=100,
                 iref=100, levels=70, log_abs=True, equal=False, grid=True,
                 colorbar=True,  figsize=(11, 5), cmap='viridis', part='norm',
                 facecolor='gray', close_others=True, pad=.02, shrink=1,
                 colorbar_frac=.05, colorbar_aspect=30, anchor=(.5, 1),
                 max_value=None, min_value=None, filled=False,
                 **contourargs):
    '''Plot complex valued function f on window in complex plane defined by
    rmin, rmax, imin, imax.'''
    if close_others:
        plt.close('all')
    xs = np.linspace(rmin, rmax, num=rref)
    ys = np.linspace(imin, imax, num=iref)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = Xs + 1j * Ys
    Fs = f(Zs, *fargs, **fkwargs)

    if part == 'real':
        F = Fs.real

    elif part == 'imag':
        F = Fs.imag
    elif part == 'norm':
        F = np.abs(Fs)
    elif part == 'phase':
        F = np.angle(Fs)
    else:
        raise ValueError('Part must be real, imag, norm or phase.')

    if log_abs:
        if part == 'phase':
            print('Taking log and absolute value of phase. May wish to set \
log_abs=False.')
        F = np.log(np.abs(F))

    if max_value is not None:
        msk = np.where(F > max_value)
        F[msk] = np.abs(max_value) * F[msk] / np.abs(F[msk])
    if min_value is not None:
        msk = np.where(F < min_value)
        F[msk] = np.abs(min_value) * F[msk] / np.abs(F[msk])

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot()

    ax.grid(grid)
    ax.set_facecolor(facecolor)
    if filled:
        im = ax.contourf(xs, ys, F, levels=levels, **contourargs)
    else:
        im = ax.contour(xs, ys, F, levels=levels, **contourargs)

    if equal:
        ax.set_aspect('equal', adjustable='box')

    if colorbar:
        plt.colorbar(im, pad=pad, shrink=shrink,
                     orientation='vertical',
                     anchor=anchor, fraction=colorbar_frac,
                     aspect=colorbar_aspect, ax=ax)
    # if colorbar:
    #     plt.colorbar(im, pad=pad, shrink=shrink,
    #                  orientation='vertical',
    #                  aspect=colorbar_aspect,
    #                  anchor=anchor, fraction=colorbar_frac)


def plot_complex_surface(f, rmin, rmax, imin, imax,
                         rref=100, iref=100, fargs=(), fkwargs={},
                         log_abs=False, equal=False,
                         colorbar=True,  figsize=(10, 5), cmap='viridis',
                         part='norm', z_lim_factor=2,
                         elev=50, roll=0, azim=-90, zoom=2, axis_off=True,
                         rstride=2, cstride=2, max_value=None,
                         min_value=None, return_vals=False,
                         **contourargs):
    '''Plot complex input function f on window in complex plane defined by
    rmin, rmax, imin, imax.'''
    plt.close('all')
    xs = np.linspace(rmin, rmax, num=rref)
    ys = np.linspace(imin, imax, num=iref)
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = Xs + 1j * Ys
    Fs = f(Zs, *fargs, **fkwargs)

    if part == 'real':
        Fs = Fs.real
    elif part == 'imag':
        Fs = Fs.imag
    elif part == 'norm':
        Fs = np.abs(Fs)
    elif part == 'phase':
        Fs = np.angle(Fs)
    else:
        raise ValueError('Part must be real, imag, norm or phase.')

    if log_abs:
        Fs = np.log(np.abs(Fs))

    if max_value is not None:
        msk = np.where(Fs > max_value)
        Fs[msk] = np.abs(max_value) * Fs[msk] / np.abs(Fs[msk])
    if min_value is not None:
        msk = np.where(Fs < min_value)
        Fs[msk] = np.abs(min_value) * Fs[msk] / np.abs(Fs[msk])

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(projection='3d')
    lims = (np.ptp(xs), np.ptp(ys), min(np.ptp(xs), np.ptp(ys)))
    ax.set_box_aspect(lims, zoom=zoom)
    rangeFs = np.ptp(Fs[~np.isnan(Fs)])
    ax.set_zlim(-z_lim_factor * rangeFs, z_lim_factor * rangeFs)

    if axis_off:
        ax.set_axis_off()

    ax.view_init(elev, azim, roll)

    im = ax.plot_surface(Xs, Ys, Fs, clip_on=False, cmap=cmap,
                         rstride=rstride, cstride=cstride,
                         **contourargs)
    if equal:
        ax.set_aspect('equal', adjustable='box')

    if colorbar:
        plt.colorbar(im, pad=.05, orientation='vertical',
                     anchor=(.5, 1), fraction=.15)
    if return_vals:
        return Fs


def plot_xy_surface(f, xmin, xmax, ymin, ymax, **kwargs):
    def f_complex(C): return f(C.real, C.imag)
    plot_complex_surface(f_complex, xmin, xmax, ymin, ymax, **kwargs)


def plot_complex_contour(f, contour, fkwargs={}, rref=20, iref=20,
                         levels=25, log_abs=False, grid=True,
                         colorbar=True,  figsize=(11, 5), part='both',
                         legend_fontsize=12, gridlw=.1, max_value=None,
                         min_value=None, **lineargs):
    '''Plot complex valued function f on contour in complex plane.'''
    plt.close('all')
    Zs, Ss = contour['Zs'], contour['Ss']
    contour_type = contour['contour_type']

    if contour_type in ['real', 'horizontal', 'sdp']:
        xs = Zs.real
    if contour_type in ['imaginary', 'vertical']:
        xs = Zs.imag
    if contour_type in ['circle', 'half_circle']:
        xs = Ss
    Fs = f(Zs, **fkwargs)

    if part == 'real':
        F = Fs.real
    elif part == 'imag':
        F = Fs.imag
    elif part == 'both':
        Fr = Fs.real
        Fi = Fs.imag
    elif part == 'norm':
        F = np.abs(Fs)
    elif part == 'phase':
        F = np.angle(Fs)
    else:
        raise ValueError('Part must be real, imag, norm or phase.')

    if log_abs:
        if part == 'phase':
            print('Taking log and absolute value of phase. May wish to set \
log_abs=False.')
        if part == 'both':
            Fr = np.log(np.abs(Fr))
            Fi = np.log(np.abs(Fi))
        else:
            F = np.log(np.abs(F))

    if max_value is not None:
        msk = np.where(F > max_value)
        F[msk] = np.abs(max_value) * F[msk] / np.abs(F[msk])
    if min_value is not None:
        msk = np.where(F < min_value)
        F[msk] = np.abs(min_value) * F[msk] / np.abs(F[msk])

    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot()
    ax.grid(grid, lw=gridlw)
    if part == 'both':
        ax.plot(xs, Fr, label='real', **lineargs)
        ax.plot(xs, Fi, label='imag', **lineargs)
    else:
        ax.plot(xs, Fr, label=part, **lineargs)
    plt.legend(fontsize=legend_fontsize)


class ThreeJSMeshSaver():

    def __init__(self, Xs, Ys):
        """
        A class for saving grid data for ThreeJS meshes.

        Takes the results of np.meshgrid(xs, zs), where, xs and zs are 1d numpy
        arrays of points in the transverse and longitudinal axes.  

        Parameters
        ----------
        Xs : ndarray
            2D array of x values for base mesh.
        Ys : ndarray
            2D array of z values for base mesh.
        """

        # Set base mesh arrays as attributes
        self.Xs = Xs
        self.Ys = Ys

        # Get total number of points in base mesh arrays
        self.L = self.Xs.shape[0] * self.Xs.shape[1]

        # Compute triangle index list and base vertex array
        self.trig_indices = self._get_triangle_indices()
        self.base_vertices = self._get_base_vertices()

    def _get_triangle_indices(self):
        """
        Get list of indices to triangulate base plane for ThreeJS mesh
        formation via indexed buffer geometry.

        Returns
        -------
        ndarray
            Array of integers determining triangles of base plane.
        """
        # Get shape of base plane discretization
        Ny, Nx = self.Xs.shape[0], self.Xs.shape[1]

        # Form index list of base mesh points
        indices = np.arange(0, Ny * Nx, dtype=np.uint32).reshape((Ny, Nx))

        # Get lower left index point of each rectangle in base mesh
        fbases = indices[:Ny-1, :Nx-1].flatten()
        Nb = len(fbases)

        # Get lower right triangles (vectorized)
        f1 = np.array([indices[:Ny-1, :Nx-1],
                       indices[:Ny-1, 1:Nx],
                       indices[1:Ny, 1:Nx]],
                      dtype=np.uint32).transpose(1, 2, 0).reshape(Nb, 3)

        # f1 = np.array([[i, i+1, i+1+Nx] for i in fbases], dtype=np.uint32)

        # Get upper left triangles (vectorized)
        f2 = np.array([indices[:Ny-1, :Nx-1],
                       indices[1:Ny, 1:Nx],
                       indices[1:Ny, :Nx-1]],
                      dtype=np.uint32).transpose(1, 2, 0).reshape(Nb, 3)
        # f2 = np.array([[i, i+1+Nx, i+Nx] for i in fbases], dtype=np.uint32)

        # Concatenate arrays and return
        f = np.concatenate((f1, f2), dtype=np.uint32)

        return f

    def _get_base_vertices(self):
        """
        Get vertices of triangles with zero height (y component in three.js).

        Returns array with correct shape for interpretation by
        IndexedBufferedGeometry in ThreeJS.

        Returns
        -------
        ndarray
            Array of base vertices with zero y component, shape is self.L x 3.
        """

        Base = np.array([self.Xs, self.Ys, np.zeros_like(self.Xs)],
                        dtype=np.float32)
        base_vertices = Base.transpose(1, 2, 0).reshape((self.L, 3))
        return base_vertices

    def create_height_array_from_complex(self, heights):
        '''Get data for height from complex ys input (needs to be formatted as
        list: [ys[0].real, ys[0].imag, ys[1].real, ys[1].imag, ... ].'''
        Heights = np.array([heights.real, heights.imag], dtype=np.float32)
        heights = Heights.transpose(1, 2, 0).reshape((self.L, 2))
        return heights

    def save_as_json(self, filepath, array):
        with open(filepath + '.json', 'w') as outfile:
            json.dump(array.flatten().tolist(), outfile, separators=(',', ':'),
                      sort_keys=True, indent=4)

    def save_trig_indices_as_json(self, filepath):
        self.save_as_json(filepath, self.trig_indices)

    def save_base_vertices_as_json(self, filepath):
        self.save_as_json(filepath, self.base_vertices)

    def save_heights_as_json(self, filepath, heights):
        heights_array = self.create_height_array_from_complex(heights)
        self.save_as_json(filepath, heights_array)

    def save_values_as_json(self, filepath, vals):
        with open(filepath + '.json', 'w') as outfile:
            json.dump(vals.tolist(), outfile, separators=(',', ':'),
                      sort_keys=True, indent=4)
