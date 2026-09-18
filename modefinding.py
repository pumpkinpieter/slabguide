#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 18:49:31 2026

@author: pv
"""
import numpy as np
from scipy.integrate import quad, quad_vec
from warnings import warn
from scipy.optimize import newton
from .fields import Field


def get_guided_propagation_constants(Slab, Nref=3000, rounding=8,
                                     field_type='TE'):
    """
    Calculate the structure's guided mode propagation constants.

    Parameters
    ----------
    Nref : int, optional
        Number of points in interval for guided mode search. The default is
        3000.
    """

    if Slab.Z_max_guided.imag == 0:
        warn("Structure does not support guided modes.  Setting empty \
lists for attribute guided_prop_constants.")
        Slab.guided_prop_constants = []
        return

    def F(x): return -np.log(np.abs(Slab.eigenvalues(x*1j))**2)

    xs = np.linspace(0.000001, 1.05 * Slab.Z_max_guided.imag, Nref)
    ys = (F(xs[1:])-F(xs[:-1])) / (xs[1:] - xs[:-1])

    sign_changes = xs[1:][np.where((np.sign(ys[:-1] * ys[1:])) < 0)[0]]
    guesses = sign_changes[::2]

    Zs = []
    for (i, guess) in enumerate(guesses):
        try:
            Z = newton(Slab.eigenvalues, guess*1j)[0]
            Zs.append(Z)
        except RuntimeError:
            print("Runtime Error at index "+str(i))
            pass

    Zs = np.array(Zs)
    Zs = 1j * np.unique(Zs.imag)  # Remove possible repeats
    Zs = np.flip(Zs)  # Invert order so fundamental comes first

    Slab.guided_prop_constants = Zs


def get_guided_modes(Slab, Nref=3000, rounding=8, field_type='TE'):
    """
    Calculate guided modes of the structure.

    Parameters
    ----------
    Nref : int, optional
        Number of points in interval for guided mode search. The default is
        3000.
    """
    if not hasattr(Slab, 'guided_prop_constants'):
        Slab.get_guided_propagation_constants(Nref=Nref,
                                              field_type=field_type)
    Zs = Slab.guided_prop_constants

    if len(Zs) == 0:
        warn("Structure does not support guided modes.  Setting empty \
list for attribute guided_modes and None  guided_regional_coeffs.")
        Slab.guided_modes = []
        Slab.guided_regional_coeffs = None
        return

    Coeffs = Slab.coefficients(Zs,
                               mode_type='guided',
                               field_type=field_type)
    Slab.guided_regional_coeffs = Coeffs
    Slab.guided_modes = [Slab.fields(Z, rounding=rounding) for Z in Zs]


def get_guided_field(Propagator, Nref=3000, field_type='TE',
                     return_coeffs=False, vectorized=True, **intargs):
    """
    Get guided field of guided induced by input function f0.

    Parameters
    ----------
    f0 : callable
        Input field.
    Lx : float
        Left endpoint of support of f0.
    Rx : float
        Right endpoint of support of f0.
    Nref : int, optional
        Number of points in interval for guided mode finding. The default is
        3000.

    Returns
    -------
    callable
        Guided field induced in guide by f0.
    array
        Coefficients of guided field (integral of f0 against each guided
        mode).
    """
    if Propagator.Slab.Z_max_guided.imag == 0:
        # Return trivial field (using Slab.fields method gives easy way to get
        # output values to have correct shape)
        return 0 * Field(Propagator.Slab.fields(1, mode_type='radiation'))
    try:
        Zs = Propagator.Slab.guided_prop_constants
    except AttributeError:
        Propagator.Slab.get_guided_modes(Nref=Nref, field_type=field_type)
        Zs = Propagator.Slab.guided_prop_constants

    if vectorized:
        def integrand(x, C):
            return Propagator.f0(x) * Propagator.Slab.vectorized_fields(
                C,
                x,
                mode_type="guided",
                field_type=field_type,
                Coeffs=Propagator.Slab.guided_regional_coeffs
            )

        coeffs = quad_vec(
            integrand, Propagator.Lx, Propagator.Rx, args=(Zs,), **intargs
        )[0]

    else:

        f0, Lx, Rx = Propagator.f0, Propagator.Lx, Propagator.Rx

        coeffs = [quad(lambda x: f0(x) * F(x), Lx, Rx, complex_func=True)[0]
                  for F in Propagator.Slab.guided_modes]

    if return_coeffs:
        return np.array(coeffs)

    Fs = Propagator.Slab.guided_modes

    def Guided(x, z): return sum([c * F(x, z)
                                 for (c, F) in zip(coeffs, Fs)])

    return Field(Guided)


def get_leaky_propagation_constants(Slab, rmin, rmax, imin, imax, rref=200,
                                    iref=100, plane='Z', field_type='TE',
                                    even=False, odd=False, return_new=False):
    """
    Calculate leaky mode propagation constants in window in complex plane given
    by rmin, rmax, imin and imax.

    Parameters
    ----------
    Slab : TYPE
        DESCRIPTION.
    rmin : TYPE
        DESCRIPTION.
    rmax : TYPE
        DESCRIPTION.
    imin : TYPE
        DESCRIPTION.
    imax : TYPE
        DESCRIPTION.
    rref : TYPE, optional
        DESCRIPTION. The default is 200.
    iref : TYPE, optional
        DESCRIPTION. The default is 100.
    plane : TYPE, optional
        DESCRIPTION. The default is 'Z'.
    field_type : TYPE, optional
        DESCRIPTION. The default is 'TE'.
    even : TYPE, optional
        DESCRIPTION. The default is False.
    odd : TYPE, optional
        DESCRIPTION. The default is False.

    """

    # Create complex Cs in window
    xs = np.linspace(rmin, rmax, num=rref)
    ys = np.linspace(imin, imax, num=iref)
    Xs, Ys = np.meshgrid(xs, ys)
    Cs = Xs + 1j * Ys

    # Get eigvenvalue magnitudes in window
    Fs = np.abs(Slab.eigenvalues(Cs, plane=plane, field_type=field_type,
                                 even=even, odd=odd))

    # Find local minima and use as initial guesses
    guesses = Cs[1:-1, 1:-1][np.where((Fs[1:-1, 1:-1] < Fs[2:, 1:-1]) *
                                      (Fs[1:-1, 1:-1] < Fs[:-2, 1:-1]) *
                                      (Fs[1:-1, 1:-1] < Fs[1:-1, 2:]) *
                                      (Fs[1:-1, 1:-1] < Fs[1:-1, :-2]))]

    # Use newton solver on guesses to get prop constants
    Zs = []
    args = (plane, 'guided', field_type, None,
            1, False, even, odd)
    for (i, guess) in enumerate(guesses):
        try:
            Z = newton(Slab.eigenvalues, guess, args=args)[0]
            Zs.append(Z)
        except RuntimeError:
            print("Runtime Error at index "+str(i))
            pass

    # Append, remove repeats and sort.
    Slab.leaky_prop_constants = Slab.leaky_prop_constants + Zs
    Zs = Slab.leaky_prop_constants
    msk = np.unique(np.round(np.array(Zs), 9), return_index=True)[1]
    Slab.leaky_prop_constants = list(np.sort(np.array(Zs)[msk]))


def get_leaky_modes(Slab, rmin=None, rmax=None, imin=None, imax=None,
                    rref=200, iref=100, plane='Z', field_type='TE',
                    even=False, odd=False,
                    rounding=11):
    """
    Calculate the leaky modes of the structure with propagation constants in 
    window in complex plane given by rmin, rmax, imin, imax.

    Parameters
    ----------
    Slab : TYPE
        DESCRIPTION.
    rmin : TYPE
        DESCRIPTION.
    rmax : TYPE
        DESCRIPTION.
    imin : TYPE
        DESCRIPTION.
    imax : TYPE
        DESCRIPTION.
    rref : TYPE, optional
        DESCRIPTION. The default is 200.
    iref : TYPE, optional
        DESCRIPTION. The default is 100.
    plane : TYPE, optional
        DESCRIPTION. The default is 'Z'.
    field_type : TYPE, optional
        DESCRIPTION. The default is 'TE'.
    even : TYPE, optional
        DESCRIPTION. The default is False.
    odd : TYPE, optional
        DESCRIPTION. The default is False.
    rounding : int, optional
        Rounding parameter for regional field coefficients. Default is 12
    """

    if any((rmin is not None, rmax is not None, imin is not None,
            imax is not None)):

        Slab.get_leaky_propagation_constants(rmin, rmax, imin, imax,
                                             rref=rref,
                                             iref=iref,
                                             plane=plane,
                                             field_type=field_type,
                                             even=even,
                                             odd=odd,
                                             )
    Zs = Slab.leaky_prop_constants
    Coeffs = Slab.coefficients(Zs,
                               mode_type='guided',
                               field_type=field_type)
    Slab.leaky_regional_coeffs = Coeffs
    Slab.leaky_modes = [Slab.fields(Z,
                                    field_type=field_type,
                                    rounding=rounding,
                                    plane=plane) for Z in Zs]


def get_leaky_field(Propagator, indices=None, **kwargs):
    """
    Get leaky field associated with input field Propagator.f0 and leaky modes
    Propagator.Slab.leaky_modes.

    Parameters
    ----------
    Propagator : Propagator
        Field propagator.
    indices : slice or list of integers, optional
        Indices of leaky modes to use in construction. The default is None.
    **kwargs : dict
        Extra keyword arguments for scipy's quad_vec function.

    Returns
    -------
    Leaky : Field
        Leaky field function.
    """
    if indices is None:
        indices = slice(0, len(Propagator.Slab.leaky_prop_constants))

    Zs = np.array(Propagator.Slab.leaky_prop_constants)[indices]
    Fs = np.array(Propagator.Slab.leaky_modes)[indices]
    Cfs = np.array(Propagator.Slab.leaky_regional_coeffs)[indices]

    def integrand(x, C):
        return Propagator.f0(x) * Propagator.Slab.vectorized_fields(
            C,
            x,
            mode_type='guided',
            Coeffs=Cfs,
            **kwargs)

    coeffs = quad_vec(
        integrand, Propagator.Lx, Propagator.Rx, args=(Zs,))[0]

    def Leaky(x, z): return sum([c * F(x, z)
                                 for (c, F) in zip(coeffs, Fs)])

    # Propagator.Leaky = Field(Leaky)

    return Field(Leaky)
