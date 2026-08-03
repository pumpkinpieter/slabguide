#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:06:36 2026

@author: pv
"""
import numpy as np
from scipy.integrate import quad


def dFdZ_approx(
    Slab,
    Z_base,
    dZ=1e-6,
    Normalizer=None,
    field_type="TE",
    sign="+1",
    order=2,
    rounding=12,
):
    """Return approximate first derivative of radiation mode F with regard
    to propagation constant Z at Z_base.  Ultimately we will do this
    exactly, and then deprecate this and related functions."""

    if order not in [1, 2]:
        raise ValueError("Order must be 1 or 2.")
    FZ0 = Slab.fields(
        Z_base,
        mode_type="radiation",
        field_type=field_type,
        plane="Z",
        Normalizer=Normalizer,
        rounding=rounding,
        sign=sign,
    )

    FdZ1 = Slab.fields(
        Z_base + dZ,
        mode_type="radiation",
        field_type=field_type,
        plane="Z",
        Normalizer=Normalizer,
        rounding=rounding,
        sign=sign,
    )
    if order == 1:
        return lambda x, zs=0: (FdZ1(x, zs=zs) - FZ0(x, zs=zs)) / dZ

    else:
        FdZ2 = Slab.fields(
            Z_base + 2 * dZ,
            mode_type="radiation",
            field_type=field_type,
            plane="Z",
            Normalizer=Normalizer,
            rounding=rounding,
            sign=sign,
        )

        return lambda x, zs=0: (
            (4 * FdZ1(x, zs=zs) - 1 * FdZ2(x, zs=zs) - 3 * FZ0(x, zs=zs))
            / (2 * dZ)
        )


def radiation_transform_dZ_approx(
    Slab,
    f0,
    Lx=None,
    Rx=None,
    Z_base=1e-5,
    dZ=1e-6,
    field_type="TE",
    sign="+1",
    Normalizer=None,
    **intargs,
):
    if Lx is None:
        print("No left x integral limit given (Lx). Defaulting to left \
end of computational domain.  This will likely slow things down a lot, so it \
is recommended to rerun using more accurate bound on the support of f0.")
        Lx = Slab.Rhos[0]
    if Rx is None:
        print("No right x integral limit given (Rx). Defaults to right \
end of computational domain.  This will likely slow things down a lot, so it \
is recommended to rerun using more accurate bound on the support of f0.")
        Rx = Slab.Rhos[-1]

    Fp = dFdZ_approx(
        Z_base=Z_base,
        dZ=dZ,
        field_type=field_type,
        Normalizer=Normalizer,
        sign=sign,
    )

    def integrand(x):
        return f0(x) * Fp(x)

    return quad(integrand, Lx, Rx, complex_func=True, **intargs)[
        0
    ]


# ------------------------------- Space Wave ---------------------------------


def space_wave_approx(
    Slab,
    f0,
    Z_base=1e-7,
    dZ=1e-6,
    Lx=None,
    Rx=None,
    field_type="TE",
    Normalizer=None,
    compute_class_A=True,
    compute_class_B=True,
    **intargs,
):
    if not compute_class_A and not compute_class_B:
        raise ValueError("At least one of compute_class_A or \
compute_class_B must be set to True.")

    Kn = Slab.K0 * Slab.n0
    C1 = -(1 + 1j) * np.sqrt(np.pi) / 2

    if compute_class_A:
        dC_A = Slab.radiation_transform_dZ_approx(
            f0,
            Z_base=Z_base,
            Lx=Lx,
            Rx=Rx,
            field_type=field_type,
            Normalizer=Normalizer,
            sign=1,
            dZ=dZ,
            **intargs,
        )
        dF_A = dFdZ_approx(
            Z_base,
            dZ=dZ,
            sign=1,
            field_type=field_type,
            Normalizer=Normalizer,
        )

    if compute_class_B:

        dC_B = Slab.radiation_transform_dZ_approx(
            f0,
            Z_base=Z_base,
            Lx=Lx,
            Rx=Rx,
            field_type=field_type,
            Normalizer=Normalizer,
            sign=-1,
            dZ=dZ,
            **intargs,
        )
        dF_B = dFdZ_approx(
            Z_base,
            dZ=dZ,
            sign=-1,
            field_type=field_type,
            Normalizer=Normalizer,
        )

        def SpaceWave(x, z):
            if np.any(z == 0):
                raise ValueError("Space wave not defined at z=0.")

            C2 = np.sqrt(Kn / z) ** 3
            z0s = np.zeros_like(z)
            front = C1 * np.exp(1j * Kn * z) * C2
            summand = 0
            if compute_class_A:
                summand += dC_A * dF_A(x, z0s).T
            if compute_class_B:
                summand += dC_B * dF_B(x, z0s).T
            return (front * (summand)).T

        return SpaceWave
