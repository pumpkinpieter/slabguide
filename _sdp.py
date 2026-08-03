#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:35:26 2026

@author: pv
"""
import numpy as np
from warnings import warn


def sdp(Slab, s, sdp_sign=-1, plane="Z"):
    """Return points along steepest descent path parameterized by 's'."""

    if sdp_sign not in [1, -1]:
        raise ValueError("Sign must be integer equal to 1 or -1.")

    if plane not in ["Z", "Beta", "Psi"]:
        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

    K0 = Slab.K0
    nr, ni = Slab.n0.real, Slab.n0.imag

    if plane == "Z":
        f = K0 * nr * s / ((K0 * nr) ** 2 + s**2)
        g = np.sqrt(s**2 + K0**2 * (nr**2 + ni**2))
        return s + 1j * f * (K0 * ni + sdp_sign * g)
    elif plane == "Beta":
        if ni != 0:
            warn("Beta plane sdp not verified for complex n.")
        return K0 * Slab.n0 + 1j * s
    else:
        if ni != 0:
            warn("Psi plane sdp not verified for complex n.")
        return sdp_sign * np.sign(s) * np.arccos(1 / np.cosh(s)) + 1j * s


def sdp_derivative(Slab, s, sdp_sign=-1, plane="Z"):
    """Return imag part of dZdx along steepest descent path."""

    if sdp_sign not in [1, -1]:
        raise ValueError("Sign must be integer equal to 1 or -1.")

    if plane not in ["Z", "Beta", "Psi"]:
        raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

    K0 = Slab.K0
    nr, ni = Slab.n0.real, Slab.n0.imag

    if plane == "Z":
        A = (
            K0
            * nr
            * sdp_sign
            * s**2
            / (
                (K0**2 * nr**2 + s**2)
                * np.sqrt(K0**2 * (ni**2 + nr**2) + s**2)
            )
        )

        B = (
            2
            * K0
            * nr
            * s**2
            * (
                K0 * ni
                + sdp_sign * np.sqrt(K0**2 * (ni**2 + nr**2) + s**2)
            )
            / (K0**2 * nr**2 + s**2) ** 2
        )

        C = (
            K0
            * nr
            * (
                K0 * ni
                + sdp_sign * np.sqrt(K0**2 * (ni**2 + nr**2) + s**2)
            )
            / (K0**2 * nr**2 + s**2)
        )
        return 1 + 1j * (A - B + C)

    elif plane == "Beta":
        raise NotImplementedError("Beta plane sdp derivartive not yet \
implemented.")

    else:
        raise NotImplementedError("Psi plane sdp derivative not yet \
implemented.")
