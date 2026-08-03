#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 17:26:17 2026

@author: pv
"""
import numpy as np
import slab2 as sg
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# %%
A = sg.SlabGuide(symmetric=True)

# %%
half_width = .5
shift = .25
amplitude = 1

Lx = -half_width + shift
Rx = half_width + shift


def quadratic_piecewise(x):
    x = np.array(x, dtype=float)
    conds = [(x < Lx), ((Lx <= x)*(x <= Rx)), (x > Rx)]
    funcs = [lambda x: 0, lambda x: amplitude *
             (1 - ((x - shift) / half_width)**2), lambda x: 0]
    return np.piecewise(x, conds, funcs)


def quadratic_piecewise2d(x, zs=None, wavenumber=A.K0):
    fx = quadratic_piecewise(x)
    fz = np.exp(1j * wavenumber * zs)
    return np.outer(fz, fx)


f0 = quadratic_piecewise
F0 = quadratic_piecewise2d
# %%

P = sg.Propagator(A, f0, Lx, Rx)

# %%
contour1 = sg.real_contour(.001, 4*A.Z_evanescent, 80)
P.add_contour(contour1)

P.plot_transform(markersize=4)
plt.show()

# %%
contour3 = sg.real_contour(A.Z_evanescent, 4*A.Z_evanescent, 80)
P.add_contour(contour3)

P.plot_transform(markersize=4)
plt.show()

# %%
contour4 = sg.real_contour(-1, -.1, 80)
P.add_contour(contour4)


# %%

P.order_contours()

# %%

sg.plot_field_1d(A, P)
plt.show()
