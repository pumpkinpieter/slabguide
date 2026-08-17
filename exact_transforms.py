#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:13:18 2026

@author: pv
"""


def delta_transform(
    Slab, C, s=0, field_type="TE", sign="+1", Normalizer=None
):
    """Evaluate radiation transform of delta function delta(x-s)."""
    return Slab.fields_vec(
        C,
        s,
        field_type=field_type,
        mode_type="radiation",
        sign=sign,
        Normalizer=Normalizer,
    )


def integrate_interval(
    Slab, Z, L, R, field_type="TE", sign="+1",
):
    """Integrate radiation field F(x, Z) on x in (a,b)."""
    # Lj = Slab.region_index(L)
    # Rj = Slab.region_index(R)
    pass
