#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:02:35 2026

@author: pv
"""
from scipy.integrate import quad_vec


def radiation_transform(
    Slab,
    Cs,
    f0=None,
    s=None,
    Lx=None,
    Rx=None,
    plane="Z",
    field_type="TE",
    sign="+1",
    Normalizer=None,
    **intargs,
):
    """
    Return transform coefficients for input field f0 or (dirac delta
    centered at s), from complex input values 'Cs' using scipy vectorized
    integration methods.
    """
    if (f0 is None and s is None) or (f0 is not None and s is not None):
        raise ValueError(
            "Exactly one of either input field 'f0' or source \
point 's' must be provided."
        )

    delta_transform = f0 is None and s is not None

    if delta_transform:
        return Slab.fields_vec(
            Cs,
            s,
            field_type=field_type,
            mode_type="radiation",
            sign=sign,
            plane=plane,
            Normalizer=Normalizer,
        )

    else:
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

        def integrand(x, C, sign="+1"):
            return f0(x) * Slab.fields_vec(
                C,
                x,
                field_type=field_type,
                mode_type="radiation",
                sign=sign,
                plane=plane,
                Normalizer=Normalizer,
            )

        return quad_vec(
            integrand, Lx, Rx, args=(Cs, sign), **intargs
        )[0]


def spectral_integrand(
    Slab,
    Cs,
    f0=None,
    x=0,
    z=0,
    s=None,
    Lx=None,
    Rx=None,
    plane="Z",
    class_A_only=False,
    class_B_only=False,
    field_type="TE",
    alpha_A=None,
    alpha_B=None,
    Normalizer=None,
    det_power=0,
):
    """Return integrand of radiation expansion of f0 at Cs, x, z."""

    if class_A_only and class_B_only:
        raise ValueError("Only one of class_A_only and class_B_only flags \
can be set to True.")

    # if Lx is None:
    #     Lx = Slab.Rhos[0]
    # if Rx is None:
    #     Rx = Slab.Rhos[-1]

    if class_A_only:
        if alpha_A is None:
            alpha_A = radiation_transform(
                Cs,
                f0=f0,
                s=s,
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                sign="+1",
                plane=plane,
                Normalizer=Normalizer,
            )
        field_value = Slab.fields_vec(
            Cs,
            x,
            z=z,
            field_type=field_type,
            mode_type="radiation",
            sign="+1",
            plane=plane,
            Normalizer=Normalizer,
        )
        return alpha_A * field_value

    elif class_B_only:
        if alpha_B is None:
            alpha_B = radiation_transform(
                Cs,
                f0=f0,
                s=s,
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                sign="-1",
                plane=plane,
                Normalizer=Normalizer,
            )

        field_value = Slab.fields_vec(
            Cs,
            x,
            z=z,
            field_type=field_type,
            mode_type="radiation",
            sign="-1",
            plane=plane,
            Normalizer=Normalizer,
        )
        return alpha_B * field_value

    else:
        if alpha_A is None:
            alpha_A = radiation_transform(
                Cs,
                f0=f0,
                s=s,
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                sign="+1",
                plane=plane,
                Normalizer=Normalizer,
            )

        if alpha_B is None:
            alpha_B = radiation_transform(
                Cs,
                f0=f0,
                s=s,
                Lx=Lx,
                Rx=Rx,
                field_type=field_type,
                sign="-1",
                plane=plane,
                Normalizer=Normalizer,
            )

        field_value_A = Slab.fields_vec(
            Cs,
            x,
            z=z,
            field_type=field_type,
            mode_type="radiation",
            sign="+1",
            plane=plane,
            Normalizer=Normalizer,
        )

        field_value_B = Slab.fields_vec(
            Cs,
            x,
            z=z,
            field_type=field_type,
            mode_type="radiation",
            sign="-1",
            plane=plane,
            Normalizer=Normalizer,
        )
        if det_power > 0:
            M = Slab.transmission_matrix(Cs, plane=plane)
            detQ = -4 * M[..., 0, 0] * M[..., 1, 1]
        else:
            detQ = 1
        return detQ**det_power * (
            alpha_A * field_value_A + alpha_B * field_value_B
        )
