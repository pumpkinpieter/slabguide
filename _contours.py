import numpy as np


def real_contour(x_start, x_end, N, s_start=0, s_end=1):
    '''
    A contour along the real line in the complex plane.

    This contour consists of N linearly spaced points between x_start and
    x_end.  This path is parameterized by the variable "s" which takes
    values in the closed interval [s_start, s_end].

    Parameters
    ----------
    x_start : float
        Left real endpoint of contour.
    x_end : float
        Right real endpoint of contour.
    N : int
        Number of points on path.
    s_start : float, optional
        Parameterization space left endpoint. The default is 0.
    s_end : TYPE, optional
        Parameterization space right endpoint. The default is 1.

    Returns
    -------
    dict
        Cs: complex array
            Points along real contour.
        dCdS: complex array
            Complex path derivative along contour,
        Ss: real array
            Points in parameterization space,
        contour_type: str
            String denoting type of contour.
    '''
    if x_start == x_end:
        raise ValueError("Please provide non-trivial real interval.")
    if N % 2 == 0:
        N += 1
    Ss = np.linspace(s_start, s_end, N)

    Cs = x_start + (x_end - x_start) * (Ss - s_start) / (s_end - s_start)
    dCdS = (x_end - x_start) / (s_end - s_start) * np.ones(N)
    return {"Cs": Cs, "dCdS": dCdS, "Ss": Ss, "contour_type": "real"}


def horizontal_contour(y, x_start, x_end, N, s_start=0, s_end=1):
    D = real_contour(x_start, x_end, N, s_start, s_end)
    D["Cs"] = D["Cs"] + 1j * y
    D["contour_type"] = "horizontal"
    return D


def circular_contour(center, radius, N, orientation="ccw", phase_shift=0,
                     total_angle=2*np.pi):
    if N % 2 == 0:
        N += 1
    if orientation not in ["cw", "ccw"]:
        raise ValueError("Orientation must be 'cw' or 'ccw'.")
    if orientation == "ccw":
        sign = 1
    else:
        sign = -1

    # Bit of an issue deciding how to parameterize cw curve.
    # See the two options below:

    # There is some problem here:
    # Ss = phase_shift + np.linspace(0, total_angle, N)
    # Cs = center + radius * np.exp(1j * sign * Ss)
    # dCdS = sign * 1j * radius * np.exp(1j * Ss)

    # This works but there is still an issue with sign,
    # since Cauchy's theorem seems to not quite add up.
    # Ss = phase_shift + np.linspace(0, sign * total_angle, N)
    # Cs = center + radius * np.exp(1j * Ss)
    # dCdS = 1j * radius * np.exp(1j * Ss)

    # Still an issue here even when I included sign inside in derivative
    # again it's Cauchy's theorem not adding right.
    Ss = phase_shift + np.linspace(0, total_angle, N)
    Cs = center + radius * np.exp(1j * sign * Ss)
    dCdS = sign * 1j * radius * np.exp(1j * sign * Ss)

    return {"Cs": Cs, "dCdS": dCdS, "Ss": Ss, "contour_type": "circle",
            "center": center, "radius": radius}


def half_circle_contour(center, radius, N, orientation="cw",
                        phase_shift=np.pi/2):

    D = circular_contour(
        center,
        radius,
        N,
        orientation=orientation,
        phase_shift=phase_shift,
        total_angle=np.pi,
    )

    D["contour_type"] = "half_circle"

    return D


def sdp_contour(Slab, x_start, x_end, N, s_start=0, s_end=1, sdp_sign=-1,
                plane="Z"):
    if x_start == x_end:
        raise ValueError("Please provide non-trivial interval.")
    if N % 2 == 0:
        N += 1
    Ss = np.linspace(s_start, s_end, N)
    xs = x_start + (x_end - x_start) * (Ss - s_start) / (s_end - s_start)
    Cs = Slab.sdp(xs, sdp_sign=sdp_sign, plane=plane)

    dxds = (x_end - x_start) / (s_end - s_start) * np.ones(N)
    dCdS = dxds * Slab.sdp_derivative(xs, sdp_sign=sdp_sign, plane=plane)

    return {"Cs": Cs, "dCdS": dCdS, "Ss": Ss, "contour_type": "sdp"}


def vertical_contour(x, y_start, y_end, N, s_start=0, s_end=1):
    if y_start == y_end:
        raise ValueError("Please provide non-trivial vertical interval.")
    if N % 2 == 0:
        N += 1
    Ss = np.linspace(s_start, s_end, N)
    ys = y_start + (y_end - y_start) * (Ss - s_start) / (s_end - s_start)
    Cs = x + 1j * ys

    dyds = (y_end - y_start) / (s_end - s_start) * np.ones(N)
    dCdS = 1j * dyds

    return {"Cs": Cs, "dCdS": dCdS, "Ss": Ss, "contour_type": "vertical"}


def vertical_contour_to_sdp(Slab, x, N, sdp_sign=-1):
    y_start, y_end = 0, Slab.sdp(x, sdp_sign=sdp_sign, plane="Z").imag
    return vertical_contour(x, y_start, y_end, N)


def imaginary_contour(y_start, y_end, N, s_start=0, s_end=1):
    D = vertical_contour(
        0, y_start, y_end, N, s_start=s_start, s_end=s_end
    )
    D["contour_type"] = "imaginary"
    return D
