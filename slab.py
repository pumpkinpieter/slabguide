#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 20:40:22 2021.

@author: pv

Class for modeling fibers with 1D transverse plane.
"""

import numpy as np
from .normalizers import OurMethod
from .sdp import sdp, sdp_derivative
from .fields import Field
from .plotting import (plot_field_1d, plot_field_2d, add_1d_plot, add_2d_plot,
                       plot_field_2d_surface, add_2d_plot_surface,
                       eigenvalue_plot, eigenvalue_plot_1d, plot_points,
                       plot_refractive_index)
from copy import deepcopy
from warnings import warn
from scipy.optimize import newton, minimize


class Slab:
    """A class for calculating the fields of step index slab waveguides."""

    def __init__(
        self,
        Ts=[6, 2, 6],
        ns=[1.3, 1.5, 1.3],
        wl=1.8e-6,
        scale=1e-6,
        xrefs=None,
        no_xs=False,
        Shift=0,
        center=True,
    ):
        """
        Parameters
        ----------
        Ts : list or array, optional
            A list of non-dimensional layer thicknesses.  These define the
            thickness of the layers of the guide, from left to right.  The
            first and last entries define the extent of the outer region of the
            guide, truncating that which would otherwise extend indefinitely.
            The default is [6, 2, 6].
        ns : list or array, optional
            A list of refractive indices. These give the refractive index of
            each layer of material. The first and last entry must be equal.
            The default is [1.3, 1.5, 1.3].
        wl : float, optional
            The dimensional operating wavelength. The default is 1.8e-6.
        scale : float, optional
            Scaling factor which non-dimensionalizes all calculations. The
            default is 1e-6.
        xrefs : list or array, optional
            Number of x values to have in each region when constructing x
            values for plotting. If None, values are chosen to give good
            resolution based on region thickness. The default is None. 
        no_xs : bool, optional
            Whether or not to create and store array of x values.  Used
            primarily in applications where plotting is not desired. The
            default is False.
        Shift : float, optional
            Non-dimensioal shift for translating computational domain away from
            origin. The default is 0.
        center : bool, optional
            Centers the geometry at the origin. The default is True.
        """
        self.no_xs = no_xs

        # Check inputs for errors
        self.check_parameters(Ts, ns, xrefs)

        # Variables beginning with upper case are non-dimensional
        self.Shift = Shift

        # Variables beginning with lower case are dimensional
        self.scale = scale
        self.shift = Shift * scale

        self.Ts = np.array(Ts, dtype=float)
        self.ts = self.Ts * self.scale

        # Material interface coordinates (non-dimensional)
        self.Rhos = np.array([sum(self.Ts[:i]) for i in range(len(self.Ts)
                                                              + 1)])
        self.N_regions = len(self.Rhos) - 1
        if xrefs is None:
            f = np.array(ns) / np.min(np.array(ns))
            k = 10
            s = (k * (f - 1.0))**(1/5) + 1
            xrefs = np.array(s * self.Ts * 50, dtype=int) + 1

        self.xrefs = xrefs

        # Center geometry at y=0.
        if center:
            self.Shift = 1 / 2 * self.Rhos[-1]

        # Translate geometry by Shift value
        self.Rhos -= self.Shift
        self.rhos = self.Rhos * self.scale

        # Create x values for plotting
        if not self.no_xs:
            self.Xs = [
                np.linspace(self.Rhos[i], self.Rhos[i + 1], self.xrefs[i])
                for i in range(len(self.Rhos) - 1)
            ]
            self.all_Xs = np.concatenate(
                [self.Xs[i][:-1] for i in range(len(self.Xs))]
            )
            self.all_Xs = np.append(self.all_Xs, self.Xs[-1][-1])

        # Copy input refractive index information. Necessary since self.ns will
        # vary from input ns depending on wavelength (see below).
        self.ns_in = deepcopy(ns)

        # Set wavelength. Note that this is a property that also sets
        # refractive indices (self.ns)
        self.wl = wl

        # Check if guide is symmetric (not yet utilized but plan to use this to
        # reduce computational load in calculating transmission matrix)
        self.is_symmetric = self.check_symmetric()

        # Define base regional models
        # def M(x, Z, i, **kwargs):
        #     return np.exp(1j * Z * (x - kwargs['Offsets'][i][0]))

        # def N(x, Z, i, **kwargs):
        #     return np.exp(-1j * Z * (x - kwargs['Offsets'][i][1]))

        # def dMdx(x, Z, i, **kwargs): return 1j * Z * M(x, Z, i, **kwargs)
        # def dNdx(x, Z, i, **kwargs): return -1j * Z * N(x, Z, i, **kwargs)

        # # Collect models in dictionary
        # model_dict = {'M': M, 'N': N, 'dMdx': dMdx, 'dNdx': dNdx}

        # # Assign to each region.
        # self.regional_models = [model_dict for i in range(self.N_regions)]

        # # Define non-dimensional offsets for regional fields
        # # self.get_optimal_offsets()  # need to fix for dual shifts
        # self.Offsets = [[self.Rhos[i], self.Rhos[i+1]]
        #                 for i in range(1, len(self.Rhos)-1)]
        # self.Offsets.insert(0, [self.Rhos[1], self.Rhos[1]])
        # self.Offsets[-1] = [self.Rhos[-2], self.Rhos[-2]]
        # self.Offsets = np.array(self.Offsets)
        # self.model_kwargs = {'Offsets': self.Offsets}

    def __name__():
        return "Slab"

    @property
    def wl(self):
        """Get wavelength."""
        return self._wl

    @wl.setter
    def wl(self, wl):
        """Set wavelength and associated material parameters."""

        self._wl = wl
        N = len(self.ns_in)

        # Allow for callable wavelength functions (i.e. for dispersion)
        self.ns = np.array(
            [
                self.ns_in[i](wl) if callable(self.ns_in[i]) else self.ns_in[i]
                for i in range(N)
            ]
        )

        # Set wavenumber
        self.k0 = 2 * np.pi / wl
        self.K0 = self.k0 * self.scale

        # Set refractive index used in integral expansion.  This is currently
        # the shared value of the outer refractive indices
        self.n0 = np.max(self.ns[[0, -1]])
        self.n_low = np.min(self.ns)
        self.n_high = np.max(self.ns)

        # Set regional wavenumbers (used to bound guided propagation constants
        # in Beta plane)
        self.ks = self.k0 * self.ns
        self.Ks = self.K0 * self.ns
        self.K_low, self.K_high = np.min(self.Ks), np.max(self.Ks)

        # Z value where radiation modes have hyperbolic parts (0 if n_low = n0)
        # Also appears to be where leaky modes cutoff.
        self.Z_hyperbolic = self.K0 * np.sqrt(self.n0**2 - self.n_low**2)

        # Z value where radiation modes shift from propagating to evanescent
        self.Z_evanescent = self.Zi_from_Beta(0).real

        # Z value before which we have one-sided radiation modes (not currently
        # implemented).
        # self.Z_one_sided = self.K0 * np.sqrt(
        #     np.max(self.ns[[0, -1]]) ** 2 - np.min(self.ns[[0, -1]]) ** 2
        # )

        # V number of guide
        self.V = self.K0 * np.sqrt(self.n_high**2 - self.n0**2, dtype=complex)

        # Max pure imaginary value for guided modes in Z plane
        self.Z_max_guided = 1j * self.V

        # Set Z_max_guided to zero if V number is imaginary
        if self.n_high <= self.n0:
            self.Z_max_guided = 0 + 0j

        if hasattr(self, "guided_modes"):
            delattr(self, "guided_modes")
            delattr(self, "guided_prop_constants")
            delattr(self, "guided_regional_coeffs")

    def check_parameters(self, ts, ns, xrefs):

        # Check that all relevant inputs have same length
        lengths = [len(ts), len(ns)]
        names = ["ts", "ns"]

        if ns[-1] != ns[0]:
            raise ValueError(
                "Outer regions must have equal refractive indices."
            )

        # Check for adjacent layers with equal refractive indices and combine.
        if np.any(np.array(ns)[:-1] == np.array(ns)[1:]):
            warn("Adjacent layers with equal refractive indices detected. \
Combining layers to avoid singular behavior in transfer matrices.")

            # Initialize index shift to correct for removals
            shift = 0
            for i in range(len(ns) - 1):
                # If next n value is same as current
                if ns[i+1+shift] == ns[i+shift]:
                    # Remove next value from ns list
                    ns.pop(i+1+shift)
                    # Combine regions by removing next t value and adding it to
                    # current one.
                    ts[i+shift] += ts.pop(i+1+shift)
                    # Adjust index shift to account for removals
                    shift -= 1

        if xrefs is not None:
            lengths.append(len(xrefs))
            names.append("xrefs")

        lengths = np.array(lengths)

        same = all(x == lengths[0] for x in lengths)

        if not same:
            string = "Provided parameters not of same length: \n\n"
            for name, length in zip(names, lengths):
                string += name + ": " + str(length) + "\n"
            raise ValueError(string + "\nModify above inputs as necessary and \
try again.")

    def check_symmetric(self,):
        '''
        Determine if guide is symmetric.

        Will be used to improve computation, but this is not yet implemented.

        Returns
        -------
        bool
            True if guide is symmetric, else False.
        '''
        b1 = np.all(np.flip(self.ns) == self.ns)
        b2 = np.all(np.flip(self.Ts) == self.Ts)
        if b1 and b2:
            return True
        return False

    def get_optimal_offsets(self,):
        '''
        Find optimal offsets for regional fields.

        Based on minimizing the impact on blow up in transfer matrix entries
        caused by large values of the exponential function.  These values occur
        when the imaginary part of regional transverse propagation constant (Z)
        is non-trival, and can be ameliorated by making the remaining portion
        of the exponent, namely (rho_j - a_j) or (rho_j - a_(j-1)) as small as
        possible (including weighting by the size of the imaginary part of Z).

        Note that since the objective funciton is a simple quadratic, there is
        a closed form solution for the minimum, which we should implement soon.

        Returns
        -------
        None
            Sets attribute self.Offsets with result of minimization.
        '''
        Z0 = self.Z_max_guided
        Zs = np.array([self.Zi_from_Z0(Z0, n) for n in self.ns])

        def F(a_s):
            s = 0
            for i in range(1, len(self.Rhos)-1):
                Z_left, Z_right = Zs[i-1], Zs[i]
                if np.abs(Z_left.imag) > 1e-9:
                    s += Z_left.imag**2 * (self.Rhos[i] - a_s[i-1])**2
                if np.abs(Z_right.imag) > 1e-9:
                    s += Z_right.imag**2 * (self.Rhos[i] - a_s[i])**2
            return s

        output = minimize(F, np.zeros(len(self.Rhos)-1))
        self.Offsets = output['x']
        return output['x']

    def condition_list(self, xs):
        '''
        Generate condition list for piecewise function definitions.

        Parameters
        ----------
        xs : list or array
            Independent variables.

        Returns
        -------
        list of boolean arrays
            Entry 'i' of list has index 'j' equal to true if xs[j] is between
            self.Rhos[i] and self.Rhos[i+1].
        '''
        # Give xs at least one dimension and ensure it's an array
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])

        # If one region, no restrictions wanted
        if len(self.ns) == 1:
            return [(-np.inf <= xs) * (xs <= np.inf)]

        # Set up piecewise condition list
        conds = [
            (self.Rhos[i] <= xs) * (xs < self.Rhos[i + 1])
            for i in range(len(self.Rhos) - 1)
        ]

        # Extend outer region conditions to +- infinity
        conds[0] = xs < self.Rhos[1]
        conds[-1] = self.Rhos[-2] <= xs

        return conds

    def Beta_from_Zi(self, Z, ni=None):
        """Find non-dimensional Beta from non-dimensional Z."""
        if ni is None:
            ni = self.n0
        return np.sqrt(self.K0**2 * ni**2 - Z**2, dtype=complex)

    def Zi_from_Beta(self, Beta, ni=None):
        """Get scaled Z from scaled Beta."""
        if ni is None:
            ni = self.n0
        return np.sqrt(self.K0**2 * ni**2 - Beta**2, dtype=complex)

    def Zi_from_Z0(self, Z0, ni):
        """Get Z on region i (with refractive index ni) from Z0. Vectorized
        in Z0 and ni"""
        if ni == self.n0:
            return Z0
        Z0, ni = np.array(Z0), np.array(ni)
        return np.sqrt(
            (ni[np.newaxis] ** 2 - self.n0**2) * self.K0**2
            + Z0[..., np.newaxis] ** 2,
            dtype=complex,
        )[..., 0]

    def Psi_i_from_Psi0(self, Psi0, ni):
        """Get Psi on region i (with refractive index ni) from Psi0."""
        if ni == self.n0:
            return Psi0
        return np.arccos(self.n0 / ni * np.cos(Psi0))

    def region_index(self, xs, continuity='left'):
        """Return region indices in which (non-dimensional) xs lie.  Indexing
        is based on left-continuity, so if x is in (rho_i, rho_i+1) it gets
        index i.  Points outside of domain on left get index 0, on right get
        index len(Rhos)-1.  Vectorized."""
        try:
            len(xs)
            xs = np.array(xs)
        except TypeError:
            xs = np.array([xs])

        if continuity not in ['left', 'right']:
            raise ValueError('Must have continuity = "left" or "right"')
        idx = np.zeros_like(xs, dtype=int)

        idx[np.where(xs <= self.Rhos[0])] = 0
        for j in range(len(self.Rhos) - 1):
            if continuity == 'left':
                condition = (xs <= self.Rhos[j + 1]) * (xs > self.Rhos[j])
            else:
                condition = (xs < self.Rhos[j + 1]) * (xs >= self.Rhos[j])
            idx[np.where(condition)] = j
        idx[np.where((xs >= self.Rhos[-1]))] = j

        if len(xs) == 1:
            return idx[0]
        return idx

    def ns_from_xs(self, xs, continuity='left'):
        """Return refractive indices from regions in which (non-dimensional) xs
        lie."""
        return self.ns[self.region_index(xs, continuity=continuity)]

    # ------------ Matrices and Determinant (Eigenvalue) Functions -----------------

    def state_matrix(
        self,
        C,
        x,
        region_idx,
        field_type='TE',
        plane='Z',
        Ztype='standard',
    ):
        if field_type not in ["TE", "TM"]:
            raise ValueError("Must have field_type either TE or TM.")
        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        n = self.ns[region_idx]

        if plane == "Beta":
            Beta = C
            if Ztype == "standard":
                Z = np.sqrt(
                    self.K0**2 * n**2 - Beta**2, dtype=complex
                )
            elif Ztype == "imag":
                Z = 1j * np.sqrt(
                    Beta**2 - self.K0**2 * n**2, dtype=complex
                )
            else:
                raise ValueError("Ztype must be standard or imag.")

        elif plane == "Z":
            Z = self.Zi_from_Z0(C, n)

        else:
            Psi0 = C
            Psi = self.Psi_i_from_Psi0(Psi0, n)
            Z = self.K0 * n * np.sin(Psi)

        S = np.zeros(C.shape + (2, 2), dtype=np.complex128)

        # Offset = self.Offsets[region_idx]

        S[..., 0, 0] = np.exp(1j * Z * x)
        S[..., 0, 1] = np.exp(-1j * Z * x)

        S[..., 1, 0] = 1j * Z / self.K0 * np.exp(1j * Z * x)
        S[..., 1, 1] = -1j * Z / self.K0 * np.exp(-1j * Z * x)

        # models = self.regional_models[region_idx]

        # S[..., 0, 0] = models['M'](x, Z, region_idx, **self.model_kwargs)
        # S[..., 0, 1] = models['N'](x, Z, region_idx, **self.model_kwargs)

        # S[..., 1, 0] = 1 / self.K0 * models['dMdx'](x, Z, region_idx,
        #                                             **self.model_kwargs)
        # S[..., 1, 1] = 1 / self.K0 * models['dNdx'](x, Z, region_idx,
        #                                             **self.model_kwargs)

        if field_type == "TM":
            S[..., 1, 0] /= n**2
            S[..., 1, 1] /= n**2

        return S

    def transfer_matrix(
        self,
        C,
        interface_idx,
        field_type="TE",
        Ztype_left="standard",
        Ztype_right="standard",
        plane="Z",
        derivate=False,
    ):
        '''
        Matrix giving coefficients of field in next layer from previous.

        Parameters
        ----------
        C : complex
            Complex input value.  Keyword argument 'plane' determines whether
            this number represents transverse propagation constant 'Z0', 
            longitudinal propagation constant 'Beta' or radiation angle 'Psi0'.
        interface_idx : int
            Index of material interface for this transfer matrix.
        field_type : TYPE, optional
            DESCRIPTION. The default is "TE".
        Ztype_left : TYPE, optional
            DESCRIPTION. The default is "standard".
        Ztype_right : TYPE, optional
            DESCRIPTION. The default is "standard".
        plane : TYPE, optional
            DESCRIPTION. The default is "Z".
        derivate : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Notes
        -----
        Given an input 'C', manipulating the values of 'Z_left' and 'Z_right' 
        shifts between values of Z satisfying $Z^2 = K^2 - C^2$. Altering them
        can move branch cuts in the eigenvalue function when working with
        'plane=Beta'.
        '''

        if field_type not in ["TE", "TM"]:
            raise ValueError("Must have field_type either TE or TM.")

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        # if derivate and plane != 'Z':
        #     raise NotImplementedError("Derivative only for Z plane.")

        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        # derivative_scaling = 1

        n_left, n_right = self.ns[interface_idx-1], self.ns[interface_idx]
        Rho = self.Rhos[interface_idx]

        if plane == "Beta":
            Beta = C
            if Ztype_left == "standard":
                Z_left = np.sqrt(
                    self.K0**2 * n_left**2 - Beta**2, dtype=complex
                )
            elif Ztype_left == "imag":
                Z_left = 1j * np.sqrt(
                    Beta**2 - self.K0**2 * n_left**2, dtype=complex
                )
            else:
                raise ValueError("Ztype must be standard or imag.")

            if Ztype_right == "standard":
                Z_right = np.sqrt(
                    self.K0**2 * n_right**2 - Beta**2, dtype=complex
                )
            elif Ztype_right == "imag":
                Z_right = 1j * np.sqrt(
                    Beta**2 - self.K0**2 * n_right**2, dtype=complex
                )
            else:
                raise ValueError("Ztype must be standard or imag.")

            if derivate:
                pass
                Z0 = self.Zi_from_Beta(Beta, ni=self.n0)
            #     derivative_scaling = -Beta / Z0

        elif plane == "Z":
            Z0 = C
            Z_left = self.Zi_from_Z0(C, n_left)
            Z_right = self.Zi_from_Z0(C, n_right)

        else:
            Psi0 = C
            Z0 = self.K0 * self.n0 * np.sin(Psi0)
            # Z_left = self.Zi_from_Z0(Z0, n_left)
            # Z_right = self.Zi_from_Z0(Z0, n_right)
            Psi_left = self.Psi_i_from_Psi0(Psi0, n_left)
            Psi_right = self.Psi_i_from_Psi0(Psi0, n_right)
            Z_left = self.K0 * n_left * np.sin(Psi_left)
            Z_right = self.K0 * n_right * np.sin(Psi_right)

            # if derivate:
            #     derivative_scaling = self.K0 * self.n0 * np.cos(Psi0)

        M = np.zeros(C.shape + (2, 2), dtype=np.complex128)

        # L_offsets = self.Offsets[interface_idx-1]
        # R_offsets = self.Offsets[interface_idx]

        # Exp_00 = 1j * (Z_right * (Rho - R_offsets[0]) -
        #                Z_left * (Rho - L_offsets[0]))

        # Exp_01 = 1j * (Z_right * (Rho - R_offsets[0]) +
        #                Z_left * (Rho - L_offsets[1]))

        # Exp_10 = 1j * (Z_right * (Rho - R_offsets[1]) +
        #                Z_left * (Rho - L_offsets[0]))

        # Exp_11 = 1j * (Z_right * (Rho - R_offsets[1]) -
        #                Z_left * (Rho - L_offsets[1]))

        Exp_m = 1j * Rho * (Z_right - Z_left)
        Exp_p = 1j * Rho * (Z_right + Z_left)

        front_term = 1 / Z_right

        if field_type == "TM":
            front_term *= 1 / (2 * n_left**2)
            A_minus = Z_right * n_left**2 - Z_left * n_right**2
            A_plus = Z_right * n_left**2 + Z_left * n_right**2

        elif field_type == "TE":
            front_term *= 1 / 2
            A_minus = Z_right - Z_left
            A_plus = Z_right + Z_left

        M[..., 0, 0] = front_term * np.exp(-Exp_m) * A_plus
        M[..., 0, 1] = front_term * np.exp(-Exp_p) * A_minus

        M[..., 1, 0] = front_term * np.exp(Exp_p) * A_minus
        M[..., 1, 1] = front_term * np.exp(Exp_m) * A_plus

        # if field_type == "TM":
        #     n_l, n_r = n_left, n_right

        # elif field_type == "TE":
        #     n_l, n_r = 1, 1

        # models_left = self.regional_models[interface_idx-1]

        # M_l = models_left['M'](Rho, Z_left, interface_idx-1,
        #                        **self.model_kwargs)
        # Mp_l = models_left['dMdx'](Rho, Z_left, interface_idx-1,
        #                            **self.model_kwargs)
        # N_l = models_left['N'](Rho, Z_left, interface_idx-1,
        #                        **self.model_kwargs)
        # Np_l = models_left['dNdx'](Rho, Z_left, interface_idx-1,
        #                            **self.model_kwargs)

        # models_right = self.regional_models[interface_idx]

        # M_r = models_right['M'](Rho, Z_right, interface_idx,
        #                         **self.model_kwargs)
        # Mp_r = models_right['dMdx'](Rho, Z_right, interface_idx,
        #                             **self.model_kwargs)
        # N_r = models_right['N'](Rho, Z_right, interface_idx,
        #                         **self.model_kwargs)
        # Np_r = models_right['dNdx'](Rho, Z_right, interface_idx,
        #                             **self.model_kwargs)

        # W_r = M_r * Np_r - Mp_r * N_r
        # f = 1 / (W_r * n_l**2)

        # M[..., 0, 0] = f * (Np_r * M_l * n_l**2 - N_r * Mp_l * n_r**2)
        # M[..., 0, 1] = f * (Np_r * N_l * n_l**2 - N_r * Np_l * n_r**2)

        # M[..., 1, 0] = f * (-Mp_r * M_l * n_l**2 + M_r * Mp_l * n_r**2)
        # M[..., 1, 1] = f * (-Mp_r * N_l * n_l**2 + M_r * Np_l * n_r**2)

        if derivate:
            D = np.zeros_like(M)
            front_term = Z0 / (Z_right**2 * Z_left)
            D[..., 0, 0] = front_term * A_minus * (1 + 1j * Z_right * Rho)
            D[..., 0, 1] = -front_term * A_plus * (1 + 1j * Z_right * Rho)
            D[..., 1, 0] = -front_term * A_plus * (1 - 1j * Z_right * Rho)
            D[..., 1, 1] = front_term * A_minus * (1 - 1j * Z_right * Rho)
            M = M * D
            # pass

        return M

    def transmission_matrix(
        self,
        C,
        field_type="TE",
        Ztype_far_left="imag",
        Ztype_far_right="imag",
        up_to_region=-1,
        plane="Z",
        derivate=False,
    ):
        '''
        Matrix giving transverse field coefficients on outer right region of
        device from those on outer left region.

        Calculated by forming product of transfer matrices from each material
        interface. 

        Parameters
        ----------
        C : TYPE
            DESCRIPTION.
        field_type : TYPE, optional
            DESCRIPTION. The default is "TE".
        Ztype_far_left : TYPE, optional
            DESCRIPTION. The default is "imag".
        Ztype_far_right : TYPE, optional
            DESCRIPTION. The default is "imag".
        up_to_region : TYPE, optional
            DESCRIPTION. The default is -1.
        plane : TYPE, optional
            DESCRIPTION. The default is "Z".
        derivate : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        TYPE
            DESCRIPTION.
        '''

        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        T = np.zeros(C.shape + (2, 2), dtype=complex)
        T[..., :, :] = np.eye(2, dtype=complex)

        Rhos = self.Rhos

        if up_to_region >= 0:
            up_to_region = up_to_region - len(Rhos) + 1

        enum = range(1, len(Rhos) + up_to_region)

        if not derivate:
            for i in enum:

                if i == 1:
                    L, R = Ztype_far_left, "standard"

                elif i == len(Rhos) - 2:
                    L, R = "standard", Ztype_far_right

                else:
                    L, R = "standard", "standard"

                T = (
                    self.transfer_matrix(C, i, field_type=field_type,
                                         Ztype_left=L, Ztype_right=R,
                                         plane=plane,
                                         ) @ T
                )
            return T
        else:
            S = np.zeros_like(T)
            for i in enum:
                T = np.zeros(C.shape + (2, 2), dtype=complex)
                T[..., :, :] = np.eye(2, dtype=complex)

                for j in enum:
                    if j == i:
                        derivate = True
                    else:
                        derivate = False

                    if j == 1:
                        L, R = Ztype_far_left, "standard"

                    elif j == len(Rhos) - 2:
                        L, R = "standard", Ztype_far_right

                    else:
                        L, R = "standard", "standard"

                    T = (
                        self.transfer_matrix(C, j,
                                             field_type=field_type,
                                             Ztype_left=L, Ztype_right=R,
                                             plane=plane, derivate=derivate,
                                             ) @ T
                    )
                S += T
            return S

    def matching_matrix(
        self,
        Z,
        field_type="TE",
        Ztype='standard',
    ):
        d = len(self.Rhos)-2
        M = np.eye(2 * (d+1), dtype=complex)
        for i in range(1, d+1):
            Rho_i = self.Rhos[i]
            S_L = self.state_matrix(Z, Rho_i, i-1, field_type=field_type,
                                    Ztype=Ztype)[0]
            S_R = self.state_matrix(Z, Rho_i, i, field_type=field_type,
                                    Ztype=Ztype)[0]
            j = 1 + 2 * (i-1)
            M[j: j+2, j-1: j+1] = S_L[:]
            M[j: j+2, j+1: j+3] = -S_R[:]

        return M

    def eigenvalues(
        self,
        C,
        plane="Z",
        mode_type="guided",
        field_type="TE",
        Normalizer=None,
        sign=1,
        derivate=False,
        even=False,
        odd=False,
    ):
        '''
        Function whose zeroes correspond to the propagation constants
        for discrete modes of the waveguide (both leaky and guided).

        Parameters
        ----------
        C : TYPE
            DESCRIPTION.
        plane : TYPE, optional
            DESCRIPTION. The default is "Z".
        mode_type : TYPE, optional
            DESCRIPTION. The default is "guided".
        field_type : TYPE, optional
            DESCRIPTION. The default is "TE".
        Normalizer : TYPE, optional
            DESCRIPTION. The default is None.
        sign : TYPE, optional
            DESCRIPTION. The default is 1.
        derivate : TYPE, optional
            DESCRIPTION. The default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            DESCRIPTION.
        '''

        if field_type not in ["TE", "TM"]:
            raise ValueError("Must have field_type either TE or TM.")

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if mode_type not in ["guided", "leaky", "radiation"]:
            raise ValueError("Mode type must be 'guided', 'leaky' or \
'radiation'.")

        if even and odd:
            raise ValueError("Only one of even and odd can be set to true")

        if (even or odd) and (not self.is_symmetric):
            raise ValueError("Non-symmetric guides don't have even/odd modes.")

        if mode_type == "guided":
            if even or odd:
                s = -1
                M = self.transmission_matrix(
                    C, plane=plane, field_type=field_type, derivate=derivate,
                    up_to_region=int(len(self.Rhos)/2)-1)
                if odd:
                    s = 1
                return M[..., 0, 1] + s * M[..., 1, 1]
            M = self.transmission_matrix(
                C, plane=plane, field_type=field_type, derivate=derivate
            )
            return M[..., 1, 1]

        elif mode_type == "leaky":
            if even or odd:
                s = -1
                M = self.transmission_matrix(
                    C, plane=plane, field_type=field_type, derivate=derivate,
                    up_to_region=int(len(self.Rhos)/2))
                if odd:
                    s = 1
                return M[..., 0, 0] + s * M[..., 1, 0]
            M = self.transmission_matrix(
                C, plane=plane, field_type=field_type, derivate=derivate
            )
            return M[..., 0, 0]

        else:
            if Normalizer is None:
                Normalizer = self.normalizer("ours")
            return Normalizer.pole_locations(
                C, sign=sign, ft=field_type, plane=plane
            )

    def transmission_determinant(
        self,
        C,
        field_type="TE",
        up_to_region=-1,
        plane="Z",
    ):
        """Determinant of transmission matrix."""
        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if plane == "Beta":
            Z0 = self.Zi_from_Beta(C, ni=self.ns[0])
            Zd = self.Zi_from_Beta(C, ni=self.ns[up_to_region])
        elif plane == "Z":
            Z0 = self.Zi_from_Z0(C, ni=self.ns[0])
            Zd = self.Zi_from_Z0(C, ni=self.ns[up_to_region])
        else:
            Psi0 = C
            Psi_d = self.Psi_i_from_Psi0(Psi0, self.ns[up_to_region])
            Z0 = self.n0 * np.sin(Psi0)
            Zd = self.ns[up_to_region] * np.sin(Psi_d)

        base = Z0 / Zd
        if field_type == "TM":
            base *= self.ns[up_to_region] ** 2 / self.ns[0] ** 2
        return base

# --------------- Building Fields from Propagation Constants ------------------

    def coefficients(
        self,
        C,
        Normalizer=None,
        field_type="TE",
        mode_type="guided",
        plane="Z",
        sign="+1",
        up_to_region=-1,
        rounding=12,
        manual_coeffs=None,
        even=False,
        odd=False,
    ):
        """Return field coefficients given propagation constant C."""

        if field_type not in ["TE", "TM"]:
            raise ValueError("Must have field_type either TE or TM.")

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if mode_type not in ["guided", "leaky", "radiation"]:
            raise ValueError("Mode type must be 'guided', 'leaky' or \
'radiation.")

        if even and odd:
            raise ValueError("Only one of even and odd can be set to true")

        if (even or odd) and (not self.is_symmetric):
            raise ValueError("Non-symmetric guides don't have even/odd modes.")

        # Single scalar inputs need to be given at least a single dimension
        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        Ztype_far_left = "standard"
        Ztype_far_right = "standard"

        if mode_type in ["guided", "leaky"]:
            Ztype_far_left = "imag"
            Ztype_far_right = "imag"

        # Set up initial vector array M0.
        # This will contain initial vector for each coefficient array,
        # and also be overwritten to store the new coefficients
        # as we apply transfer matrix.
        M0 = np.zeros(C.shape + (2, 1), dtype=complex)

        # Work normalization into coefficients
        if mode_type == "guided":
            M = self.transmission_matrix(
                C,
                field_type=field_type,
                Ztype_far_left=Ztype_far_left,
                Ztype_far_right=Ztype_far_right,
                plane=plane,
                derivate=False,
            )

            M_prime = self.transmission_matrix(
                C,
                field_type=field_type,
                Ztype_far_left=Ztype_far_left,
                Ztype_far_right=Ztype_far_right,
                plane=plane,
                derivate=True,
            )

            d_prime, c = M_prime[..., 1, 1], M[..., 1, 0]
            check_idx = 1

            # Correct slicing for vectorization in C
            M0[..., 1, :][:, 0] = np.sqrt(1j * c / d_prime, dtype=complex)

            if plane == "Beta":
                M0 *= 1j

        elif mode_type == "leaky":
            M = self.transmission_matrix(
                C,
                field_type=field_type,
                Ztype_far_left=Ztype_far_left,
                Ztype_far_right=Ztype_far_right,
                plane=plane,
                derivate=False,
            )

            M_prime = self.transmission_matrix(
                C,
                field_type=field_type,
                Ztype_far_left=Ztype_far_left,
                Ztype_far_right=Ztype_far_right,
                plane=plane,
                derivate=True,
            )

            a_prime, b = M_prime[..., 0, 0], M[..., 0, 1]
            check_idx = 0

            # Correct slicing for vectorization in C
            M0[..., 0, :][:, 0] = np.sqrt(1j * b / a_prime, dtype=complex)

            if plane == "Beta":
                M0 *= 1j
        else:
            if Normalizer is None:
                Normalizer = OurMethod(self, field_type=field_type,
                                       plane=plane)

            M0 = Normalizer.normalization(C, sign=sign)
        if manual_coeffs is not None:
            M0[..., :, 0] = manual_coeffs

        Rhos = self.Rhos

        if up_to_region >= 0:
            up_to_region = up_to_region - len(Rhos) + 1

        Coeffs = np.zeros(
            C.shape + (2, len(Rhos) + up_to_region), dtype=complex
        )

        # Do this after setting array for Coeffs to keep it correct shape
        if even or odd:
            up_to_region = int(len(self.Rhos)/2) - len(Rhos) + 1

        # set first vectors in each coefficient array
        Coeffs[..., :, 0] = M0[..., :, 0]

        if even:
            Coeffs[..., 0, -1] = M0[..., 1, 0]
            Coeffs[..., 1, -1] = M0[..., 0, 0]
        if odd:
            Coeffs[..., 0, -1] = -M0[..., 1, 0]
            Coeffs[..., 1, -1] = -M0[..., 0, 0]

        Ztypes = ["standard" for i in range(len(self.Rhos) - 1)]
        Ztypes[0], Ztypes[-1] = Ztype_far_left, Ztype_far_right

        for i in range(1, len(Rhos) + up_to_region):
            T = self.transfer_matrix(
                C,
                i,
                field_type=field_type,
                Ztype_left=Ztypes[i - 1],
                Ztype_right=Ztypes[i],
                plane=plane,
            )

            # Apply T to coefficient vector to get coeffs in next region.
            M0 = T @ M0

            # Update coefficient arrays
            Coeffs[..., :, i] = M0[..., :, 0]

            # Complete corresponding data in mirrored regions for even/odd case
            if even:
                Coeffs[..., 0, -i-1] = M0[..., 1, 0]
                Coeffs[..., 1, -i-1] = M0[..., 0, 0]
            if odd:
                Coeffs[..., 0, -i-1] = -M0[..., 1, 0]
                Coeffs[..., 1, -i-1] = -M0[..., 0, 0]

        # Reduce dimension for length 1 inputs
        if len(C) == 1:
            Coeffs = Coeffs[0]

        # Round to avoid numerical blowup, noted in guided modes.
        # Fundamental had lower error and rounding=16 worked, but HOMs
        # had more noise and required rounding=12
        Coeffs = np.round(Coeffs, decimals=rounding)

        # Check for correct coefficients if mode type is guided or leaky
        if mode_type in ["guided", "leaky"] and len(C) == 1:
            if Coeffs.T[-1, check_idx] != 0:
                warn(
                    message="Provided mode type %s, but coefficients in outer \
region do not align with this. User may wish to check supplied \
propagation constant and/or rounding parameter." % mode_type
                )

        return Coeffs

    def regional_field(self, C, index, coeffs, Ztype="standard", plane="Z"):
        """Return field on one region of fiber."""

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        A, B = coeffs[:]
        K0 = self.K0
        n = self.ns[index]

        if plane == "Beta":
            Beta = C
            if Ztype == "standard":
                Z = np.sqrt(K0**2 * n**2 - C**2, dtype=complex)
            elif Ztype == "imag":
                Z = 1j * np.sqrt(C**2 - K0**2 * n**2, dtype=complex)
            else:
                raise ValueError("Ztype must be standard or imag.")
        elif plane == "Z":
            Z = self.Zi_from_Z0(C, ni=n)
            Beta = self.Beta_from_Zi(Z, ni=n)
        else:
            Psi0 = C
            Beta = K0 * self.n0 * np.cos(Psi0)
            Z = K0 * n * np.sin(self.Psi_i_from_Psi0(Psi0, n))

        # models = self.regional_models[index]
        # M, N = models['M'], models['N']

        def F(xs, zs=None):

            try:
                len(xs)
                xs = np.array(xs)
            except TypeError:
                xs = np.array([xs])

            if len(xs.shape) > 1:
                raise ValueError("Please provide single dimension arrays for \
xs and zs to levarage product nature of fields.")
            ys = A * np.exp(1j * Z * xs) + B * np.exp(-1j * Z * xs)
            # ys = A * M(xs, Z, index, **self.model_kwargs) + \
            #     B * N(xs, Z, index, **self.model_kwargs)

            if zs is not None:
                try:
                    len(zs)
                    zs = np.array(zs)
                except TypeError:
                    zs = np.array([zs])

                if len(zs.shape) > 1:
                    raise ValueError("Please provide single dimension arrays \
for xs and zs to levarage product nature of fields.")

                ys = np.outer(np.exp(1j * Beta * zs), ys)

            return ys

        return F

    def fields(
        self,
        C,
        Normalizer=None,
        field_type="TE",
        mode_type="guided",
        sign="+1",
        rounding=12,
        plane="Z",
        manual_coeffs=None,
        user_coeffs=None,
        even=False,
        odd=False,
    ):
        """Return fields at propagation constant given by 'C' in plane
        determined by 'plane'. Vectorized in x,and z but not in C."""

        # Give C at least one dimension
        try:
            len(C)
            C = np.array(C, dtype=complex)
        except TypeError:
            C = np.array([C], dtype=complex)

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if even and odd:
            raise ValueError("Only one of even and odd can be set to true")

        if (even or odd) and (not self.is_symmetric):
            raise ValueError("Non-symmetric guides don't have even/odd modes.")

        if user_coeffs is not None:
            M = user_coeffs
        else:
            M = self.coefficients(
                C,
                Normalizer=Normalizer,
                mode_type=mode_type,
                field_type=field_type,
                sign=sign,
                rounding=rounding,
                plane=plane,
                manual_coeffs=manual_coeffs,
                even=even,
                odd=odd,
            ).T

        Ztypes = ["standard" for i in range(len(self.Rhos) - 1)]

        if mode_type in ["guided", "leaky"]:
            Ztypes[0], Ztypes[-1] = "imag", "imag"

        if plane == "Beta":
            Beta = C
        elif plane == "Z":
            Beta = self.Beta_from_Zi(C)
        else:
            Beta = self.K0 * self.n0 * np.cos(C)

        # Get list of functions, one for each region
        Fs = []
        for i in range(len(self.Rhos) - 1):
            Fs.append(
                self.regional_field(C, i, M[i], Ztype=Ztypes[i], plane=plane)
            )

        # Return piecewise defined function
        def F(xs, zs=np.array([0])):

            # Give xs at least one dimension (for np.piecewise function)
            try:
                len(xs)
                xs = np.array(xs)
            except TypeError:
                xs = np.array([xs])

            if len(xs.shape) > 1:
                raise ValueError("Please provide single dimension arrays \
for xs and zs to levarage product nature of fields.")

            # Set up piecewise condition list
            conds = self.condition_list(xs)
            ys = np.piecewise(xs + 0j, conds, Fs)  # 0j gives complex output

            try:
                len(zs)
                zs = np.array(zs)
            except TypeError:
                zs = np.array([zs])
            if len(zs.shape) > 1:
                raise ValueError("Please provide single dimension \
arrays for xs and zs to levarage product nature of fields.")
            ys = np.outer(np.exp(1j * Beta * zs), ys)

            if len(xs) == 1 and len(zs) == 1:
                return ys[0, 0]
            elif len(xs) == 1 and len(zs) > 1:
                return ys[:, 0]
            elif len(xs) > 1 and len(zs) == 1:
                return ys[0]
            else:
                return ys

            return ys

        return Field(F)

    def vectorized_fields(
        self,
        C,
        x,
        z=0,
        Normalizer=None,
        field_type="TE",
        mode_type="radiation",
        sign="+1",
        plane="Z",
        Coeffs=None,
        even=False,
        odd=False,
    ):
        """Return value of field with propagation constant C at x, z in plane
        determined by 'plane'. Vectorized in C.

        Note: a goal is to fully vectorize the fields both in C, x and z."""

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        if even and odd:
            raise ValueError("Only one of even and odd can be set to true")

        if (even or odd) and (not self.is_symmetric):
            raise ValueError("Non-symmetric guides don't have even/odd modes.")

        i = self.region_index(x)
        # Offset = self.Offsets[i]

        if plane == "Beta":
            Beta = C
            Zi = np.array(self.Zi_from_Beta(C, ni=self.ns[i]))

        elif plane == "Z":
            Zi = np.array(self.Zi_from_Z0(C, ni=self.ns[i]))
            Beta = self.Beta_from_Zi(Zi, ni=self.ns[i])

        else:
            Beta = self.K0 * self.n0 * np.cos(C)
            Zi = (
                self.K0
                * self.ns[i]
                * np.sin(self.Psi_i_from_Psi0(C, self.ns[i]))
            )

        if Coeffs is None:
            Coeffs = self.coefficients(
                C,
                Normalizer=Normalizer,
                up_to_region=i,
                mode_type=mode_type,
                field_type=field_type,
                sign=sign,
                plane=plane,
                even=even,
                odd=odd,
            )

        # models = self.regional_models[i]
        # M, N = models['M'], models['N']

        return (Coeffs[..., 0, i] * np.exp(1j * Zi * x)
                + Coeffs[..., 1, i] * np.exp(-1j * Zi * x)
                ) * np.exp(1j * Beta * z)

# --------------Automated Mode Finding and Field Construction -----------------

    def get_guided_propagation_constants(self, Nref=3000, rounding=8,
                                         field_type='TE'):
        """
        Calculate the structure's guided mode propagation constants.

        Parameters
        ----------
        Nref : int, optional
            Number of points in interval for guided mode search. The default is
            3000.
        """

        if self.Z_max_guided.imag == 0:
            warn("Structure does not support guided modes.  Setting empty \
lists for attribute guided_prop_constants.")
            self.guided_prop_constants = []
            return

        def F(x): return -np.log(np.abs(self.eigenvalues(x*1j))**2)

        xs = np.linspace(0.000001, 1.05 * self.Z_max_guided.imag, Nref)
        ys = (F(xs[1:])-F(xs[:-1])) / (xs[1:] - xs[:-1])

        sign_changes = xs[1:][np.where((np.sign(ys[:-1] * ys[1:])) < 0)[0]]
        guesses = sign_changes[::2]

        Zs = []
        for (i, guess) in enumerate(guesses):
            try:
                Z = newton(self.eigenvalues, guess*1j)[0]
                Zs.append(Z)
            except RuntimeError:
                print("Runtime Error at index "+str(i))
                pass

        Zs = np.array(Zs)
        Zs = 1j * np.unique(Zs.imag)  # Remove possible repeats
        Zs = np.flip(Zs)  # Invert order so fundamental comes first

        self.guided_prop_constants = Zs

    def get_guided_modes(self, Nref=3000, rounding=8, field_type='TE'):
        """
        Calculate guided modes of the structure.

        Parameters
        ----------
        Nref : int, optional
            Number of points in interval for guided mode search. The default is
            3000.
        """
        if not hasattr(self, 'guided_prop_constants'):
            self.get_guided_propagation_constants(Nref=Nref,
                                                  field_type=field_type)
        Zs = self.guided_prop_constants

        if len(Zs) == 0:
            warn("Structure does not support guided modes.  Setting empty \
list for attribute guided_modes and None  guided_regional_coeffs.")
            self.guided_modes = []
            self.guided_regional_coeffs = None
            return

        Coeffs = self.coefficients(Zs,
                                   mode_type='guided',
                                   field_type=field_type)
        self.guided_regional_coeffs = Coeffs
        self.guided_modes = [self.fields(Z, rounding=rounding) for Z in Zs]

# ------------------------Steepest Descent Path Methods------------------------

    def sdp(self, s, **kwargs):
        return sdp(self, s, **kwargs)

    def sdp_derivative(self, s, **kwargs):
        return sdp_derivative(self, s, **kwargs)
# ------------------------------ Plotting -------------------------------------

    def plot_refractive_index(self, **kwargs):
        return plot_refractive_index(self, **kwargs)

    def plot_points(self, Cs, **kwargs):
        return plot_points(Cs, **kwargs)

    def plot_field_1d(self, F, *F_args, **kwargs):
        return plot_field_1d(self, F, *F_args, **kwargs)

    def add_1d_plot(self, F, *F_args, **kwargs):
        return add_1d_plot(self, F, *F_args, **kwargs)

    def plot_field_2d(self, F, *F_args, **kwargs):
        return plot_field_2d(self, F, *F_args, **kwargs)

    def add_2d_plot(self, F, *F_args, **kwargs):
        return add_2d_plot(self, F, *F_args, **kwargs)

    def plot_field_2d_surface(self, F, *F_args, **kwargs):
        return plot_field_2d_surface(self, F, *F_args, **kwargs)

    def add_2d_plot_surface(self, F, *F_args, **kwargs):
        return add_2d_plot_surface(self, F, *F_args, **kwargs)

    def eigenvalue_plot(self, rmin, rmax, imin, imax, **kwargs):
        return eigenvalue_plot(self, rmin, rmax, imin, imax, **kwargs)

    def eigenvalue_plot_1d(self, contour, **kwargs):
        return eigenvalue_plot_1d(self, contour, **kwargs)


A = Slab()

# A.transmission_matrix(10j, derivate=True)
