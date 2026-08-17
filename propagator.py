import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from .normalizers import OurMethod
from .contours import real_contour
from .fields import Field
from .spacewave import space_wave_approx


class Propagator():
    def __init__(
        self,
        Slab,
        f0=None,
        window=None,
        exact_transform=None,
        exact_kwargs={},
        field_type="TE",
        plane="Z",
        Normalization_Class=OurMethod,
        compute_class_I=True,
        compute_class_II=True,
        integration_args={},
    ):
        '''
        Class for computing induced fields in slab waveguides.

        Given a guide "Slab" and either input field "f0" or "exact_transform",
        an instance of this class enables computation of the resultant field in
        the guide by forming the radiation integral over intervals of the real
        line. It also allows for the extension of the radiation integral to
        the complex plane, giving users the ability to verify the leaky mode
        representation of the radiation field.

        Instantiation requires either a single variable input field 'f0' and 
        interval of transverse domain 'window' or exact_transform function.

        Parameters
        ----------
        Slab : Slab
            An instance of the Slab class, representing a slab waveguide.
        f0 : function, optional
            Input field function. Single variable function of the tranverse
            variable x.  The default is None.
        window : list or array, optional
            Support window for f0.  Outside this interval f0 is zero. Required
            if 'f0' is provided. Default is None.
        exact_transform : function, optional
            Exact radiation tranform function. Single variable function of the
            radiation variable of integration Z.  The default is None.
        exact_kwargs : dict, optional
            Keyword variables for exact transform. The default is {}.
        field_type : str, optional
            Desired field type of solution, either tranverse electric 'TE' or
            transverse magnetic 'TM. The default is "TE".
        plane : str, optional
            String indicating which variable in the integral is taken as 
            the operating variable. Options are "Beta", "Psi" and "Z". 
            The default is "Z".
        Normalization_Class : class, optional
            Class for normalizing radiation modes. The default is OurMethod.
        compute_class_I : bool, optional
            Whether or not to compute the radiation transform using class I 
            radiation modes. The default is True.
        compute_class_II : bool, optional
            Whether or not to compute the radiation transform using class I 
            radiation modes. The default is True.
        integration_args : dict
            Extra arguments for scipy's quad_vec routine. Default is {}.
        '''
        both = f0 is None and exact_transform is None
        neither = f0 is not None and exact_transform is not None

        if both or neither:
            raise ValueError("Must provide precisely one of either \
input field f0 or exact transform function.")

        if not compute_class_I and not compute_class_II:
            raise ValueError("At least one class of 'compute_class_I' and \
'compute_class_II' must be set to True.'")

        self.Slab = Slab

        self.field_type = field_type
        self.ft = field_type
        self.plane = plane

        self.compute_class_I = compute_class_I
        self.compute_class_II = compute_class_II

        self.Normalizer = Normalization_Class(self.Slab,
                                              field_type=self.field_type,
                                              plane=self.plane)

        # Set transform function.
        if f0 is not None:
            if window is None:
                raise ValueError("User must give support window [Lx, Rx] \
when providing input field f0.")
            self.Lx, self.Rx = window[0], window[1]
            self.f0 = f0

            def func(C_lambda, sign=1, Coeffs=None):

                return self.radiation_transform(
                    C_lambda,
                    sign=sign,
                    Coeffs=Coeffs,
                    **integration_args,
                )

            self.transform = func

        else:
            def exact_func(C_lambda): return exact_transform(
                C_lambda, **exact_kwargs,
            )
            self.transform = exact_func

        # Initiate list of complex contours owned by this propagator
        self.contours = []

        # Check for complex refractive indices and warn re splitting up
        # propagating and evanescent parts of the radiation field.
#         self.split_prop_ev = split_prop_ev
#         if self.Slab.ns[0].imag and split_prop_ev:
#             warn("Outer refractive index complex but split_prop_ev set to \
# True; radiation field cannot strictly be separated in this way for complex \
# n0. Proceeding as normal but user should be aware.")

        # Define Field type callable attributes
        def Propagating(xs, zs=np.array([0]), method="simpsons"):
            return self(xs, zs=zs, method="simpsons",
                        portion='propagating')

        self.Propagating = Field(Propagating)

        def Evanescent(xs, zs=np.array([0]), method="simpsons"):
            return self(xs, zs=zs, method="simpsons",
                        portion='evanescent')

        self.Evanescent = Field(Evanescent)

        def Radiation(xs, zs=np.array([0]), method="simpsons"):
            return self(xs, zs=zs, method="simpsons",
                        portion='radiation')

        self.Radiation = Field(Radiation)

        def SDP(xs, zs=np.array([0]), method="simpsons"):
            return self(xs, zs=zs, method="simpsons",
                        portion='sdp')

        self.SDP = Field(SDP)

        # Get guided field
        Guided = self.get_guided_field(f0, self.Lx, self.Rx)
        self.Guided = Field(Guided)

        self.Full = Field(self)

    def verify_contour(self, contour):
        if (len(set([len(contour['Cs']),
                     len(contour['Ss']),
                     len(contour['dCdS'])])) != 1):
            raise ValueError(
                "Length of Cs, dCdS and Ss must be equal."
            )

    def order_contours(self,):
        '''
        Order self.contours.

        Orders real contours from left to right on real line.

        Returns
        -------
        None
        '''

        L_old = self.contours[0]['Cs'].real[0]

        for idx in range(len(self.contours)):
            # Only order real contours
            if self.contours[idx]['contour_type'] != 'real':
                continue
            L_new = self.contours[idx]['Cs'].real[0]
            if L_new < L_old:
                self.contours.insert(0, self.contours.pop(idx))
                L_old = L_new

    def split_contour(self, idx, split_indices, remove_indices=None):
        '''
        Split up contour into parts based on 'split_indices'.

        Parameters
        ----------
        idx : int
            Index of contour in contour list.
        split_indices : list or array of ints
            Indices at which to split up contour and contour data.
        remove_indices : list or array of ints
            Indices of splits to remove.  Indexing based on list resulting from
            np.split function.
        Returns
        -------
        None
        '''

        # Record shifting of indices as we add and remove contours
        index_shift = 0

        # Get contour (removing it from list) and array length
        contour = self.contours.pop(idx)

        # Adjust outer loop index to account for removal
        index_shift -= 1

        # Split up contour into new dictionaries
        Ds = [{} for i in range(len(split_indices)+1)]
        for key, val in contour.items():
            if key == 'contour_type' or key == 'radiation_type':
                for D in Ds:
                    D[key] = contour[key]
                continue
            array_list = np.split(contour[key], split_indices)
            for k, D in enumerate(Ds):
                D[key] = array_list[k]

        # Save new contours and data in correct places
        for l, D in enumerate(Ds):
            # Remove trivial splits
            if len(D['Cs']) == 0:  # trivial split
                continue
            # Remove (local) index if in remove_indices
            if l in remove_indices:
                l -= 1  # adjust (local) index to account for removals
                continue
            # Add new contours to end of list (avoiding indexing shift)
            self.contours.append(D)

        return index_shift

    def overlap_possible(self, contour1, contour2):

        ct1, ct2 = contour1['contour_type'], contour2['contour_type']

        # Only compare same type contours
        if ct1 != ct2:
            return False

        # Vertical contours must have same real part
        if ct1 == 'vertical':
            if ct1['Cs'][0].real != ct2['Cs'][0].real:
                return False

        # Horizontal contours must have same imaginary part
        if ct1 == 'horizontal':
            if ct1['Cs'][0].imag != ct2['Cs'][0].imag:
                return False

        # Circular contours must have same center and radius
        if ct1 in ['circle', 'half_circle']:
            if ct1['center'] != ct2['center']:
                return False
            if ct1['radius'] != ct2['radius']:
                return False

        return True

    def check_overlap(self, new_contour):
        """
        Check for overlap between new_contour and previous ones.

        Removes or truncates previous contour if overlap is detected.  Bridging
        contours are added, if necessary, in order to ensure interval spacing
        conforms to requirment of Simpon's rule (homogeneous spacing with odd
        number of intervals).'

        Parameters
        ----------
        new_contour : dict
            Dictionary containing new contour information.
        """

        new_contour_type = new_contour['contour_type']
        new_Cs = new_contour['Cs']

        same_contour_flag = False
        index_shift = 0

        for idx in range(len(self.contours)):

            contour = self.contours[idx + index_shift]

            # Only compare contours if same type
            if not self.overlap_possible(contour, new_contour):
                continue

            if new_contour_type == 'real':
                old_Cs = contour['Cs']

                # Check if contour already exists
                if len(new_Cs) == len(old_Cs) and np.all(new_Cs == old_Cs):
                    same_contour_flag = True
                    return same_contour_flag

                L_new, R_new = new_Cs[0], new_Cs[-1]
                L_old, R_old = old_Cs[0], old_Cs[-1]

                if R_old <= L_new or L_old >= R_new:
                    # New and old intervals disjoint, no trimming necessary
                    continue

                # Find beginning and end of overlap (may extend to end of
                # arrays).
                split_indices = np.where((old_Cs <= R_new) *
                                         (old_Cs >= L_new))[0][[0, -1]]

                # Adjust indices if overlap aligns exactly with an element
                # in old contour
                if L_new in old_Cs:
                    split_indices[0] += 1

                if R_new in old_Cs:
                    split_indices[-1] -= 1

                split_indices[-1] += 1  # accounts for np.split behavior

                # Adjust indices to ensure remaining arrays have odd numbers
                # of points, either 0 or 3 and above (for Simpson's rule).
                L, R = split_indices[:]

                # Note calculations for L below are all one higher than would
                # be expected based on zero indexing. This is due to the
                # behavior of np.split
                if L <= 2:
                    L = 0
                elif L % 2 == 0:
                    L -= 1
                if (len(old_Cs) - R) <= 2:
                    R = len(old_Cs)
                elif (len(old_Cs) - R) % 2 == 0:
                    R += 1

                split_indices[:] = [L, R]

                # Split up old contour, removing overlap with new contour.
                new_shift = self.split_contour(idx+index_shift, split_indices,
                                               remove_indices=[1])
                index_shift += new_shift

                # Add bridging contours if necessary
                if L > 0 and old_Cs[L-1] < L_new:
                    bridge_contour = real_contour(old_Cs[L-1], L_new, 3)
                    self.add_contour(bridge_contour, check_overlap=True)
                    # index_shift += 1
                if len(old_Cs) > R and R_new < old_Cs[R]:
                    bridge_contour = real_contour(R_new, old_Cs[R], 3)
                    self.add_contour(bridge_contour, check_overlap=True)
                    # index_shift += 1

        return same_contour_flag

    def add_contour(self, contour, check_overlap=True):
        '''
        Add a complex contour and associated data to the propagator.

        Performs radiation transform of input function along contour and
        collects fields.

        Parameters
        ----------
        contour : dict
            Dictionary of contour data.
        check_overlap : bool, optional
            If True, checks for overlap with previous contours and deletes or
            truncates them appropriately. The default is True.

        Returns
        -------
        None
        '''

        # Verify data in contour
        self.verify_contour(contour)

        # Check for overlap with other contours and trim them/truncate data
        if check_overlap:

            same_contour_flag = self.check_overlap(contour)

            # Exit if contour already exists.
            if same_contour_flag:
                print("Contour already exists.")
                return

        Cs = contour['Cs']

        # Add data to contour
        if self.compute_class_I:

            # Pre-compute regional coefficients for vectorized tranform
            Coeffs = self.Slab.coefficients(Cs, mode_type='radiation',
                                            field_type=self.field_type,
                                            Normalizer=self.Normalizer,
                                            plane=self.plane, sign=1)

            # Get class I radiation transform of input field
            contour['alphas_I'] = self.transform(Cs, sign=1, Coeffs=Coeffs)

            # Get class I radiation mode fields
            F_I = []
            for C in Cs:
                F = self.Slab.fields(C, mode_type="radiation",
                                     field_type=self.field_type, sign=1,
                                     Normalizer=self.Normalizer,
                                     plane=self.plane
                                     )
                F_I.append(F)
            contour['fields_I'] = F_I

        if self.compute_class_II:

            # Pre-compute regional coefficients for vectorized tranform
            Coeffs = self.Slab.coefficients(Cs, mode_type='radiation',
                                            field_type=self.field_type,
                                            Normalizer=self.Normalizer,
                                            plane=self.plane, sign=-1)

            # Get class II radiation transform of input field
            contour['alphas_II'] = self.transform(Cs, sign=-1, Coeffs=Coeffs)

            # Get class II radiation mode fields
            F_II = []
            for C in Cs:
                F = self.Slab.fields(C, mode_type="radiation",
                                     field_type=self.field_type, sign=-1,
                                     Normalizer=self.Normalizer,
                                     plane=self.plane
                                     )
                F_II.append(F)
            contour['fields_II'] = F_II

        # Append contour to end of list
        self.contours.append(contour)

        # Split contours that span self.Slab.Z_evanescent
        self.separate_prop_ev()

        # Mark radiation type
        self.mark_prop_ev()

        self.N_contours = len(self.contours)

    def separate_prop_ev(self,):
        '''
        Split any contours spanning propagating and evanescent regions into
        separate parts.
        '''
        for contour in self.contours:

            if contour['contour_type'] == 'real':
                Cs = contour['Cs']
                Z_ev = self.Slab.Z_evanescent

                if ((Cs[0] < Z_ev) * (Cs[-1] > Z_ev)):

                    # Find leftmost index under Z_evanescent and adjust for
                    # Simpson's rule
                    L = np.where(Cs < Z_ev)[0][-1]
                    if L <= 2:
                        L = 0
                    elif L % 2 == 0:
                        L -= 1
                    L_C = Cs[L-1]

                    # Create contour going from that point to Z_evanescent
                    # (again conforming to Simpson's rule).
                    splitting_contour = real_contour(L_C, Z_ev, 3)

                    # Add the contour.  This will split the spanning contour
                    # and create the appropriate right bridge.
                    self.add_contour(splitting_contour)

    def mark_prop_ev(self,):
        '''
        Mark real contours as propagating or evanescent.
        '''
        for contour in self.contours:
            if contour["contour_type"] == 'real':

                if np.all(contour['Cs'] <= self.Slab.Z_evanescent):
                    contour['radiation_type'] = 'propagating'
                else:
                    contour['radiation_type'] = 'evanescent'
            else:
                contour['radiation_type'] = 'mixed'

    def radiation_transform(
        self,
        Cs,
        sign=1,
        Coeffs=None,
        **intargs,
    ):
        """
        Return transform coefficients (alphas) for input field f0 from
        Cs using scipy vectorized integration techniques.
        """

        def integrand(x, C, sign=sign, Coeffs=Coeffs):
            return self.f0(x) * self.Slab.vectorized_fields(
                C,
                x,
                field_type=self.field_type,
                mode_type='radiation',
                Normalizer=self.Normalizer,
                plane=self.plane,
                sign=sign,
                Coeffs=Coeffs,
            )

        return quad_vec(
            integrand, self.Lx, self.Rx, args=(Cs, sign, Coeffs), **intargs
        )[0]

    def quadrature(
        self,
        xs,
        zs,
        alphas,
        Fs,
        contour,
        method='simpson'
    ):

        Cs, dCdS, Ss = contour['Cs'], contour['dCdS'], contour['Ss']
        dS = Ss[1:] - Ss[:-1]
        Nz = len(Cs)

        if method == "left_endpoint":
            ys = sum(
                [
                    alphas[i] * Fs[i](xs, zs) * dCdS[i] * dS[i]
                    for i in range(Nz - 1)
                ]
            )

        elif method == "right_endpoint":
            ys = sum(
                [
                    alphas[i + 1]
                    * Fs[i + 1](xs, zs)
                    * dCdS[i + 1]
                    * dS[i]
                    for i in range(Nz - 1)
                ]
            )

        elif method == "trapezoid":
            ys = alphas[0] * Fs[0](xs, zs) * dCdS[0] * dS[0]
            ys += sum(
                [
                    alphas[i]
                    * Fs[i](xs, zs)
                    * dCdS[i]
                    * (dS[i] + dS[i - 1])
                    for i in range(1, Nz - 1)
                ]
            )
            ys += alphas[-1] * Fs[-1](xs, zs) * dCdS[-1] * dS[-1]
            ys *= 1 / 2

        elif method == "simpsons":
            if len(Cs) % 2 != 1:
                raise ValueError("Contour must have odd number of \
points points (even number of intervals) for Simpsons rule.")

            set_dS = set(np.round(dS, decimals=12))
            if len(set_dS) != 1:
                raise ValueError("Simpsons rule requires equally \
spaced spaced intervals.")

            dS = set_dS.pop()
            upper = int((len(Cs) - 1) / 2) + 1
            ys = alphas[0] * Fs[0](xs, zs) * dCdS[0]
            ys += 4 * sum(
                [
                    alphas[2 * i - 1]
                    * Fs[2 * i - 1](xs, zs)
                    * dCdS[2 * i - 1]
                    for i in range(1, upper)
                ]
            )
            ys += 2 * sum(
                [
                    alphas[2 * i] * Fs[2 * i](xs, zs) * dCdS[2 * i]
                    for i in range(1, upper - 1)
                ]
            )
            ys += alphas[-1] * Fs[-1](xs, zs) * dCdS[-1]
            ys *= 1 / 3 * dS

        return ys

    def __call__(self, xs, zs=np.array([0]), method="simpsons",
                 portion='full'):
        """
        Propagate radiation field of input function self.f0 (or exact
        transform).

        Parameters
        ----------
        xs : float or float array
            Value of x along direction transverse to propagation.
        zs : float or float array
            Value of z along direction of propagation.
        method: str, optional
            Quadrature method.  May be left_endpoint, right_endpoint, simpsons
            or trapezoidal. Default is simpsons.

        Returns
        -------
        ys: complex or complex array
            Field strengths at input xs and ys.
        """
        result = 0

        if portion == 'radiation' or portion == 'full':
            contours = self.contours
        elif portion == 'propagating':
            contours = []
            for c in self.contours:
                if c['radiation_type'] == 'propagating':
                    contours.append(c)
        elif portion == 'evanescent':
            contours = []
            for c in self.contours:
                if c['radiation_type'] == 'evanescent':
                    contours.append(c)
        elif portion == 'sdp':
            contours = []
            for c in self.contours:
                if c['contour_type'] == 'sdp':
                    contours.append(c)
        else:
            raise ValueError("Portion must be 'full', 'radiation', 'sdp', \
'propagating' or 'evanescent'.")

        for contour in contours:

            if self.compute_class_I:
                alphas, Fs = contour['alphas_I'], contour['fields_I']
                ys = self.quadrature(xs, zs, alphas, Fs, contour,
                                     method=method)
                result += ys

            if self.compute_class_II:
                alphas, Fs = contour['alphas_II'], contour['fields_II']
                ys = self.quadrature(xs, zs, alphas, Fs, contour,
                                     method=method)
                result += ys

        if portion == 'full':
            result += self.Guided(xs, zs)
        return result

    def get_guided_field(self, f0, Lx, Rx, Nref=3000, field_type='TE',
                         return_coeffs=False, **intargs):
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
            Number of points in interval for guided mode findig. The default is
            3000.

        Returns
        -------
        callable
            Guided field induced in guide by f0.
        array
            Coefficients of guided field (integral of f0 against each guided
            mode).
        """
        try:
            Zs = self.Slab.guided_mode_propagation_constants
        except AttributeError:
            self.Slab.get_guided_modes(Nref, field_type=field_type)
            Zs = self.Slab.guided_mode_propagation_constants

        def integrand(x, C):
            return f0(x) * self.Slab.vectorized_fields(
                C,
                x,
                mode_type="guided",
                field_type=field_type,
                Coeffs=self.Slab.guided_mode_regional_coefficients
            )

        coeffs = quad_vec(
            integrand, Lx, Rx, args=(Zs,), **intargs
        )[0]

        Fs = self.Slab.guided_modes

        def Guided(x, z): return sum([c * F(x, z)
                                     for (c, F) in zip(coeffs, Fs)])

        if return_coeffs:
            return np.array(coeffs)
        else:
            return Guided

    def plot_transform(
        self,
        indices=None,
        figsize=(11, 4),
        part="real",
        marker="o",
        markersize=3,
        ax=None,
        plot_axis=True,
        axis_linewidth=1.25,
        legend_fontsize=12,
        color_I=None,
        color_II=None,
        plot_Z_evanescent=True,
        **linekwargs,
    ):
        plt.close('all')
        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            fig = plt.gcf()

        if indices is None:
            indices = slice(0, self.N_contours)

        contours = np.array(self.contours)[indices]

        for i in range(len(contours)):

            contour = contours[i]
            bools = [self.compute_class_I, self.compute_class_II]

            for idx, boolean in enumerate(bools):

                if idx == 0:
                    alphas = contour['alphas_I']
                    color = color_I
                    if i == 0:
                        label = 'class_I %s' % part
                    else:
                        label = None
                else:
                    alphas = contour['alphas_II']
                    color = color_II
                    if i == 0:
                        label = 'class_II %s' % part
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

                        flag = (np.min(xs) <= self.Slab.Z_evanescent.real) * \
                            (np.max(xs) >= self.Slab.Z_evanescent.real)

                        if flag:
                            plt.axvline(self.Slab.Z_evanescent.real, lw=.75,
                                        ls='--', color='gray')

                    if plot_axis:

                        ax.axhline(0, color="gray",
                                   linewidth=axis_linewidth)

                        near_y_axis = (min(np.abs(xs)) / np.ptp(xs)) <= 0.1
                        if near_y_axis:
                            ax.axvline(0, color="gray",
                                       linewidth=axis_linewidth)

            ax.legend(fontsize=legend_fontsize)

        return fig, ax

    def space_wave_approx(self, f0, **kwargs):
        return space_wave_approx(self.Slab, f0, **kwargs)
