import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
from ._normalizers import OurMethod
from ._contours import real_contour


class Propagator:
    def __init__(
        self,
        Slab,
        f0=None,
        Lx=None,
        Rx=None,
        exact_transform=None,
        exact_kwargs={},
        field_type="TE",
        Normalization_Class=OurMethod,
        plane="Z",
        compute_class_I=True,
        compute_class_II=True,
        **integration_args,
    ):

        both = f0 is None and exact_transform is None
        neither = f0 is not None and exact_transform is not None

        if both or neither:
            raise ValueError("Must provide precisely one of either \
input field f0 or exact transform function.")

        if not compute_class_I and not compute_class_II:
            raise ValueError("At least one class of 'compute_class_I' and \
'compute_class_II' must be set to True.'")

        self.Slab = Slab

        if Lx is None:
            print("No left x integral limit given (Lx). Defaulting to left \
end of computational domain.  This will likely slow things down a lot, so it \
is recommended to reinstantiate using more accurate bound on the support of \
f0.")
            self.Lx = self.Slab.Rhos[0]
        else:
            self.Lx = Lx
        if Rx is None:
            print("No right x integral limit given (Rx). Defaults to right \
end of computational domain.  This will likely slow things down a lot, so it \
is recommended to reinstantiate using more accurate bound on the support of \
f0.")
            self.Rx = self.Slab.Rhos[-1]
        else:
            self.Rx = Rx

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
            self.f0 = f0

            def func(C_lambda, sign=1, Coeffs=None):

                return self.radiation_transform(
                    C_lambda,
                    sign=sign,
                    Coeffs=Coeffs,
                    **integration_args
                )

            self.transform = func

        else:
            def exact_func(C_lambda): return exact_transform(
                C_lambda, **exact_kwargs
            )
            self.transform = exact_func

        # Initiate list of complex contours owned by this propagator
        self.contours = []

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
            if key == 'contour_type':
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
            # Add new contours to start of list (thus avoiding indexing shift)
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

                split_indices[-1] += 1  # accounts for np.split behavior

                # Adjust indices to ensure remaining arrays have odd numbers
                # of points, either 0 or 3 and above (for Simpson's rule).
                L, R = split_indices[:]
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
                if L > 0:
                    bridge_contour = real_contour(old_Cs[L-1], L_new, 3)
                    self.add_contour(bridge_contour, check_overlap=True)
                    # index_shift += 1
                if len(old_Cs) > R:
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
            return self.f0(x) * self.Slab.fields_vec(
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

    def quadrature(self, xs, zs, alphas, Fs, contour, method='simpson'):

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

    def __call__(self, xs, zs=np.array([0]), method="simpsons"):
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

        for contour in self.contours:

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

        return result

    def plot_transform(
        self,
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

        for i in range(len(self.contours)):

            contour = self.contours[i]
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
