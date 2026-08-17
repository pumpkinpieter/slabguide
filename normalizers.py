import numpy as np


class PaperMethod:
    def __init__(self,
                 Slab,
                 field_type='TE',
                 plane='Z',
                 ):
        self.Slab = Slab

        if field_type not in ['TE', 'TM']:
            raise ValueError("Field type must be 'TE' or 'TM'.")

        self.field_type = field_type
        self.ft = field_type

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        self.plane = plane

    def normalization(self, C, sign=1):

        try:
            len(C)
            C = np.array(C)
        except TypeError:
            C = np.array([C])

        M0 = np.zeros(C.shape + (2, 1), dtype=complex)
        M = self.Slab.transmission_matrix(
            C,
            field_type=self.ft,
            plane=self.plane,
            Ztype_far_left="standard",
            Ztype_far_right="standard",
        )

        r1 = -M[..., 1, 0] / M[..., 1, 1]
        detM = self.Slab.transmission_determinant(
            C, field_type=self.ft, plane=self.plane
        )
        t2 = 1 / (M[..., 1, 1] * detM)
        frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

        b = int(sign) * np.sqrt(frac, dtype=complex)

        C = np.sqrt(1 / (r1 + b * t2), dtype=complex)

        M0[..., 0, :] = C[..., np.newaxis]
        M0[..., 1, :] = C[..., np.newaxis].conjugate()

        M0 *= 1 / (2 * np.sqrt(np.pi))

        return M0

    def pole_locations(self, C, sign=1):

        M = self.Slab.transmission_matrix(
            C,
            field_type=self.ft,
            plane=self.plane,
            Ztype_far_left="standard",
            Ztype_far_right="standard",
        )

        detM = self.Slab.transmission_determinant(
            C, field_type=self.ft, plane=self.plane
        )
        b = np.sqrt(
            (-M[..., 1, 0] * detM) / M[..., 0, 1], dtype=complex
        )
        plus = -M[..., 1, 0] + b / detM
        minus = -M[..., 1, 0] - b / detM
        return plus * minus / M[..., 1, 1] ** 2


class OurMethod:
    def __init__(self,
                 Slab,
                 field_type='TE',
                 plane='Z',
                 ):
        self.Slab = Slab

        if field_type not in ['TE', 'TM']:
            raise ValueError("Field type must be 'TE' or 'TM'.")

        self.field_type = field_type
        self.ft = field_type

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        self.plane = plane

    def normalization(self, C, sign=1):

        try:
            len(C)
            C = np.array(C)
        except TypeError:
            C = np.array([C])

        if len(self.Slab.ns) == 1:
            M0 = np.ones((2, 1))
            M0[-1] *= int(sign)
            return M0 / (2 * np.sqrt(np.pi))

        M0 = np.zeros(C.shape + (2, 1), dtype=complex)
        M = self.Slab.transmission_matrix(
            C,
            field_type=self.ft,
            plane=self.plane,
            Ztype_far_left="standard",
            Ztype_far_right="standard",
        )

        detM = self.Slab.transmission_determinant(
            C, field_type=self.ft, plane=self.plane
        )

        frac = (-M[..., 1, 0] * detM) / M[..., 0, 1]

        b = int(sign) * np.sqrt(frac, dtype=complex)

        C1 = np.sqrt(
            (b.conj() - M[..., 0, 1]) / M[..., 0, 0], dtype=complex
        )
        C2 = np.sqrt(
            M[..., 0, 0] / (b.conj() - M[..., 0, 1]), dtype=complex
        )

        M0[..., 0, :] = C1[..., np.newaxis]
        M0[..., 1, :] = C2[..., np.newaxis]

        M0 *= 1 / (2 * np.sqrt(np.pi))

        return M0

    def pole_locations(self, C, sign=1):

        if len(self.Slab.ns) == 1:
            return np.ones_like(C)

        M = self.Slab.transmission_matrix(
            C,
            field_type=self.ft,
            plane=self.plane,
            Ztype_far_left="standard",
            Ztype_far_right="standard",
        )

        detM = self.Slab.transmission_determinant(
            C, field_type=self.ft, plane=self.plane
        )

        b = int(sign) * np.sqrt(
            (-M[..., 1, 0] * detM) / M[..., 0, 1], dtype=complex
        )

        return (b.conj() - M[..., 0, 1]) * M[..., 0, 0]


class EigvecMethod:
    def __init__(self, Slab, field_type='TE', plane='Z'):

        self.Slab = Slab

        if field_type not in ['TE', 'TM']:
            raise ValueError("Field type must be 'TE' or 'TM'.")

        self.field_type = field_type
        self.ft = field_type

        if plane not in ["Z", "Beta", "Psi"]:
            raise ValueError("Plane must be 'Beta', 'Z' or 'Psi'.")

        self.plane = plane

    def normalization(self, C, sign=1):

        sign = int(sign)

        if sign not in [-1, 1]:
            raise ValueError("Sign must be 1 or -1 (int).")

        try:
            len(C)
            C = np.array(C)
        except TypeError:
            C = np.array([C])

        M0 = np.zeros(C.shape + (2, 1), dtype=complex)
        M = self.Slab.transmission_matrix(
            C,
            field_type=self.ft,
            plane=self.plane,
            Ztype_far_left="standard",
            Ztype_far_right="standard",
        )

        trQs = (
            M[..., 0, 0] * M[..., 1, 0]
            + M[..., 0, 1] * M[..., 1, 1]
        )

        L = trQs + sign * np.sqrt(
            trQs**2 + 4 * M[..., 0, 0] * M[..., 1, 1],
            dtype=complex,
        )

        C1 = 2 * M[..., 0, 0] * M[..., 1, 1]
        C2 = L - 2 * M[..., 0, 0] * M[..., 1, 0]

        factor = np.sqrt(L * (C1**2 + C2**2), dtype=complex)

        C1 /= factor
        C2 /= factor

        M0[..., 0, :] = C1[..., np.newaxis]
        M0[..., 1, :] = C2[..., np.newaxis]

        M0 *= np.sqrt(1 / np.pi)

        return M0

    def pole_locations(self, C, sign=1):

        M = self.Slab.transmission_matrix(
            C,
            field_type=self.ft,
            plane=self.plane,
            Ztype_far_left="standard",
            Ztype_far_right="standard",
        )

        return -4 * M[..., 1, 1] * M[..., 0, 0]
