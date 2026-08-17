#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 10:58:29 2026

@author: pv
"""
import numpy as np


class Field():
    def __init__(self, F):
        '''
        Class representing fields in a step-index slab waveguide.

        Allows for easy addition and scaling of fields.

        Parameters
        ----------
        F : function
            The field function. This must be a function of two variables.
        '''
        if not callable(F):
            raise ValueError('Input must be callable.')
        self.F = F
        # self.F_kwargs = F_kwargs
        self.compatible_dtypes = [int, float, complex,
                                  np.complex128, np.complex256, np.complex64,
                                  np.int16, np.int64, np.int32, np.int8,
                                  np.float128, np.float16, np.float32,
                                  np.float64]

    def __call__(self, x, z=np.array([0]), **kwargs):
        return self.F(x, z, **kwargs)

    def __str__(self,):
        return 'Field'

    def __add__(self, other):

        if callable(other):
            def FpG(x, z=np.array([0]), **kwargs):
                return self.F(x, z, **kwargs) + other(x, z, **kwargs)
            return Field(FpG)

        # compat_array = (type(other) == np.ndarray and
        #                 other.dtype in self.compatible_dtypes)
        compat_scalar = type(other) in self.compatible_dtypes

        # if compat_array or compat_scalar:
        if compat_scalar:
            def FpG(x, z, **kwargs):
                return self.F(x, z, **kwargs) + other
            return Field(FpG)

        else:
            raise ValueError("Can't add Field and object of type \
%s" % type(other))

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        if callable(other):
            def FtG(x, z=np.array([0]), **kwargs):
                return self.F(x, z, **kwargs) * other(x, z, **kwargs)
            return Field(FtG)

        # compat_array = (type(other) == np.ndarray and
        #                 other.dtype in self.compatible_dtypes)
        compat_scalar = type(other) in self.compatible_dtypes

        # if compat_array or compat_scalar:
        if compat_scalar:
            def FtG(x, z=np.array([0]), **kwargs):
                return self.F(x, z, **kwargs) * other
            return Field(FtG)

        else:
            raise ValueError("Can't multiply Field and object of type \
%s" % type(other))

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * (-1)

    def __sub__(self, other):
        return self + -other

    def __rsub(self, other):
        return -self + other

    def __abs__(self):
        def absF(x, z, **kwargs):
            return np.abs(self.F(x, z, **kwargs))
        return Field(absF)

    def __pow__(self, n):
        def powF(x, z, **kwargs):
            return self.F(x, z, **kwargs)**n
        return Field(powF)

    def __truediv__(self, other):
        if callable(other):
            def FdG(x, z=np.array([0]), **kwargs):
                return self.F(x, z, **kwargs) / other(x, z, **kwargs)
            return Field(FdG)

        # compat_array = (type(other) == np.ndarray and
        #                 other.dtype in self.compatible_dtypes)
        compat_scalar = type(other) in self.compatible_dtypes

        # if compat_array or compat_scalar:
        if compat_scalar:
            def FdG(x, z=np.array([0]), **kwargs):
                return self.F(x, z, **kwargs) / other
            return Field(FdG)

        else:
            raise ValueError("Can't divide Field by object of type \
%s" % type(other))

    def __rtruediv__(self, other):
        return other * self**(-1)


# def f(x, z=1, a=1, **kwargs):
#     return x + z + a


# def g(x, z=4, b=2, **kwargs):
#     return x - z * b


# A = Field(f)
# B = Field(g)
# C = A/2


#     def __add__(self, other):
#         if callable(other):
#             if type(other) == Field:
#                 kwargs = {}
#                 for key, val in self.F_kwargs.items():
#                     kwargs[key] = val
#                 for key, val in other.F_kwargs.items():
#                     kwargs[key] = val

#                 def FpG(x, z, **args):
#                     return self.F(x, z, **self.F_kwargs) + \
#                         other.F(x, z, **other.F_kwargs)
#                 return Field(FpG, **kwargs)

#             else:
#                 kwargs = self.F_kwargs

#                 def FpG(x, z, **kwargs):
#                     return self.F(x, z, **self.F_kwargs) + other(x, z, **kwargs)
#                 return Field(FpG, **kwargs)

#         compat_array = (type(other) == ndarray and
#                         other.dtype in self.compatible_dtypes)
#         compat_scalar = type(other) in self.compatible_dtypes

#         if compat_array or compat_scalar:
#             def FpG(x, z, **kwargs):
#                 return self.F(x, z, **self.F_kwargs) + other
#             return Field(FpG, **self.F_kwargs)

#         else:
#             raise ValueError("Can't add Field type to object of type \
# %s" % type(other))
