import numpy as np
# import torch
# import torch.nn as nn
from typing import Union, Any
from itertools import product
from copy import deepcopy

class DualNumber():
    def __init__(self,
                 real,
                 isT:bool=False,
                 **kwargs
                 ):
        assert not isinstance(real, DualNumber), "Conversion from DualNumber to DualNumber is not supported"
        self.real = real
        self.isT = isT
        transposed = {}
        self.kwargs = kwargs
        self.names = []
        self.dual = {}
        for name in kwargs.keys():
            if isinstance(kwargs[name], int) or isinstance(kwargs[name], float):
                try:
                    shape = self.real.shape
                except:
                    shape = (1,)
                indices = [list(range(s)) for s in shape]
                prod = product(*indices)
                for p in prod:
                    ijk_name = name + '_' + '_'.join([str(_) for _ in list(p)])
                    self.names.append(ijk_name)
                    value = self.real[*p]
                    self.dual[ijk_name] = value
                    
            else:
                dual = kwargs[name]

        try:
            self.realT = deepcopy(real.T)
        except:
            self.realT = real
        if not self.isT:
            self.isT = True
            self.T = DualNumber(self.realT, self.isT, **transposed)
        else:
            return
        
    """
    Basic 4 arithmetic operations.
    """

    def __add__(self, other):
        if isinstance(other, DualNumber):
            duals = {}
            for k in self.names:
                duals[k] = self.dual[k] + other.dual[k]
            return DualNumber(self.real + other.real, **duals)
        else:
            self.real = self.real + other
            return self

    def __radd__(self, other):
        return self.__add__(other)

    # def __sub__(self, other):
        
    # def __rsub__(self, other):
        

    # def __mul__(self, other):

    # def __rmul__(self, other):
    #     return self.__mul__(other)

    # def __truediv__(self, other):

    # def __rtruediv__(self, other):
    

    # def __pow__(self, power):
    #     assert isinstance(power, int)
    #     pow = 1
    #     for _ in range(power):
    #         pow *= self
    #     return pow
    
    # def __neg__(self):
    #     return -1 * self
    
    # def __pos__(self):
    #     return self

    def __repr__(self):
        string = ''
        for e in self.names:
            string += ' + \n' + repr(getattr(self, e))
        return repr(self.real) + string

    # def sum(self, axis=None):
    
    # def mean(self, axis=0):
        
        
    # """
    # Calculating only real axis. Dual number will not be calculated.
    # "l" implies left dual of the operand will be inherited.
    # "r" implies right dual of the operand will be inherited.
    # """

    # def dot(self, other, reduction=False):
    #     assert isinstance(other, DualNumber), "Dot prouct only between dual numbers is supported."
    #     dual_dots = {}
    #     for d1 in self.names:
    #         dual_dots[d1] = getattr(self, d1).dot(other.real)
    #         if reduction:
    #             dual_dots[d1] /= self.real.shape[1]
    #         for d2 in other.names:
    #             dual_dots[d2] = self.real.dot(getattr(other, d2))
    #             if reduction:
    #                 dual_dots[d2] /= self.real.shape[1]
    #             if d1==d2:
    #                 if reduction:
    #                     dual_dots[d1] = (self.real.dot(getattr(other, d1)) + \
    #                                     getattr(self, d1).dot(other.real)) / self.real.shape[1]
    #                 else:
    #                     # print(self.real.dot(getattr(other, d1)))
    #                     dual_dots[d1] = self.real.dot(getattr(other, d1)) + \
    #                                     getattr(self, d1).dot(other.real)
    #     if reduction:
    #         real_part = self.real.dot(other.real) / self.real.shape[1]
    #     else:
    #         real_part = self.real.dot(other.real)
    #     return DualNumber(real_part, **dual_dots)
