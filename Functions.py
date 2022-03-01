import math
from DualClass import DualNumber
from typing import Union, Any
import numpy as np

class Exp():
    def __init__(self, real, dual=1, nth_expansion=50):
        if isinstance(real, DualNumber):
            self.dn = real
        else:
            self.dn = DualNumber(real, dual)
        self.nth_expansion = nth_expansion
        
    def __call__(self):
        exp_x = DualNumber(0, 0)
        for i in range(self.nth_expansion):
            exp_x += (self.dn**i) / math.factorial(i)
        return exp_x
    
    
class Sin():
    def __init__(self, real, dual=1, nth_expansion=50):
        if isinstance(real, DualNumber):
            self.dn = real
        else:
            self.dn = DualNumber(real, dual)
        self.nth_expansion = nth_expansion

    def __call__(self):
        sin_x = DualNumber(0, 0)
        for i in range(self.nth_expansion):
            sin_x += (self.dn ** (2*i+1)) * ((-1) ** i) / math.factorial(2*i+1)
        return sin_x
    
class Cos():
    def __init__(self, real, dual=1, nth_expansion=50):
        if isinstance(real, DualNumber):
            self.dn = real
        else:
            self.dn = DualNumber(real, dual)
        self.nth_expansion = nth_expansion

    def __call__(self):
        cos_x = DualNumber(0, 0)
        for i in range(self.nth_expansion):
            cos_x += (self.dn ** (2*i)) * ((-1) ** i) / math.factorial(2*i)
        return cos_x
    
class Sigmoid():
    def __init__(self, real, dual=1):
        if isinstance(real, DualNumber):
            self.dn = real
        else:
            self.dn = DualNumber(real, dual)
    def __call__(self):
        """
        1/(1+exp(-x)) cannot be implemented due to indivisibility of dual number system
            : Dual number system have ring structure, not field structure.
        Used Wolfram Alpha's Taylor expansion for sigmoid function up to 11th order.
        https://www.wolframalpha.com/input?i=taylor+series+of+sigmoid
        """
        sigmoid_x = 1/2
        sigmoid_x = (self.dn)/4 + sigmoid_x
        sigmoid_x = -1*((self.dn)**3)/48 + sigmoid_x
        sigmoid_x = ((self.dn)**5)/480 + sigmoid_x
        sigmoid_x = -1*((self.dn)**7)*17/80640 + sigmoid_x
        sigmoid_x = ((self.dn)**9)*31/1452510 + sigmoid_x
        sigmoid_x = -1*((self.dn)**11)*691/319334400 + sigmoid_x
        return sigmoid_x
    
class ReLU():
    def __init__(self, real, dual=1):
        if self.real>=0:
            self.dn = DualNumber(real, dual)
        else:
            self.dn = DualNumber(0, dual)
    def __call__(self):
        return self.dn

    