import math
from DualClass import DualNumber

class Exp(DualNumber):
    def __init__(self, real, dual=1, nth_expansion=50):
        super().__init__(real, dual)
        self.nth_expansion = nth_expansion
        self.dn = DualNumber(real, dual)
        
    def __call__(self):
        exp_x = DualNumber(0, 0)
        for i in range(self.nth_expansion):
            exp_x += (self.dn**i) / math.factorial(i)
        return exp_x
    
    
class Sin(DualNumber):
    def __init__(self, real, dual=1, nth_expansion=50):
        super().__init__(real, dual)
        self.nth_expansion = nth_expansion
        self.dn = DualNumber(real, dual)

    def __call__(self):
        sin_x = DualNumber(0, 0)
        for i in range(self.nth_expansion):
            sin_x += (self.dn ** (2*i+1)) * ((-1) ** i) / math.factorial(2*i+1)
        return sin_x
    
class Cos(DualNumber):
    def __init__(self, real, dual=1, nth_expansion=50):
        super().__init__(real, dual)
        self.nth_expansion = nth_expansion
        self.dn = DualNumber(real, dual)

    def __call__(self):
        cos_x = DualNumber(0, 0)
        for i in range(self.nth_expansion):
            cos_x += (self.dn ** (2*i)) * ((-1) ** i) / math.factorial(2*i)
        return cos_x