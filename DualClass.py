from typing import Union, Any
import numpy as np

class DualNumber():
    def __init__(self,
                 real,
                 dual=1):
        self.real = real
        self.dual = dual
        
    """
    Basic 4 arithmetic operations.
    """

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)
    def __radd__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real, self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return (self.real - other.real, self.dual - other.dual)
        else:
            return DualNumber(self.real - other, self.dual)
        
    def __rsub__(self, other):
        if isinstance(other, DualNumber):
            return (other.real - self.real, other.dual - self.dual)
        else:
            return DualNumber(other - self.real, -self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)
    def __rmul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)

    def __truediv__(self, other):
        assert other != 0
        if isinstance(other, DualNumber):
            return DualNumber(self.real / other.real, (self.dual * other.real - self.real * other.dual)) / (other.real**2)
        else:
            return DualNumber(self.real / other, self.dual / other)
    
    def __pow__(self, other):
        assert isinstance(other, int)
        return DualNumber(self.real ** other, self.dual * other * self.real ** (other-1))
    
    def __neg__(self):
        return DualNumber(-self.real, -self.dual)
    
    def __pos__(self):
        return DualNumber(self.real, self.dual)

    def __repr__(self):
        if self.dual>=0:
            return repr(self.real) + '+' + repr(self.dual) + 'e'
        else:
            return repr(self.real) + '-' + repr(abs(self.dual)) + 'e'
        
    """
    Calculating only real axis. Dual number will not be calculated.
    "l" implies left dual of the operand will be inherited.
    "r" implies right dual of the operand will be inherited.
    """
        
        
    def mul_lreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real * other.real, self.dual)
        else:
            return self.__mul__(self, other)
        
    def mul_rreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real * other.real, other.dual)
        else:
            return self.__mul__(self, other)
        
    def add_lreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real + other.real, self.dual)
        else:
            return self.__add__(self, other)
    
    def add_rreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real + other.real, other.dual)
        else:
            return self.__add__(self, other)
    
    def div_lreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real / other.real, self.dual)
        else:
            return self.__div__(self, other)
    
    def div_rreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real + other.real, other.dual)
        else:
            return self.__div__(self, other)
        
    def sub_lreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real - other.real, self.dual)
        else:
            return self.__sub__(self, other)
    
    def sub_rreal(self, other):
        if isinstance(other, DualNumber) or isinstance(other, DualTensor):
            return (self.real - other.real, other.dual)
        else:
            return self.__sub__(self, other)
        
    def sum(self):
        return DualNumber(np.sum(self.real), np.sum(self.dual))

        
class DualTensor(DualNumber):
    def __init__(self, 
                 real,
                 dual=1):
        
        super().__init__(real, dual)
        self.dtypeDual = Union[DualTensor, DualNumber]
        self.dtypeUsual = Union[np.ndarray, int, float]
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, DualTensor) or isinstance(other, DualNumber):
            return DualTensor(self.real + other.real, self.dual + other.dual)
        else:
            return DualTensor(self.real + other, self.dual)
    def __radd__(self, other):
        if isinstance(other, DualTensor) or isinstance(other, DualNumber):
            return DualTensor(self.real + other.real, self.dual + other.dual)
        else:
            return DualTensor(self.real + other, self.dual)

    def __sub__(self, other):
        if isinstance(other, DualTensor) or isinstance(other, DualNumber):
            return (self.real - other.real, self.dual - other.dual)
        else:
            return DualTensor(self.real - other, self.dual)
        
    def __rsub__(self, other):
        if isinstance(other, DualTensor) or isinstance(other, DualNumber):
            return (other.real - self.real, other.dual - self.dual)
        else:
            return DualTensor(other - self.real, -self.dual)

    def __mul__(self, other):
        if isinstance(other, DualTensor) or isinstance(other, DualNumber):
            return (self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return DualTensor(self.real * other, self.dual * other)
    def __rmul__(self, other):
        if isinstance(other, DualTensor) or isinstance(other, DualNumber):
            return (self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return DualTensor(self.real * other, self.dual * other)

    def __truediv__(self, other):
        assert other != 0, "Divisor should not be zero."
        if isinstance(other, DualTensor) or isinstance(other, DualNumber):
#             return DualTensor(self.real / other.real, (self.dual * other.real - self.real * other.dual)) / (other.real**2)

            assert(0), "Dividing multi-dimensional dual number into dual number is not permitted.\n" + \
                       "                This is due to ring structure of dual number system and ambiguity of division.\n" + \
                       "                For more details, refer 'https://en.wikipedia.org/wiki/Dual_number'"
        else:
            return DualTensor(self.real / other, self.dual / other)
    
    def __pow__(self, other):
        assert isinstance(other, int)
        return DualTensor(self.real ** other, self.dual * other * self.real ** (other-1))
    
    def __neg__(self):
        return DualTensor(-self.real, -self.dual)
    
    def __pos__(self):
        return DualTensor(self.real, self.dual)

    def __repr__(self):
        if self.dual>=0:
            return repr(self.real) + '+' + repr(self.dual) + 'e'
        else:
            return repr(self.real) + '-' + repr(abs(self.dual)) + 'e'
        


def check_dtype(tensor_list:Any,
                dtype:Any=DualNumber
                ) -> None:
    if type(tensor_list) == dtype:
        pass
    else:
        try:
            _ = iter(tensor_list)
            del _
            for tensor in tensor_list:
                check_dtype(tensor)
        except TypeError as e:
            raise e

def from_numpy(arr:np.ndarray,
               dual:Union[float, int]=1
               ) -> DualTensor:
    assert type(arr) == np.ndarray, f"Array should be 'numpy.ndarray'. Given dtype : {type(arr)}"
    dual = DualTensor(arr, dual)
    return dual