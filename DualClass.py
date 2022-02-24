class DualNumber():
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

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

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return (self.real * other.real, self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)
    def __rmul__(self, other):
        if isinstance(other, DualNumber):
            return (self.real * other.real, self.real * other.dual + self.dual * other.real)
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

