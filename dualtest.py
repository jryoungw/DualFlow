import numpy as np
from numpy_dsingle import dsingle

a = np.ones((10,10),dtype=dsingle)
b = np.ones((10),dtype=dsingle)

print(np.dot(a,b))

