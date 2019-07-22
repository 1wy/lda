

from _lda import searchsorted
# from pymath import pyadd
import numpy as np


x = np.array([1,2,3], dtype=np.double)
length = len(x)
value = 1.1
idx = searchsorted(x, length, value)
print(idx)

