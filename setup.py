
from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

extension = Extension(
    "_lda",
    sources=["_lda.pyx", "gamma.c"],
    include_dirs = [np.get_include()]
)

setup(
    ext_modules=cythonize([extension])
)