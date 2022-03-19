from distutils.core import setup, Extension
import numpy as np

ext_modules = []


ext = Extension('numpy_dsingle',
                sources=['dsingle.c',
                         'numpy_dsingle.c'],
                include_dirs=[np.get_include()],
                extra_compile_args=['-std=c11'])
ext_modules.append(ext)

setup(name='pydual',
      version='0.1',
      description='Dual number type for NumPy',
      ext_modules=ext_modules)
