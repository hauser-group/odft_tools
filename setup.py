from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'odft_tools.cython_kernels',
        ['odft_tools/cython_kernels.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
    )
]

setup(name='odft_tools',
      version='0.1',
      description='A collection of tools used in our investigations of orbital free dft',
      url='http://github.com/storborg/funniest',
      author='Ralf Meyer',
      author_email='meyer.ralf@yahoo.com',
      packages=['odft_tools'],
      ext_modules = cythonize(ext_modules),
      zip_safe=False)