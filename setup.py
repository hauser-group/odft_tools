from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

ext_modules = [
    Extension(
        'odft_tools.cython_kernels',
        ['odft_tools/cython_kernels.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
    ),
    Extension(
        'odft_tools.modular_cython_kernels',
        ['odft_tools/modular_cython_kernels.pyx'],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[np.get_include()],
    )
]

setup(name='odft_tools',
      version='0.1',
      description='A collection of tools used in our investigations of orbital free dft',
      url='https://github.com/hauser-group/odft_tools',
      author='Ralf Meyer',
      author_email='meyer.ralf@yahoo.com',
      packages=['odft_tools'],
      ext_modules = cythonize(ext_modules),
      install_requires=requirements,
      zip_safe=False)
