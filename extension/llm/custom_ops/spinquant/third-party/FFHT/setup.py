import sys

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except(IOError, ImportError):
    long_description = open('README.md').read()

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    sys.stderr.write('Setuptools not found!\n')
    raise

try:
    import numpy as np
except ImportError:
    sys.stderr.write('NumPy not found!\n')
    raise

if sys.version_info[0] == 2:
    arr_sources = ['_ffht_2.c', 'fht.c']

if sys.version_info[0] == 3:
    arr_sources = ['_ffht_3.c', 'fht.c']

module = Extension('ffht',
                   sources= arr_sources,
                   extra_compile_args=['-march=native', '-O3', '-Wall', '-Wextra', '-pedantic',
                                       '-Wshadow', '-Wpointer-arith', '-Wcast-qual',
                                       '-Wstrict-prototypes', '-Wmissing-prototypes',
                                       '-std=c99', '-DFHT_HEADER_ONLY'],
                   include_dirs=[np.get_include()])

setup(name='FFHT',
      version='1.1',
      author='Ilya Razenshteyn, Ludwig Schmidt',
      author_email='falconn.lib@gmail.com',
      url='https://github.com/FALCONN-LIB/FFHT',
      description='Fast implementation of the Fast Hadamard Transform (FHT)',
      long_description=long_description,
      license='MIT',
      keywords='fast Fourier Hadamard transform butterfly',
      packages=find_packages(),
      include_package_data=True,
      ext_modules=[module])
