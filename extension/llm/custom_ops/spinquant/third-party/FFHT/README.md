# Fast Fast Hadamard Transform

FFHT (Fast Fast Hadamard Transform) is a library that provides a heavily
optimized C99 implementation of the Fast Hadamard Transform. FFHT also provides
a thin Python wrapper that allows to perform the Fast Hadamard Transform on
one-dimensional [NumPy](http://www.numpy.org/) arrays.

The Hadamard Transform is a linear orthogonal map defined on real vectors whose
length is a _power of two_. For the precise definition, see the
[Wikipedia entry](https://en.wikipedia.org/wiki/Hadamard_transform). The
Hadamard Transform has been recently used a lot in various machine learning
and numerical algorithms.

FFHT uses [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)
to speed up the computation.

The header file `fht.h` exports two functions: `int fht_float(float *buf, int
log_n)` and `int fht_double(double *buf, int log_n)`. The
only difference between them is the type of vector entries. So, in what follows,
we describe how the version for floats `fht_float` works.

The function `fht_float` takes two parameters:

* `buf` is a pointer to the data on which one needs to perform the Fast
Hadamard Transform.
* `log_n` is the binary logarithm of the length of `buffer`.
That is, the length is equal to `2^log_n`.

The return value is -1 if the input is invalid and is zero otherwise.

A header-only version of the library is provided in `fht_header_only.h`.

In addition to the Fast Hadamard Transform, we provide two auxiliary programs:
`test_float` and `test_double`, which are implemented in C99. The exhaustively
test and benchmark the library.

FFHT has been tested on 64-bit versions of Linux, OS X and Windows (the latter
is via Cygwin).

To install the Python package, run `python setup.py install`. The script
`example.py` shows how to use FFHT from Python.

## Benchmarks

Below are the times for the Fast Hadamard Transform for vectors of
various lengths. The benchmarks were run on a machine with Intel
Core&nbsp;i7-6700K and 2133 MHz DDR4 RAM. We compare FFHT,
[FFTW 3.3.6](http://fftw.org/), and
[fht](https://github.com/nbarbey/fht) by
[Nicolas Barbey](https://github.com/nbarbey).

Let us stress that FFTW is a great versatile tool, and the authors of FFTW did
not try to optimize the performace of the Fast Hadamard Transform. On the other
hand, FFHT does one thing (the Fast Hadamard Transform), but does it extremely
well.

Vector size | FFHT (float) | FFHT (double) | FFTW 3.3.6 (float) | FFTW 3.3.6 (double) | fht (float) | fht (double)
:---: | :---: | :---: | :---: | :---: | :---: | :---:
2<sup>10</sup> | 0.31 us | 0.49 us | 4.48 us | 7.72 us | 17.4 us | 19.3 us
2<sup>20</sup> | 0.68 ms | 1.39 ms | 8.81 ms | 17.07 ms | 29.8 ms | 35.0 ms
2<sup>27</sup> | 0.22 s | 0.50 s | 2.08 s | 3.57 s | 6.89 s | 7.49 s

## Troubleshooting

For some versions of OS X the native `clang` compiler (that mimicks `gcc`) may
not recognize the availability of AVX. A solution for this problem is to use a
genuine `gcc` (say from [Homebrew](http://brew.sh/)) or to use `-march=corei7-avx`
instead of `-march=native` for compiler flags.

A symptom of the above happening is the undefined macros `__AVX__`.

## Related Work

FFHT has been created as a part of
[FALCONN](https://github.com/falconn-lib/falconn): a library for similarity
search over high-dimensional data. FALCONN's underlying algorithms are described
and analyzed in the following research paper:

> Alexandr Andoni, Piotr Indyk, Thijs Laarhoven, Ilya Razenshteyn and Ludwig
> Schmidt, "Practical and Optimal LSH for Angular Distance", NIPS 2015, full
> version available at [arXiv:1509.02897](http://arxiv.org/abs/1509.02897)

This is the right paper to cite, if you use FFHT for your research projects.

## Acknowledgments

We thank Ruslan Savchenko for useful discussions.

Thanks to:

* Clement Canonne
* Michal Forisek
* Rati Gelashvili
* Daniel Grier
* Dhiraj Holden
* Justin Holmgren
* Aleksandar Ivanovic
* Vladislav Isenbaev
* Jacob Kogler
* Ilya Kornakov
* Anton Lapshin
* Rio LaVigne
* Oleg Martynov
* Linar Mikeev
* Cameron Musco
* Sam Park
* Sunoo Park
* Amelia Perry
* Andrew Sabisch
* Abhishek Sarkar
* Ruslan Savchenko
* Vadim Semenov
* Arman Yessenamanov

for helping us with testing FFHT.
