This page provides a description of the Portable Kernel Library and the Optimized Kernel Library, which are the default kernel libraries shipped with ExecuTorch. It is recommended reading for those who are interested in executing ExecuTorch programs with these kernel libraries, or for those who want to implement their own kernels and kernel libraries.

# Overview of ExecuTorch’s Kernel Libraries

An ExecuTorch program encodes instructions that describe the computation that should be performed by the program. Many of these instructions will correspond to calling a specific ATen operator, for example `aten.convolution`. However, one of the core design principles of ExecuTorch is that the signature of an operator should be separate from the implementation of the operator. This means that the ExecuTorch runtime does not ship with any standard implementation for ATen operators; users must make sure to link against kernel libraries that contain implementations of the operators required by their ExecuTorch program, and configure [operator registration](./kernel-library-custom-aten-kernel.md) to map an operator signature to the desired implementation. This makes it easy to adjust the implementation of operators such as `aten.convolution` that will be called when executing an ExecuTorch program; it allows users to select the exact operator implementations that will meet the unique performance, memory usage, battery usage, etc. constraints of their use-case.

**In essence, a kernel library is simply a collection of ATen operator implementations that follow a common theme or design principle**. Note that due to ExecuTorch’s selective build process (discussed in the following section), operator implementations are linked individually. This means that users can easily mix different kernel libraries in their build without sacrificing build size.

ExecuTorch ships with two kernel libraries by default: the **Portable Kernel Library** and the **Optimized Kernel Library**, both of which provide CPU operator implementations.

## Portable Kernel Library

The Portable Kernel Library is in a sense the “reference” kernel library that is used by ExecuTorch. The Portable Kernel Library was developed with the following goals in mind:

* Correctness
    * Provide straightforward implementations of ATen operators that are strictly consistent with the original implementation of the operator in PyTorch’s ATen library
* Readability / Simplicity
    * Provide clear, readable source code so that those who want to develop custom implementations of an operator can easily understand the desired behavior of the operator.
* Portability
    * Portable Kernels should be just as portable as the ExecuTorch runtime; operator implementations should not use any external dependencies, or use any unsanctioned features of C++.
* Operator Coverage
    * As the “reference” kernel library for ExecuTorch, the Portable Kernel Library aims to have a high degree of operator coverage. The goal is for the Portable Kernel library to provide an implementation for every operator listed as a Core ATen operator. However, note that operator coverage for the Portable Kernel Library is still a work in progress.

The Portable Kernel Library primarily aims to provide easily accessible operator implementations that will “just work” on most platforms, and are guaranteed to provide correct output. Performance is a non-goal for the Portable Kernel Library. In fact, many bottleneck operators such as convolution and matrix multiplication are implemented in the most straightforward way possible in the interest of prioritizing simplicity and readability. Therefore, one should not expect to observe fast inference times if exclusively using the Portable Kernel library. However, outside of specific bottleneck operators, most operators are simple enough where the straightforward implementation of the Portable Kernel Library should still provide adequate performance. Binary size is also a non-goal for the Portable Kernel Library.

## Optimized Kernel Library

The Optimized Kernel Library is a supplemental kernel library shipped with ExecuTorch that, in contrast to the Portable Kernel Library, aims to provide performance focused implementations of operators at the cost of portability and readability. Many operator implementations in the Optimized Kernel Library are inspired or based off of the corresponding implementation in PyTorch’s ATen library, so in many cases one can expect the same degree of performance.

Generally speaking, operators in the Optimized Kernel Library are optimized in one of two ways:

1. Using CPU vector intrinsics
2. Using optimized math libraries, such as `sleef` and `OpenBLAS`

Although portability is not a design goal of the Optimized Kernel Library, implementations are not meant to be fine-tuned for a specific CPU architecture. Instead, the Optimized Kernel library seeks to provide performant implementations that can be applied across a variety of platforms, rather than using optimizations that are specific to a single platform.

Another important note is that operator coverage is also a non-goal for the Optimized Kernel Library. There are no plans to add optimized kernels for every Core ATen operator; rather, optimized kernels are added on an as-needed basis to improve performance on specific models. Thus, the operator coverage in the Optimized Kernel Library will be much more limited compared to the Portable Kernel Library.
