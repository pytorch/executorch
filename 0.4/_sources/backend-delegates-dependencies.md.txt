# Third-Party Dependency Management for Backend Delegates

Disclaimer: We are planning to restructure the repository around delegates.
With that some of these guidelines will change in the future.

A delegate may depend on external, third-party libraries to efficiently
implement ahead-of-time (AOT) `partition()` or `preprocess()` functions, and/or
to implement runtime functions such as `init()` or `execute()`, or to run tests
in a specific manner. This guide aims to classify different types of third-party
dependencies a delegate might depend on, and provide a high level guidance on
how to include them.

## Ahead-of-Time Dependencies

This includes dependencies used by the delegate's `partitioner()` and
`preprocess()` functions to generate preprocessed result which
will be used at later at runtime.

Depending on how the `preprocess()` function is implemented this can be either
Python or C++ dependency. This guide will talk about only Python AOT dependencies.

**Guidelines:**

* If ExecuTorch already includes a dependency you require, prefer
  to use that if possible.
* If the dependency is only needed by the files inside the
  `executorch/backends/<delegate_name>/` directory, it should be introduced in a
  way such that it is used only by the code under that directory.
* The dependency should not be installed by default when installing
  the ExecuTorch Python package.

More details in the section [below](#python-dependencies).

## Runtime Dependencies

This category covers C++ dependencies used by the delegate runtime code.
It can be as simple as a third-party math library to implement some
delegate operator, or can be a whole framework handling the lowered
subgraph for the delegate.

**Guidelines:**

At a high level, "only pay for what you use" should be the desired approach
for these third-party dependencies.

* Similar to the AOT dependencies, the use of this should also be restricted to
  only the delegate runtime source files.
* If a delegate has a dependency which is already part of
  `executorch/third-party` then try to use that if possible. This
  helps with reducing the binary size when the delegate is enabled.
* The rest of the ExecuTorch code, outside of the delegate, should not depend on
  this. And it should should build and run correctly without this dependency
  when the delegate is disabled at build time.

More details in the section [below](#runtime-dependencies).

## Testing-Only Dependencies

Some libraries or tools are only used for executing the delegate tests. These
can either be a Python dependency or a C++ dependency depending on the type of
the test.

**Guidelines:**

* For a Python test dependency, it should not be installed by default when
  installing the ExecuTorch Python package.
* For a C++ test dependency, it should not be part of the ExecuTorch runtime
  even when the delegate is built/enabled.

## Other Considerations

### Versioning

Explicit and specific is preferred. For example a PyPI version (or range) or
a git tag/release.

<!---
### End-User vs. Developer Experience

TODO
Need to add more about developer experiences, users selecting which delegates
to enable/install for both AOT and Runtime
--->

### Documenting Dependencies
At a minimum, some documentation under `executorch/backends/<delegate_name>/`
should be provided when introducing a new dependency which includes,

* Rationale for introducing a new third-party dependency
* How to upgrade the dependency
* Any special considerations for the new dependency

***

After listing the high level guidelines, let's now talk about specific
logistics to actually include a dependency for your delegate,

## Python Dependencies

Python packaging is complicated and continuously evolving. For delegate
dependencies, we recommend that a delegate specifies its third-party
dependencies under `executorch/backends/<delegate_name>/requirements.txt` to be
supplied to pip at installation time. The goal is to decouple them from the core
ExecuTorch dependencies.

Version conflicts should be avoided by trying to use the dependency already
included by ExecuTorch or by some other backend if possible. Otherwise try some
other
[recommended](https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts)
ways to mitigate version conflicts.

#### Local Python Packages
If it is a git repository, it should be added as a git submodule.

<!--
TODO: Add more details. Something like
https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-vcs,
but the URLs can't be in the requirements.txt, so not recommending this for now.
-->

## C++ Dependencies

The recommended approach is to include a git submodule for a given C++
dependency in the `executorch/backends/<delegate_name>/third-party` directory.

### CMake Support
At a minimum CMake support is required.

<!---
TODO: Add more details about: complying with ET runtime build configurations;
standard switches for library linking (i.e. static, PIC), optimization flags
pass through, etc.
--->
