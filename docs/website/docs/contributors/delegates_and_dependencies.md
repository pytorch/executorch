# Dependencies Management for Backend Delegatation

Disclaimer: We are planning to restructure the repository around delegates.
With that some of these guidelines will change in the future.

A delegate may depend on external, third-party libraries to efficiently
implement ahead-of-time (AoT) `partition()` or `preprocess()` functions, and/or
to implement runtime functions such as `init()` or `execute()`, or run tests in
a specific manner. This guide aims to classify different types of third-party
dependencies a delegate might depend on, and provide a high level guidance on
how to include them.

## Ahead-of-Time Dependencies

This includes dependencies used by the delegate's `partitioner()` and
`preprocess()` functions to generate preprocessed result which
will be used at later at runtime.

Depending on how the `preprocess()` function is implemented this can be either
Python or C++ dependency. This guide will talk about only Python AoT dependencies.

**Guidelines:**

* If Executorch already includes a dependency you require, prefer
  to use that if possible.
* Since the dependency is only used by the files inside the
  `executorch/backends/<delegate_name>/` - it should introduced in
  a way that it is needed only by the code inside the backend delegate
  directory.
* The dependency should not be installed by default when installing
  the Executorch Python package.

More details in the section [below](#python-dependencies).

## Runtime Dependencies

This category covers C++ dependencies used by the delegate runtime code.
It can be as simple as a third-party math library to implement some
delegate operator, or can be a whole framework handling the lowered
subgraph for the delegate.

**Guidelines:**

At a high level, only pay for what you use should be the desired approach
for these third-party dependencies.

* Similar to the AoT dependencies, the use of this should also be restricted to
  only the delegate runtime source files.
* If a delegate has a dependency which is already part of
  `executorch/third-party` then try to use that if possible. This
  helps with reducing the binary size when the delegate is enabled.
* Rest of the Executorch code, outside of the delegate, should not depend on
  this. And it should should build and run correctly without this dependency
  when the delegate is disabled at build time.

More details in the section [below](#runtime-dependencies).

## Testing-Only Dependencies

Some libraries, or tools are only used for executing the delegate tests. These
can either be a Python dependency or a C++ dependency depending on the type of
the test.

**Guidelines:**

* For a Python dependency, it should not be installed by default when
  installing the Executorch Python package.
* If for C++ tests, it should not be part of the
  Executorch runtime even when the delegate is built/enabled.

## Other considerations

### Versioning

Explicit and specific is preferred. For example a pypi version (or a criteria) or
a git tag/release.

### End user vs. Developer experience

* More details TBD

<!---
TODO
Need to add more about developer experiences, users selecting which delegates
to enable/install for both AoT and Runtime
--->

### Documenting the dependency
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
Executorch dependencies.

Version conflict should be avoided by trying to use the already included
dependency by Executorch or by some other backend if possible. Otherwise
try some other
[recommended](https://pip.pypa.io/en/latest/topics/dependency-resolution/#dealing-with-dependency-conflicts)
ways to mitigate version conflicts.

#### Local Python Packages
If it is a git repository, it should be added as a git submodule.

* More details TBD

<!-- Something like
https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-from-vcs,
but the URLs can't be in the requirements.txt, so not recommending this for now. -->

## C++ Dependencies

The recommended approach is to include a git submodule for a given C++
dependency in the `executorch/backends/<delegate_name>/third-party`.

### buck2/CMake support
At a minimum CMake support is required. Adding buck2 support should make
the delegate available to more Executorch users.

* More details TBD

<!---
TODO
Complying with ET runtime build configurations. Standard switches for library
linking (i.e. static, PIC), optimization flags pass through, etc.
--->
