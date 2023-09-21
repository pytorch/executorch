# How to integrate a Backend Delegate with Executorch?

Disclaimer: We are planning to restructure the repository around delegates.
With that some of these guidelines will change in the future.

This is a high level guideline when integrating a backend delegate with Executorch.

## Directory Structure

Delegate files should be under this directory:
`executorch/backends/<delegate_name>/`. Delegate name should be unique.

## Python files

Delegate Python files such as one implementing `preprocess()` or `partition()`
functions for Executorch AoT flow, excluding any external third-party
dependencies and their files, should be installed and available with
the top level Executorch package. For third-party dependencies, please refer to
[this](./delegates_and_dependencies.md).

## C++ sources

At a minimum, a delegate must provide CMake support for building its C++
sources.

For the CMake setup, the delegate dir should be included by the
top level `CMakeLists.txt` file using `add_subdirectory` CMake command, and
should be built conditionally with an Executorch build flag like
`EXECUTORCH_BUILD_<DELEGATE_NAME>`, see `EXECUTORCH_BUILD_XNNPACK` for example.
For third-party dependencies, please refer to
[this](./delegates_and_dependencies.md).

Adding buck2 support should make the delegate available to more
Executorch users.

* More details TBD

<!---
TODO
Need to insert a CMake layer in `executorch/backends` to provide some
uniform abstraction across delegates.
--->


## Tests

Tests should be added under `executorch/backends/<delegate_name>/test`.
Tests can be either python or C++ tests, for adding more complex e2e please reach out to us.

* Python unit-tests, which are simple python tests to test AoT logic such as
`partitioner()`, AoT export flow i.e., `nn.Module` to generating the `.pte` file.

* Runtime C++ tests, using gtest, can be implemented to test delegate `init()`
or `execute()` logic.

## Documentation

A delegate must contain a `executorch/backends/<delegate_name>/README.md`
explaining the basics of the delegate, directory structure, features, and known-issues if any.

Any extra setup step(s) beyond the ones listed above, should be documented in
`executorch/backends/<delegate_name>/setup.md`
