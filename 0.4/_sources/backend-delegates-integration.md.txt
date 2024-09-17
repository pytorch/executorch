# Integrating a Backend Delegate into ExecuTorch

Disclaimer: We are planning to restructure the repository around delegates.
With that some of these guidelines will change in the future.

This is a high level guideline when integrating a backend delegate with ExecuTorch.

## Directory Structure

Delegate files should be under this directory:
`executorch/backends/<delegate_name>/`. The delegate name should be unique.

## Python Source Files

Delegate Python files such as those implementing `preprocess()` or `partition()`
functions for ExecuTorch AOT flow, excluding any external third-party
dependencies and their files, should be installed and available with
the top level ExecuTorch package. For third-party dependencies, please refer to
[this](./backend-delegates-dependencies.md).

## C++ Source Files

At a minimum, a delegate must provide CMake support for building its C++
sources.

For the CMake setup, the delegate dir should be included by the
top level `CMakeLists.txt` file using `add_subdirectory` CMake command, and
should be built conditionally with an ExecuTorch build flag like
`EXECUTORCH_BUILD_<DELEGATE_NAME>`, see `EXECUTORCH_BUILD_XNNPACK` for example.
For third-party dependencies, please refer to
[this](./backend-delegates-dependencies.md).

<!---
TODO: Add more details. Need to insert a CMake layer in `executorch/backends` to
provide some uniform abstraction across delegates.
--->

## Tests

Tests should be added under `executorch/backends/<delegate_name>/test`. Tests
can be either python or C++ tests. For adding more complex end-to-end (e2e)
tests, please reach out to us.

Common test types:
* Simple python unit tests that test AOT logic such as `partitioner()` or AOT
  export flow (generating a `.pte` file from an `nn.Module`)
* Runtime C++ tests, using gtest, that test delegate `init()` or `execute()`
  runtime logic.

## Documentation

A delegate must contain a `executorch/backends/<delegate_name>/README.md`
explaining the basics of the delegate, directory structure, features, and known
issues if any.

Any extra setup steps beyond the ones listed above should be documented in
`executorch/backends/<delegate_name>/setup.md`
