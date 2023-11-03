Thank you for your interest in contributing to ExecuTorch!

This document (CONTRIBUTING.md) covers some of the more technical aspects of
contributing.

## Coding Style

Goal: Encourage standards that make it easier to read, edit, maintain, and debug
the ExecuTorch code.

You can see [lintrunner](https://pypi.org/project/lintrunner/) for making sure the code follows our standard. Here's how to set up `lintrunner`:

```
pip install lintrunner==0.11.0
pip install lintrunner-adapters==0.11.0
lintrunner init
```

### Python Style

ExecuTorch Python code follows the style used by the PyTorch core project.

### C++ Style

ExecuTorch code uses the [Google C++
Style](https://google.github.io/styleguide/cppguide.html), with modifications.

Rationale: Google style is close to the C++ style used by PyTorch core, although
PyTorch core does not explicitly document its C++ style. Google style is well
documented, and has exceptional tooling support.

**Modifications** to the Google C++ style, to make it closer to the code in
PyTorch core:
- Function and method names should use `lower_snake_case()`. This follows the
  convention that PyTorch core inherited from its namesake Python, and is the
  biggest modification to the Google C++ style.
- File names should use `lower_snake_case.cpp` (not `.cc`, and not
  `PascalCase.cpp`). This follows the most common pattern in PyTorch core.
- Headers should use `#pragma once` instead of manual include guards. This
  follows the most common pattern in PyTorch core.
- All includes should use `<angle brackets>`, not `"double quotes"`. This
  ensures that headers are included using the compiler's include path, and not
  relative to the local file.
- Documentation comments should follow Doxygen syntax, either `//** ... */`
  (multi-line) or `/// ...` (single line), with `@`-style parameters like
  `@param`, `@retval`. Public APIs must be documented in the `.h` files that
  declare them.
- TODOs should prefer to reference a task or issue number like `TODO(#123):
  <description>`, rather than a username. A task can manage much-more-nuanced
  information, and can change ownership as people leave and join the project.

See the rest of this file for other portability- and efficiency-related
modifications to the Google C++ style guide.

### C++ Portability Guidelines

See also [Portable C++ Programming](/docs/source/portable-cpp-programming.md)
for detailed advice.

#### C++ language version

**C++11.**

NOTE: The code does not yet fully conform to this, and some files require C++17.

Rationale: This is a compromise between being compatible with older, proprietary
toolchains, and having access to relatively modern C++ features.

#### C/C++ standard library usage

**Restricted usage of the C++ standard library.**

Rationale: ExecuTorch is intended to be portable to bare-metal systems that lack
certain features, like dynamic memory, threading, and locking, required by parts
of the standard library. It is also intended to be as small as possible, and
some convenient stdlib features may grow the binary size unacceptably.

Generally, do not instantiate types that allocate memory under the hood, like
`std::vector` or `std::string`. Do not call `new`, `malloc()` or `mmap()`; do
not use iostreams; do not operate on files.

However, it is convenient and portable (and sometimes necessary) to use static
standard library concepts like `std::move`, or metaprogramming helpers like
`std::is_floating_point<>`.  Pure code like `<cmath>` and `<cstring>` is fine,
as long as you stay away from functions that allocate memory (like `strdup()`).

It is also allowed (and sometimes necessary) to use "placement `new`", but be
careful to also manually destroy objects initialized in this way.

#### C++ language features

**Exceptions: Do not use.**
- Rationale: Exceptions are not widely supported on some classes of
  microcontrollers and DSPs, and they can significantly increase binary size.

**Threads, thread_local, locking: Do not use, except in optional libraries that
must work with threading**
- Rationale: The core runtime must work on systems that do not have threading
  support.

**RTTI, dynamic_cast, and `<typeid>`: Do not use.**
- Rationale: RTTI adds extra data to every virtual class. ExecuTorch doesn't
  have a strong need for `dynamic_cast` and friends, so it's better to reduce
  the binary size.

**Templates and template metaprogramming: Be careful and avoid if possible.**
- Rationale: Most templating results in code generation, and is one of the most
  common sources of binary bloat. Some use of templates is fine (e.g. an
  `ArrayRef<T>`, or code that handles multiple `ScalarType` types), but for the
  most part avoid them if possible.

## For Backend Delegate Authors

- Use [this](/docs/source/backend-delegates-integration.md) guide when
  integrating your delegate with ExecuTorch.
- Refer to [this](/docs/source/backend-delegates-dependencies.md) set of
  guidelines when including a third-party depenency for your delegate.
