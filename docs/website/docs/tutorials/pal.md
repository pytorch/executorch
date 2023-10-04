# Platform Abstraction Layer

DEPRECATED: This document is moving to //executorch/docs/source/runtime-platform-abstraction-layer.md

The ExecuTorch Platform Abstraction Layer (PAL) provides a way for execution
environments to override operations like:
- Getting the current time
- Printing a log statement
- Panicking the process/system

The PAL function declarations are in `//executorch/runtime/platform/platform.h`.

## Overriding the default PAL

The default PAL implementation is in `//executorch/runtime/platform/target/Posix.cpp`.
It uses `std::chrono::steady_clock` for the time, prints log messages to
`stderr`, and makes other default assumptions.

But, if they don't work for your system, you can override the default PAL by:
- Including `executorch/runtime/platform/platform.h`
- Defining an implementation of one or more of the `et_pal_*()` functions

No build system changes necessary. The default PAL functions are weak symbols,
so providing your own strong-symbol definition will override them at link time.

## Minimal PAL
If you run into build problems because your system doesn't support the functions
called by `Posix.cpp`, you can instead use the no-op minimal PAL at
`//executorch/runtime/platform/target/Posix.cpp` by building with `-c
executorch.pal_default=minimal`. This will avoid calling `fprintf()`,
`std::chrono::steady_clock`, and anything else that `Posix.cpp` uses. But since
the `Minimal.cpp` `et_pal_*()` functions are no-ops, you will need to override
all of them.
