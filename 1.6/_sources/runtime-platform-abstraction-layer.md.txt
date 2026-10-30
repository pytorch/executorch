# Runtime Platform Abstraction Layer (PAL)

The ExecuTorch _Platform Abstraction Layer_ (PAL) provides a way for execution
environments to override operations like:
- Getting the current time.
- Printing a log statement.
- Panicking the process/system.

The PAL function declarations are in
[`executorch/runtime/platform/platform.h`](https://github.com/pytorch/executorch/blob/main/runtime/platform/platform.h).

## Overriding the default PAL

The default PAL implementation is in
[`executorch/runtime/platform/default/posix.cpp`](https://github.com/pytorch/executorch/blob/main/runtime/platform/default/posix.cpp).
It uses `std::chrono::steady_clock` for the time, prints log messages to
`stderr`, and makes other default assumptions.

But, if they don't work for your system, you can override the default PAL by:
- Including
  [`executorch/runtime/platform/platform.h`](https://github.com/pytorch/executorch/blob/main/runtime/platform/platform.h)
  in one of your application's `.c` or `.cpp` files.
- Defining an implementation of one or more of the `et_pal_*()` functions.

The default PAL functions are weak symbols, so providing your own strong-symbol
definition can override them at link time. To ensure that your definitions take
precedence, you may need to ensure that the strong definitions precede the weak
definitions in the link order.

## Minimal PAL

If you run into build problems because your system doesn't support the functions
called by `posix.cpp`, you can instead use the no-op minimal PAL at
[`executorch/runtime/platform/default/minimal.cpp`](https://github.com/pytorch/executorch/blob/main/runtime/platform/default/minimal.cpp)
by passing `-DEXECUTORCH_PAL_DEFAULT=minimal` to `cmake`. This will avoid
calling `fprintf()`, `std::chrono::steady_clock`, and anything else that
`posix.cpp` uses. But since the `minimal.cpp` `et_pal_*()` functions are no-ops,
you will need to override all of them.
