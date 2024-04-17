# Portable C++ Programming

NOTE: This document covers the code that needs to build for and execute in
target hardware environments. This applies to the core execution runtime, as
well as kernel and backend implementations in this repo. These rules do not
necessarily apply to code that only runs on the development host, like authoring
or build tools.

The ExecuTorch runtime code is intendend to be portable, and should build for a
wide variety of systems, from servers to mobile phones to DSPs, from POSIX to
Windows to bare-metal environments.

This means that it can't assume the existence of:
- Files
- Threads
- Exceptions
- `stdout`, `stderr`
- `printf()`, `fprintf()`
- POSIX APIs and concepts in general

It also can't assume:
- 64 bit pointers
- The size of a given integer type
- The signedness of `char`

To keep the binary size to a minimum, and to keep tight control over memory
allocation, the code may not use:
- `malloc()`, `free()`
- `new`, `delete`
- Most `stdlibc++` types; especially container types that manage their own
  memory like `string` and `vector`, or memory-management wrapper types like
  `unique_ptr` and `shared_ptr`.

And to help reduce complexity, the code may not depend on any external
dependencies except:
- `flatbuffers` (for `.pte` file deserialization)
- `flatcc` (for event trace serialization)
- Core PyTorch (only for ATen mode)

## Platform Abstraction Layer (PAL)

To avoid assuming the capabilities of the target system, the ExecuTorch runtime
lets clients override low-level functions in its Platform Abstraction Layer
(PAL), defined in `//executorch/runtime/platform/platform.h`, to perform operations
like:
- Getting the current timestamp
- Printing a log message
- Panicking the system

## Memory Allocation

Instead of using `malloc()` or `new`, the runtime code should allocate memory
using the `MemoryManager` (`//executorch/runtime/executor/memory_manager.h`)
provided by the client.

## File Loading

Instead of loading files directly, clients should provide buffers with the data
already loaded, or wrapped in types like `DataLoader`.

## Integer Types

ExecuTorch runtime code should not assume anything about the sizes of primitive
types like `int`, `short`, or `char`. For example, the C++ standard only
guarantees that `int` will be at least 16 bits wide. And ARM toolchains treat
`char` as unsigned, while other toolchains often treat it as signed.

Instead, the runtime APIs use a set of more predictable, but still standard,
integer types:
- `<cstdint>` types like `uint64_t`, `int32_t`; these types guarantee the bit
  width and signedness, regardless of the architecture. Use these types when you
  need a very specific integer width.
- `size_t` for counts of things, or memory offsets. `size_t` is guaranteed to be
  big enough to represent any memory byte offset; i.e., it will be as wide as
  the native pointer type for the target system. Prefer using this instead of
  `uint64_t` for counts/offsets so that 32-bit systems don't need to pay for the
  unnecessary overhead of a 64-bit value.
- `ssize_t` for some ATen-compatibility situations where `Tensor` returns a
  signed count. Prefer `size_t` when possible.

## Floating Point Arithmetic

Not every system has support for floating point arithmetic: some don't even enable
floating point emulation in their toolchains. Therefore, the core runtime code
must not perform any floating point arithmetic at runtime, although it is ok to
simply create or manage `float` or `double` values (e.g., in an `EValue`).

Kernels, being outside of the core runtime, are allowed to perform floating point
arithmetic. Though some kernels may choose not to, so that they can run on systems
without floating point support.

## Logging

Instead of using `printf()`, `fprintf()`, `cout`, `cerr`, or a library like
`folly::logging` or `glog`, the ExecuTorch runtime provides the `ET_LOG`
interface in `//executorch/runtime/platform/log.h` and the `ET_CHECK` interface in
`//executorch/runtime/platform/assert.h`. The messages are printed using a hook in the PAL,
which means that clients can redirect them to any underlying logging system, or
just print them to `stderr` if available.

### Logging Format Portability

#### Fixed-Width Integers

When you have a log statement like
```
int64_t value;
ET_LOG(Error, "Value %??? is bad", value);
```
what should you put for the `%???` part, to match the `int64_t`? On different
systems, the `int64_t` typdef might be `int`, `long int`, or `long long int`.
Picking a format like `%d`, `%ld`, or `%lld` might work on one target, but break
on the others.

To be portable, the runtime code uses the standard (but admittedly awkward)
helper macros from `<cinttypes>`. Each portable integer type has a corresponding
`PRIn##` macro, like
- `int32_t` -> `PRId32`
- `uint32_t` -> `PRIu32`
- `int64_t` -> `PRId64`
- `uint64_t` -> `PRIu64`
- See https://en.cppreference.com/w/cpp/header/cinttypes for more

These macros are literal strings that can concatenate with other parts of the
format string, like
```
int64_t value;
ET_LOG(Error, "Value %" PRId64 " is bad", value);
```
Note that this requires chopping up the literal format string (the extra double
quotes). It also requires the leading `%` before the macro.

But, by using these macros, you're guaranteed that the toolchain will use the
appropriate format pattern for the type.

#### `size_t`, `ssize_t`

Unlike the fixed-width integer types, format strings already have a portable
way to handle `size_t` and `ssize_t`:
- `size_t` -> `%zu`
- `ssize_t` -> `%zd`

#### Casting

Sometimes, especially in code that straddles ATen and lean mode, the type of the
value itself might be different across build modes. In those cases, cast the
value to the lean mode type, like:
```
ET_CHECK_MSG(
    input.dim() == output.dim(),
    "input.dim() %zd not equal to output.dim() %zd",
    (ssize_t)input.dim(),
    (ssize_t)output.dim());
```
In this case, `Tensor::dim()` returns `ssize_t` in lean mode, while
`at::Tensor::dim()` returns `int64_t` in ATen mode. Since they both conceptually
return (signed) counts, `ssize_t` is the most appropriate integer type.
`int64_t` would work, but it would unnecessarily require 32-bit systems to deal
with a 64-bit value in lean mode.

This is the only situation where casting should be necessary, when lean and ATen
modes disagree. Otherwise, use the format pattern that matches the type.
