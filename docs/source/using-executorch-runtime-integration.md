# Runtime Integration

This section describes options for configuring and customizing the ExecuTorch runtime. While the pre-built packages are designed to provide an "out-of-box" experience, it is common to require additional configuration when shipping into production. ExecuTorch provides the ability to compile-time gate features, such as logging, customize system integration, and include only the operators needed to run specific models (selective build).

## Logging

ExecuTorch runtime code includes logging statements at various levels, to aid with integration and debugging. Logging inclusion is controlled at build time by the `EXECUTORCH_ENABLE_LOGGING` and `EXECUTORCH_LOG_LEVEL` CMake options. Having these exposed as compile-time configuration allows for all logging-related code to be excluded when not used, which is critical for resource constrained systems.

Logging is sent to STDOUT and STDERR by default on host platforms, and is redirected to OS-specific logging on Android and iOS. See [Platform Abstraction Layer](#platform-abstraction-layer-pal) below for more information on log routing.

To configure log level when building from source, specify `EXECUTORCH_ENABLE_LOGGING` as on or off and `EXECUTORCH_LOG_LEVEL` as one of debug, info, error, or fatal. Logging is enabled by default in debug builds and disabled in release. Log level defaults to info.

See [Building from Source](using-executorch-building-from-source.md) for more information.

```
cmake -b cmake-out -DEXECUTORCH_ENABLE_LOGGING=ON -DEXECUTORCH_LOG_LEVEL=DEBUG ...
```

## Platform Abstraction Layer (PAL)

The ExecuTorch Platform Abstraction Layer, or PAL, is a glue layer responsible for providing integration with a particular host system. This includes log routing, timestamps, and abort handling. ExecuTorch provides a default implementation for POSIX-compliant targets, as well as a Android and iOS-specific implementations under the appropriate extensions.

For non-POSIX-compliant systems, a minimal no-op PAL implementation is provided. It is expected that users override the relevant PAL methods in order to enable logging, timestamps, and aborts. The minimal PAL can be selected by building with `-DEXECUTORCH_PAL_DEFAULT=minimal`.

### Overriding the PAL

Overriding the default PAL implementation is commonly done to route logs to a user-specified destination or to provide PAL functionality on embedded systems. The PAL can be overriden usinn runtime APIs or at link time. Prefer the runtime API unless you specifically need link-time overrides.

### Runtime PAL Registration

To register a custom PAL implementation, take the following steps:

- Include
  [`executorch/runtime/platform/platform.h`](https://github.com/pytorch/executorch/blob/main/runtime/platform/platform.h)
  in one of your application's `.c` or `.cpp` files.
- Create an instance of the [PalImpl](https://github.com/pytorch/executorch/blob/7b39a0ce63bfb5124d4d29cfb6c8af85a3c580ba/runtime/platform/platform.h#L163) struct.
  - Set one or more fields to custom PAL function implementations. Leave fields as null to use the default platform implementation.
  - The PalImpl struct provides a [create](https://github.com/pytorch/executorch/blob/7b39a0ce63bfb5124d4d29cfb6c8af85a3c580ba/runtime/platform/platform.h#L168) method for this purpose.
- Call `executorch::platform::register_pal(pal_impl)` to register the implementation.
  - This can be done from as as a global constructor, as in the example below.

Here is a complete example from [pybindings.cpp](https://github.com/pytorch/executorch/blob/7b39a0ce63bfb5124d4d29cfb6c8af85a3c580ba/extension/pybindings/pybindings.cpp#L1178), where logs are redirected to show up properly in a Python notebook environment.

```cpp
namespace {
  void emit_log_message(
      et_timestamp_t timestamp,
      et_pal_log_level_t level,
      const char* filename,
      ET_UNUSED const char* function,
      size_t line,
      const char* message,
      ET_UNUSED size_t length) {
    std::cerr << "[" << filename << ":" << line << "] " << message << std::endl;
  }

  runtime::PalImpl build_pal() {
    return runtime::PalImpl::create(emit_log_message, __FILE__);
  }

  // Update PAL to redirect logs.
  ET_UNUSED bool registration_result = runtime::register_pal(build_pal());
}
```

### Weak Symbol Override
ExecuTorch also provides a link-time method to override the PAL using weak symbols. This method is primarily maintained for backwards compatibility.

To override one or more PAL methods, take the following steps:

- Include
  [`executorch/runtime/platform/platform.h`](https://github.com/pytorch/executorch/blob/main/runtime/platform/platform.h)
  in one of your application's `.c` or `.cpp` files.
- Define an implementation of one or more of the `et_pal_*()` functions.

The default PAL functions are weak symbols, so providing your own strong-symbol
definition can override them at link time. To ensure that your definitions take
precedence, you may need to ensure that the strong definitions precede the weak
definitions in the link order.

See [runtime/platform/platform.h](https://github.com/pytorch/executorch/blob/main/runtime/platform/platform.h) for the PAL function signatures and [runtime/platform/default/posix.cpp](https://github.com/pytorch/executorch/blob/main/runtime/platform/default/posix.cpp) for the reference POSIX implementation.

## Kernel Libraries

During export, a model is broken down into a list of operators, each providing some fundamental computation. Adding two tensors is an operator, as is convolution. Each operator requires a corresponding operator kernel to perform the computation on the target hardware. ExecuTorch backends are the preferred way to do this, but not all operators are supported on all backends.

To handle this, ExecuTorch provides two implementations - the *portable* and *optimized* kernel libraries. The portable kernel library provides full support for all operators in a platform-independent manner. The optimized library carries additional system requirements, but is able to leverage multithreading and vectorized code to achieve greater performance. Operators can be drawn for both for a single build, allowing the optimized library to be used where available with the portable library as a fallback.

The choice of kernel library is transparent to the user when using mobile pre-built packages. However, it is important when building from source, especially on embedded systems. On mobile, the optimized operators are preferred where available. See [Overview of ExecuTorch's Kernel Libraries](kernel-library-overview.md) for more information.

## Selective Build

By default, ExecuTorch ships with all supported operator kernels, allowing it to run any supported model at any precision. This comes with a binary size of several megabytes, which may be undesirable for production use cases or resource constrained systems. To minimize binary size, ExecuTorch provides selective build functionality, in order to include only the operators needed to run specific models.

Note the selective build only applies to the portable and optimized kernel libraries. Delegates do not participate in selective build and can be included or excluded by linking indivually. See [Kernel Library Selective Build](kernel-library-selective-build.md) for more information.
