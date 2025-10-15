# Troubleshooting

This page describes common issues that you may encounter when using the XNNPACK backend and how to debug and resolve them.

## XNNPACK Backend Not Found

This error indicates the XNNPACK backend is not registered with the runtime. This can happen because the backend was not compiled or linked, or because the registration code was optimized out.

The XNNPACK backend is built by default for Python, Android, iOS, and in most CMake presets.

* Set the `EXECUTORCH_BUILD_XNNPACK=ON` CMake option option when building from source.
  * Either by passing the option during CMake configuration or setting it inside the user CMake logic before including ExecuTorch.
  * See [Building from Source](/using-executorch-building-from-source).
* On iOS, link the `backend_xnnpack` [framework](/using-executorch-ios).
* If the backend is still not found, link with `WHOLE_ARCHIVE`.
   * Pass `"LINK_LIBRARY:WHOLE_ARCHIVE,xnnpack_backend>"` to `target_link_libraries` in CMake.

## Slow Performance

 * Try reducing the thread count using [_unsafe_reset_threadpool](/using-executorch-faqs.md#inference-is-slow-performance-troubleshooting).
   * Small models may benefit from using fewer threads than default.
   * Try values between 1 and 4 threads and measure performance on your model.
 * Use [op-level profiling](/tutorials/devtools-integration-tutorial) to understand which operators are taking the most time. <!-- @lint-ignore linter doesn't like this link for some reason -->
   * The XNNPACK backend provides operator-level timing for delegated operators.
 * See general performance troubleshooting tips in [Performance Troubleshooting](/using-executorch-faqs.md#inference-is-slow-performance-troubleshooting).
