add_executable(executorch-test-runner
    test_runner.cpp
    # TODO
    ../../../runtime/platform/runtime.cpp
)

target_link_libraries(
    executorch-test-runner
    PRIVATE executorch
    gflags
    extension_flat_tensor
    extension_flat_tensor_serialize
    extension_module
    extension_tensor
    optimized_native_cpu_ops_lib
    xnnpack_backend)
