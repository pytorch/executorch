load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_test(
            name = "test" + aten_suffix,
            srcs = [
                "tensor_accessor_test.cpp",
                "tensor_ptr_maker_test.cpp",
                "tensor_ptr_test.cpp",
            ],
            deps = [
                "//executorch/extension/tensor:tensor" + aten_suffix,
            ],
        )

    # tensor_ptr_device_test.cpp is guarded by `#ifndef USE_ATEN_LIB` from top to bottom, so
    # it only has tests to run in portable mode. Defining it once, outside the aten_mode
    # loop, keeps its `tensor` dependency honest: the `_aten` variant used to link the
    # portable library and then compile zero tests.
    runtime.cxx_test(
        name = "tensor_ptr_device_test",
        srcs = [
            "tensor_ptr_device_test.cpp",
        ],
        deps = [
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/core:device_allocator",
            "//executorch/runtime/core/test:mock_cuda_allocator",
        ],
    )
