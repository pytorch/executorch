load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "arrayref_test",
        srcs = ["ArrayRefTest.cpp"],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "span_test",
        srcs = ["span_test.cpp"],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "StringTest",
        srcs = ["StringTest.cpp"],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "logging_test",
        srcs = [
            "LoggingTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
        compiler_flags = [
            # Turn on debug logging.
            "-DET_MIN_LOG_LEVEL=Debug",
        ],
    )

    runtime.cxx_test(
        name = "error_handling_test",
        srcs = [
            "ErrorHandlingTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "function_ref_test",
        srcs = [
            "FunctionRefTest.cpp",
        ],
        deps = [
            "//executorch/core:core",
        ],
    )

    runtime.cxx_test(
        name = "freeable_buffer_test",
        srcs = [
            "FreeableBufferTest.cpp",
        ],
        deps = [
            "//executorch/core:freeable_buffer",
        ],
    )

    runtime.cxx_test(
        name = "operator_registry_test",
        srcs = [
            "OperatorRegistryTest.cpp",
        ],
        deps = [
            "//executorch/core:operator_registry",
            "//executorch/kernels:kernel_runtime_context",
        ],
    )

    et_operator_library(
        name = "executorch_all_ops",
        include_all_operators = True,
        define_static_targets = True,
    )

    executorch_generated_lib(
        name = "test_generated_lib_1",
        deps = [
            ":executorch_all_ops",
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        visibility = [
            "//executorch/...",
        ],
    )

    runtime.export_file(
        name = "functions.yaml",
    )

    executorch_generated_lib(
        name = "specialized_kernel_generated_lib",
        deps = [
            ":executorch_all_ops",
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = ":functions.yaml",
        visibility = [
            "//executorch/...",
        ],
    )

    runtime.cxx_test(
        name = "kernel_double_registration_test",
        srcs = [
            "KernelDoubleRegistrationTest.cpp",
        ],
        deps = [
            "//executorch/core:operator_registry",
            ":specialized_kernel_generated_lib",
        ],
    )
