load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "tensor_util_test",
        srcs = ["tensor_util_test.cpp"],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""
        preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else []

        runtime.cxx_test(
            name = "tensor_factory_test" + aten_suffix,
            srcs = ["tensor_factory_test.cpp"],
            preprocessor_flags = preprocessor_flags,
            deps = [
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util" + aten_suffix,
                "//executorch/runtime/core/portable_type/c10/c10:c10",
            ],
        )
