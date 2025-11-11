load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    runtime.cxx_test(
        name = "backend_options_test",
        srcs = ["backend_options_test.cpp"],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/backend:interface",
            "//executorch/test/utils:utils",
        ],
    )

    runtime.cxx_test(
        name = "backend_interface_update_test",
        srcs = ["backend_interface_update_test.cpp"],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/backend:interface",
        ],
    )
