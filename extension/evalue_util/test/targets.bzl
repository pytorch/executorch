load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_test(
            name = "print_evalue_test" + aten_suffix,
            srcs = [
                "print_evalue_test.cpp",
            ],
            deps = [
                "//executorch/extension/evalue_util:print_evalue" + aten_suffix,
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util" + aten_suffix,
            ],
        )
