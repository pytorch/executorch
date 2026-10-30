load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in get_aten_mode_options():
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

        if aten_mode:
            # Prints lean and ATen EValues from one binary.
            runtime.cxx_library(
                name = "print_evalue_lean_helper",
                srcs = ["print_evalue_lean_helper.cpp"],
                exported_headers = ["print_evalue_lean_helper.h"],
                deps = ["//executorch/extension/evalue_util:print_evalue"],
            )

            runtime.cxx_test(
                name = "print_evalue_lean_and_aten_test",
                srcs = ["print_evalue_lean_and_aten_test.cpp"],
                deps = [
                    ":print_evalue_lean_helper",
                    "//executorch/extension/evalue_util:print_evalue_aten",
                ],
            )
