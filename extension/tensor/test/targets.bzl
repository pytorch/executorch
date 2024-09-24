load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_test(
            name = "test" + aten_suffix,
            srcs = [
                "tensor_impl_ptr_test.cpp",
                "tensor_ptr_maker_test.cpp",
                "tensor_ptr_test.cpp",
            ],
            deps = [
                "//executorch/extension/tensor:tensor" + aten_suffix,
            ],
        )
