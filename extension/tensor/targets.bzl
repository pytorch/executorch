load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        runtime.cxx_library(
            name = "tensor" + aten_suffix,
            srcs = [
                "tensor_impl_ptr.cpp",
                "tensor_ptr.cpp",
                "tensor_ptr_maker.cpp",
            ],
            exported_headers = [
                "tensor.h",
                "tensor_impl_ptr.h",
                "tensor_ptr.h",
                "tensor_ptr_maker.h",
            ],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "//executorch/runtime/core/exec_aten/util:dim_order_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
            ],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
            ],
        )
