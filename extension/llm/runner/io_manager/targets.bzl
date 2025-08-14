load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():

    for aten in (True, False):
        aten_suffix = "_aten" if aten else ""

        # Interface for IOManager. No concrete impl from this dep.
        runtime.cxx_library(
            name = "io_manager" + aten_suffix,
            exported_headers = [
                "io_manager.h",
            ],
            deps = [
                "//executorch/extension/tensor:tensor" + aten_suffix,
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/executor:program_no_prim_ops" + aten_suffix,
            ],
            visibility = [
                "@EXECUTORCH_CLIENTS",
            ],
        )
