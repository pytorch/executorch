load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define c10 core targets for SlimTensor.

    These headers provide c10 APIs needed by SlimTensor that are not
    available in ExecuTorch's c10 directory (which is synced from PyTorch).
    """

    runtime.cxx_library(
        name = "c10_core",
        exported_headers = [
            "Contiguity.h",
            "MemoryFormat.h",
            "SizesAndStrides.h",
            "WrapDimMinimal.h",
        ],
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
        ] + ([] if runtime.is_oss else [
            "//executorch/runtime/core/portable_type/c10/c10:aten_headers_for_executorch",
        ]),
    )
