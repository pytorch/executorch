load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define SlimTensor library targets.

    SlimTensor is a lightweight tensor implementation for AOTI (Ahead-of-Time Inference)
    that provides a minimal, efficient tensor abstraction for ExecuTorch CUDA backend.

    This is a direct port from torchnative/standalone/slim with minimal modifications.
    """

    # Utility library (SharedPtr, SizeUtil)
    runtime.cxx_library(
        name = "util",
        exported_headers = glob(["util/*.h"]),
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            "//executorch/backends/aoti/slim/c10:c10",
        ],
    )

    # Core SlimTensor library (CPU only)
    runtime.cxx_library(
        name = "core",
        exported_headers = glob(["core/*.h"]),
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":util",
            "//executorch/backends/aoti/slim/c10:c10",
        ],
    )

    # Factory functions library
    runtime.cxx_library(
        name = "factory",
        exported_headers = glob(["factory/*.h"]),
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":core",
            "//executorch/backends/aoti/slim/c10:c10",
        ],
    )

    # CUDA support library
    runtime.cxx_library(
        name = "cuda",
        exported_headers = glob(["cuda/*.h"]),
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_preprocessor_flags = ["-DUSE_CUDA"],
        exported_deps = [
            ":core",
            "//executorch/backends/aoti/slim/c10:c10",
            "//executorch/backends/aoti/slim/c10:c10_cuda",
        ],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
    )

    # CPU-only SlimTensor library (no CUDA dependencies)
    runtime.cxx_library(
        name = "slim_tensor_cpu",
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":core",
            ":factory",
            ":util",
        ],
    )

    # Full SlimTensor library (with CUDA support)
    runtime.cxx_library(
        name = "slim_tensor",
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [
            ":core",
            ":factory",
            ":cuda",
            ":util",
        ],
    )
