load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define c10 library targets for SlimTensor.

    These are portable c10 utilities adapted from torchnative/standalone.
    """

    # c10 utility headers (ArrayRef, Half, BFloat16, complex, etc.)
    # Excludes CUDA-specific headers which require CUDA SDK
    runtime.cxx_library(
        name = "c10",
        exported_headers = glob(
            ["**/*.h"],
            exclude = ["cuda/**/*.h"],
        ),
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_deps = [],
    )

    # c10 CUDA-specific headers (requires CUDA SDK)
    runtime.cxx_library(
        name = "c10_cuda",
        exported_headers = glob(["cuda/*.h"]),
        visibility = ["@EXECUTORCH_CLIENTS"],
        exported_preprocessor_flags = ["-DUSE_CUDA"],
        exported_deps = [":c10"],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
    )
