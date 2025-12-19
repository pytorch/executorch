load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Define c10 library targets for SlimTensor.

    These are portable c10 utilities adapted from torchnative/standalone.
    Many utility headers are now thin wrappers that reuse implementations
    from executorch/runtime/core/portable_type/c10/c10.

    Headers reused from portable_type/c10:
    - Macros.h (with STANDALONE_* -> C10_* mappings)
    - bit_cast.h
    - irange.h
    - floating_point_utils.h
    - TypeSafeSignMath.h
    - llvmMathExtras.h
    - safe_numerics.h

    SlimTensor-specific headers (kept due to additional features):
    - Half.h, Half-inl.h (SlimTensor has its own inline implementation)
    - BFloat16.h, BFloat16-inl.h, BFloat16-math.h
    - complex.h (has complex<Half> specialization)
    - Float8_* types (not in portable_type)
    - Quantized types (qint8, quint8, etc.)
    - Array.h, accumulate.h (SlimTensor-specific utilities)
    - core/* headers (Device, Scalar, ScalarType, etc.)
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
        exported_deps = [
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
            # Reuse c10 utility implementations from portable_type
            "//executorch/runtime/core/portable_type/c10/c10:c10",
        ],
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
