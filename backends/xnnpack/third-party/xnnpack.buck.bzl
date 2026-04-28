load(
    ":xnnpack_src_defs.bzl",
    "LOGGING_SRCS",
    "OPERATOR_SRCS",
    "SUBGRAPH_SRCS",
    "TABLE_SRCS",
    "get_xnnpack_headers",
    "get_ukernel_config_srcs",
    "prod_srcs_for_arch_wrapper",
)

def define_xnnpack():
    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "interface",
        headers = get_xnnpack_headers(),
        header_namespace = "",
        exported_headers = {
            "xnnpack.h": "XNNPACK/include/xnnpack.h",
        },
        compiler_flags = [
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        visibility = ["PUBLIC"],
        exported_deps = [
            ":pthreadpool",
        ],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "operators",
        srcs = OPERATOR_SRCS + [
            "XNNPACK/src/allocator.c",
            "XNNPACK/src/cache.c",
            "XNNPACK/src/indirection.c",
            "XNNPACK/src/memory.c",
            "XNNPACK/src/mutex.c",
            "XNNPACK/src/normalization.c",
            "XNNPACK/src/operator-utils.c",
            "XNNPACK/src/reference/packing.cc",
        ],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
            "-DXNN_ENABLE_GEMM_M_SPECIALIZATION=0",
        ],
        exported_deps = [
            ":FP16",
            ":FXdiv",
            ":clog",
            ":interface",
            ":ukernels_f16c",
            ":cpuinfo",
        ],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "subgraph",
        srcs = SUBGRAPH_SRCS + ["XNNPACK/src/datatype.c"],
        compiler_flags = [
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ],
        headers = get_xnnpack_headers(),
        header_namespace = "",
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
            "-DXNN_ENABLE_SPARSE=0",
            "-DXNN_ENABLE_GEMM_M_SPECIALIZATION=0",
            "-DXNN_ENABLE_MEMOPT",
        ],
        exported_deps = [
            ":FP16",
            ":FXdiv",
            ":clog",
            ":interface",
        ],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "tables",
        srcs = TABLE_SRCS,
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":FXdiv",
            ":clog",
            ":interface",
        ],
    )

    DEFAULT_DUMMY_SRC = []

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_scalar",
        srcs = prod_srcs_for_arch_wrapper("scalar"),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O3",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
            "-fno-fast-math",
            "-fno-math-errno",
            "-ffp-contract=off",
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":FXdiv",
            ":interface",
        ],
    )

    ARMSIMD32_COMPILER_FLAGS = [
        "-marm",
        "-march=armv6",
        "-mfpu=vfp",
        "-munaligned-access",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_armsimd32",
        srcs = select({
            "DEFAULT": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("armsimd32"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
            "-fno-fast-math",
            "-fno-math-errno",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": ARMSIMD32_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":FXdiv",
            ":interface",
        ],
    )

    FP16ARITH_COMPILER_FLAGS = [
        "-marm",
        "-march=armv8.2-a+fp16",
        # GCC emits wrong directives for assembler with -mfpu=fp-armv8
        "-mfpu=neon-fp-armv8",
        # For vsqrth_f16 polyfill using sqrtf
        "-fno-math-errno",
        # For vminh_f16/vmaxh_f16 polyfills using compare + select
        "-ffinite-math-only",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_fp16arith",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("fp16arith"),
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
            "-fno-fast-math",
            "-fno-math-errno",
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": FP16ARITH_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":FXdiv",
            ":interface",
        ],
    )

    SSE_COMPILER_FLAGS = ["-msse"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_sse",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("sse"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": SSE_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    SSE2_COMPILER_FLAGS = ["-msse2"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_sse2",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("sse2"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": SSE2_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    SSE2_FMA_COMPILER_FLAGS = [
        "-msse2",
        "-mno-sse3",
    ]

    native.cxx_library(
        name = "ukernels_sse2fma",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("sse2fma"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": SSE2_FMA_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    SSE3_COMPILER_FLAGS = ["-mssse3"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_ssse3",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("ssse3"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": SSE3_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    SSE41_COMPILER_FLAGS = ["-msse4.1"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_sse41",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("sse41"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": SSE41_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVX_COMPILER_FLAGS = ["-mavx"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avx",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avx"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVX_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    F16C_COMPILER_FLAGS = ["-mf16c"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_f16c",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("f16c"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": F16C_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    FMA3_COMPILER_FLAGS = [
        "-mfma",
        "-mf16c",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_fma3",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("fma3"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": FMA3_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVX2_COMPILER_FLAGS = [
        "-mavx2",
        "-mfma",
        "-mf16c",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avx2",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avx2"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVX2_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVX512F_COMPILER_FLAGS = ["-mavx512f"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avx512",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avx512f"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVX512F_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVX512SKX_COMPILER_FLAGS = [
        "-mavx512f",
        "-mavx512cd",
        "-mavx512bw",
        "-mavx512dq",
        "-mavx512vl",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avx512skx",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avx512skx"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVX512SKX_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    NEON_COMPILER_FLAGS = [
        "-march=armv7-a",
        "-fpu=neon",
        "-mfloat-abi=softfp",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_asm",
        srcs = select({
            "DEFAULT": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("aarch32"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        platform_compiler_flags = [
            (
                "(aarch64|arm64)",
                [
                    "-march=armv8.2-a+fp16+dotprod",
                ],
            ),
            (
                "(aarch32|arm32)",
                [
                    "-marm",
                    "-march=armv8.2-a+dotprod",
                    "-mfpu=neon-fp-armv8",
                ],
            ),
        ],
        compiler_flags = [
            "-O2",
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("neon"),
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": NEON_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVX512VBMI_COMPILER_FLAGS = ["-mavx512vbmi"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avx512vbmi",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avx512vbmi"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVX512VBMI_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    NEON64_AARCH64_COMPILER_FLAGS = []

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon_aarch64",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("neon_aarch64"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": NEON64_AARCH64_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    NEON_FP16_COMPILER_FLAGS = ["-mfpu=neon-fp16"]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon_fp16",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("neonfp16"),
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": NEON_FP16_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":interface",
            ":FP16",
        ],
    )

    NEON32_FMA_COMPILER_FLAGS = ["-mfpu=neon-vfp4"]
    NEON64_FMA_COMPILER_FLAGS = [
        "-march=armv8-a",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon_fma",
        srcs = select({
            "DEFAULT": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonfma"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonfma") + prod_srcs_for_arch_wrapper("neonfma_aarch64"),
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": NEON32_FMA_COMPILER_FLAGS,
            "ovr_config//cpu:arm64": NEON64_FMA_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    NEON64_V8_COMPILER_FLAGS = [
        "-march=armv8-a",
    ]

    NEON32_V8_COMPILER_FLAGS = [
        "-march=armv8-a",
        "-mfpu=neon-fp-armv8",
        "-mfloat-abi=softfp",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon_v8",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("neonv8"),
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": NEON64_V8_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": NEON32_V8_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    NEON64_FP16ARITH_COMPILER_FLAGS = ["-march=armv8.2-a+fp16"]
    NEON32_FP16ARITH_COMPILER_FLAGS = [
        "-marm",
        "-march=armv8.2-a+fp16",
        "-mfpu=neon-fp-armv8",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon_fp16arith",
        srcs = select({
            "DEFAULT": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neonfp16arith"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neonfp16arith") + prod_srcs_for_arch_wrapper("neonfp16arith_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": NEON64_FP16ARITH_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": NEON32_FP16ARITH_COMPILER_FLAGS,
            "ovr_config//cpu:arm64": NEON64_FP16ARITH_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    NEONDOTFP16ARITH_COMPILER_FLAGS = [
        "-marm",
        "-march=armv8.2-a+dotprod+fp16",
        "-mfpu=neon-fp-armv8",
    ]

    NEONDOTFP16ARITH_AARCH64_COMPILER_FLAGS = [
        "-march=armv8.2-a+dotprod+fp16",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neondotfp16arith",
        srcs = select({
            "DEFAULT": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neondotfp16arith"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neondotfp16arith") + prod_srcs_for_arch_wrapper("neondotfp16arith_aarch64"),
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "ovr_config//cpu:arm32": NEONDOTFP16ARITH_COMPILER_FLAGS,
            "ovr_config//cpu:arm64": NEONDOTFP16ARITH_AARCH64_COMPILER_FLAGS,
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    NEON64_DOT_COMPILER_FLAGS = ["-march=armv8.2-a+dotprod"]

    NEON32_DOT_COMPILER_FLAGS = [
        "-march=armv8.2-a+dotprod",
        "-mfpu=neon-fp-armv8",
        "-mfloat-abi=softfp",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon_dot",
        srcs = select({
            "DEFAULT": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm32": prod_srcs_for_arch_wrapper("neondot"),
            "ovr_config//cpu:arm64": prod_srcs_for_arch_wrapper("neondot") + prod_srcs_for_arch_wrapper("neondot_aarch64"),
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": [],
            "ovr_config//cpu:arm32": NEON32_DOT_COMPILER_FLAGS,
            "ovr_config//cpu:arm64": NEON64_DOT_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    NEON32_I8MM_COMPILER_FLAGS = [
        "-marm",
        "-march=armv8.2-a+i8mm+fp16",
        "-mfpu=neon-fp-armv8",
    ]

    NEON64_I8MM_COMPILER_FLAGS = [
        "-march=armv8.2-a+i8mm+fp16",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_neon_i8mm",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("neoni8mm"),
            "ovr_config//cpu:x86_32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:x86_64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": NEON64_I8MM_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": NEON32_I8MM_COMPILER_FLAGS,
            "ovr_config//cpu:x86_32": [],
            "ovr_config//cpu:x86_64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVX512VNNI_COMPILER_FLAGS = [
        "-mavx512f",
        "-mavx512cd",
        "-mavx512bw",
        "-mavx512dq",
        "-mavx512vl",
        "-mavx512vnni",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avx512vnni",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avx512vnni"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVX512VNNI_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AMD64_COMPILER_FLAGS = [
        "-mf16c",
        "-mfma",
        "-mavx512f",
        "-mavx512cd",
        "-mavx512bw",
        "-mavx512dq",
        "-mavx512vl",
        "-mavx512vnni",
        "-mgfni",
    ]
    native.cxx_library(
        name = "ukernels_amd64",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("amd64"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AMD64_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVX512VNNIGFNI_COMPILER_FLAGS = AVX512VNNI_COMPILER_FLAGS + [
        "-mgfni",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avx512vnnigfni",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avx512vnnifgni"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVX512VNNIGFNI_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    AVXVNNI_COMPILER_FLAGS = [
        "-mavx2",
        "-mavxvnni",
        "-mf16c",
        "-mfma",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "ukernels_avxvnni",
        srcs = select({
            "DEFAULT": prod_srcs_for_arch_wrapper("avxvnni"),
            "ovr_config//cpu:arm32": DEFAULT_DUMMY_SRC,
            "ovr_config//cpu:arm64": DEFAULT_DUMMY_SRC,
        }),
        headers = get_xnnpack_headers(),
        header_namespace = "",
        compiler_flags = [
            "-O2",
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ] + select({
            "DEFAULT": AVXVNNI_COMPILER_FLAGS,
            "ovr_config//cpu:arm32": [],
            "ovr_config//cpu:arm64": [],
        }),
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
        ],
        exported_deps = [
            ":FP16",
            ":interface",
        ],
    )

    COMMON_XNNPACK_DEPS = [
        ":operators",
        ":subgraph",
        ":tables",
        ":ukernels_scalar",
    ]

    X86_64_XNNPACK_DEPS = [
        ":ukernels_avx",
        ":ukernels_avx2",
        ":ukernels_avx512",
        ":ukernels_avx512skx",
        ":ukernels_f16c",
        ":ukernels_fma3",
        ":ukernels_sse",
        ":ukernels_sse2",
        ":ukernels_sse2fma",
        ":ukernels_sse41",
        ":ukernels_ssse3",
        ":ukernels_avx512vbmi",
        ":ukernels_avx512vnnigfni",
        ":ukernels_avx512vnni",
        ":ukernels_avxvnni",
        ":ukernels_amd64",
    ]

    ARM_XNNPACK_DEPS = [
        ":ukernels_armsimd32",
        ":ukernels_fp16arith",
        ":ukernels_asm",
        ":ukernels_neon",
        ":ukernels_neon_aarch64",
        ":ukernels_neon_fp16",
        ":ukernels_neon_fma",
        ":ukernels_neon_v8",
        ":ukernels_neon_fp16arith",
        ":ukernels_neon_dot",
        ":ukernels_neon_i8mm",
        ":ukernels_neondotfp16arith",
    ]

    # @lint-ignore BUCKLINT: native and fb_native are explicitly forbidden in fbcode.
    native.cxx_library(
        name = "XNNPACK",
        srcs = get_ukernel_config_srcs() + LOGGING_SRCS + [
            "XNNPACK/src/init.c",
            "XNNPACK/src/params.c",
            "XNNPACK/src/configs/hardware-config.c",
            "XNNPACK/src/microparams-init.c",
            "XNNPACK/src/microkernel-utils.c",
            "XNNPACK/src/reference/binary-elementwise.cc",
            "XNNPACK/src/reference/unary-elementwise.cc",
            "XNNPACK/src/pack-lh.cc",
        ],
        headers = get_xnnpack_headers(),
        exported_headers = {
            "xnnpack.h": "XNNPACK/include/xnnpack.h",
        },
        header_namespace = "",
        compiler_flags = [
            "-Wno-error=missing-braces",  # required since the SGX toolchain does not have this by default
        ],
        preferred_linkage = "static",
        preprocessor_flags = [
            "-DXNN_LOG_LEVEL=0",
            "-DXNN_ENABLE_MEMOPT",
            "-DXNN_ENABLE_SPARSE=0",
            "-DXNN_ENABLE_ASSEMBLY",
            "-DXNN_ENABLE_GEMM_M_SPECIALIZATION",
            "-DXNN_ENABLE_ARM_DOTPROD",
            "-DXNN_ENABLE_CPUINFO",
            # "-DXNN_ENABLE_DWCONV_MULTIPLASS=0",
            "-DXNN_ENABLE_ARM_I8MM=1",
            "-DXNN_ENABLE_ARM_FP16_VECTOR=1",
            "-DXNN_ENABLE_AVX512F=1",
            "-DXNN_ENABLE_AVX512SKX=1",
            "-DXNN_ENABLE_AVX512VNNI=1",
            "-DXNN_ENABLE_AVX512VBMI=1",
            "-DXNN_ENABLE_AVXVNNI=0",
            "-DXNN_ENABLE_AVXVNNIINT8=0",
            "-DXNN_ENABLE_AVX512FP16=0",
            "-DXNN_ENABLE_AVX512VNNIGFNI=0",
            "-DXNN_ENABLE_AVX512BF16=0",
            "-DXNN_ENABLE_AVX256VNNIGFNI=0",
            "-DXNN_ENABLE_AVX512AMX=0",
            "-DXNN_ENABLE_AVX256SKX=0",
            "-DXNN_ENABLE_AVX256VNNI=0",
        ],
        visibility = ["PUBLIC"],
        exported_deps = COMMON_XNNPACK_DEPS + [
            ":FP16",
            ":pthreadpool",
            ":interface",
            ":cpuinfo",
        ] + select({
            "DEFAULT": X86_64_XNNPACK_DEPS,
            "ovr_config//cpu:arm32": ARM_XNNPACK_DEPS,
            "ovr_config//cpu:arm64": ARM_XNNPACK_DEPS,
        }),
    )
