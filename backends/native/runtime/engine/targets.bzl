load("@fbsource//third-party/llama.cpp:ggml_ci.bzl", "ggml_ci_labels")
load("@fbsource//tools/build_defs:default_platform_defs.bzl", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")

# Platform constraints of the ggml backend sets. Targets that link one repeat
# its constraint so builds for other platforms skip them. The CUDA and HIP sets
# also link the x86_64 Linux CPU backend, so their settings require it too, and
# stay compatible_with to match the ggml backend targets they link.
CPU_COMPAT = {"target_compatible_with": ["ovr_config//cpu:x86_64", "ovr_config//os:linux"]}
CUDA_COMPAT = {"compatible_with": ["fbsource//xplat/executorch/backends/native/runtime/engine:ggml_cuda_platform"]}
HIP_COMPAT = {"compatible_with": ["fbsource//xplat/executorch/backends/native/runtime/engine:ggml_hip_platform"]}
METAL_COMPAT = {"target_compatible_with": ["ovr_config//cpu:arm64", "ovr_config//os:macos"]}

def define_common_targets():
    # The runtime <-> compute-backend boundary: EngineHost owns process-wide
    # device state, EngineContext owns one loaded program, and EngineExecutable
    # runs one lowered Method.
    runtime.cxx_library(
        name = "engine",
        srcs = ["Engine.cpp"],
        exported_headers = [
            "Engine.h",
        ],
        deps = [
            "//executorch/backends/native/runtime:runtime",
            "//executorch/backends/native/runtime:validation",
            "//executorch/backends/native/runtime/deserialize:package",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime/graph:scalar_type",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # The engine source is backend-neutral: it picks devices from ggml's registry,
    # so each variant differs only in which ggml backend set it links. Each
    # variant repeats its backend's platform constraint so builds for other
    # platforms skip it instead of failing on the incompatible dependency.
    variants = [
        ("cpu", "fbsource//third-party/llama.cpp:ggml_cpu", CPU_COMPAT),
        ("cuda", "fbsource//third-party/llama.cpp:ggml_cuda", CUDA_COMPAT),
        ("hip", "fbsource//third-party/llama.cpp:ggml_hip", HIP_COMPAT),
    ]

    # Mac builds go through xplat.
    if is_xplat():
        variants.append(("metal", "fbsource//third-party/llama.cpp:ggml_metal", METAL_COMPAT))
    for variant, ggml_dep, compat in variants:
        runtime.cxx_library(
            name = "ggml_engine_" + variant,
            srcs = [
                "ggml/GgmlEngine.cpp",
                "ggml/GgmlExecutor.cpp",
                "ggml/GgmlGraphBuilder.cpp",
                "ggml/GgmlHostInts.cpp",
                "ggml/GgmlLowerAttention.cpp",
                "ggml/GgmlLowerConv.cpp",
                "ggml/GgmlLowerElementwise.cpp",
                "ggml/GgmlLowerLinear.cpp",
                "ggml/GgmlLowerShape.cpp",
                "ggml/GgmlTypes.cpp",
            ],
            headers = [
                "ggml/GgmlExecutor.h",
                "ggml/GgmlGraphBuilder.h",
                "ggml/GgmlHostInts.h",
                "ggml/GgmlTypes.h",
            ],
            exported_headers = ["ggml/GgmlEngine.h"],
            exported_deps = [":engine"],
            deps = [
                "//executorch/backends/native/runtime:method",
                "//executorch/backends/native/runtime:runtime",
                "//executorch/backends/native/runtime/deserialize:byte_span",
                "//executorch/backends/native/runtime/deserialize:owned_bytes",
                "//executorch/backends/native/runtime/deserialize:package",
                "//executorch/backends/native/runtime/graph:argument",
                "//executorch/backends/native/runtime/graph:graph",
                "//executorch/backends/native/runtime/graph:ids",
                "//executorch/backends/native/runtime/graph:node",
                "//executorch/backends/native/runtime/graph:scalar_type",
                "//executorch/backends/native/runtime/graph:tensor_meta",
                "//executorch/backends/native/runtime/graph:value",
                ggml_dep,
            ],
            labels = ggml_ci_labels(variant),
            # The GPU sets have no mobile flavors.
            platforms = None if variant == "cpu" else [CXX],
            visibility = ["//executorch/backends/native/..."],
            **compat
        )
