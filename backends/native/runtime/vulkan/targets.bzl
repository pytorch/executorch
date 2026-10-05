load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")

def define_common_targets():
    runtime.cxx_library(
        name = "constant_materialization_tracker",
        srcs = ["VulkanConstantMaterializationTracker.cpp"],
        exported_headers = [
            "VulkanConstantMaterializationTracker.h",
        ],
        visibility = ["//executorch/backends/native/..."],
    )

    # Host-only for now; Android integration is not wired up.
    if is_xplat():
        return

    runtime.cxx_library(
        name = "vulkan_passes",
        srcs = [
            "passes/FuseQuantizedEmbedding.cpp",
            "passes/FuseQuantizedLinear.cpp",
            "passes/InsertPrepack.cpp",
            "passes/LowerHFAttention.cpp",
            "passes/LowerRMSNorm.cpp",
            "passes/MaterializeViewCopies.cpp",
        ],
        exported_headers = [
            "passes/FuseQuantizedEmbedding.h",
            "passes/FuseQuantizedLinear.h",
            "passes/InsertPrepack.h",
            "passes/LowerHFAttention.h",
            "passes/LowerRMSNorm.h",
            "passes/MaterializeViewCopies.h",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime:method",
            "//executorch/backends/native/runtime/graph:graph",
        ],
        deps = [
            "//executorch/backends/native/runtime/graph:graph_utils",
        ],
        visibility = ["PUBLIC"],
    )

    # The Vulkan engine: implements the EngineContext / EngineExecutable
    # boundary by lowering a native Method's Graph onto ET-VK's ComputeGraph,
    # reusing every ET-VK compute shader and prepack path via the operator
    # registry.
    runtime.cxx_library(
        name = "vulkan_engine",
        srcs = [
            "VulkanEngine.cpp",
            "VulkanEngineExecutable.cpp",
        ],
        headers = ["VulkanEngineExecutable.h"],
        exported_headers = [
            "VulkanEngine.h",
        ],
        exported_deps = [
            # The header exposes only the interface; ET-VK stays internal.
            "//executorch/backends/native/runtime/engine:engine",
        ],
        deps = [
            "//executorch/backends/native/runtime:method",
            "//executorch/backends/native/runtime/deserialize:package",
            "//executorch/backends/native/runtime/vulkan:constant_materialization_tracker",
            "//executorch/backends/native/runtime/graph:argument",
            "//executorch/backends/native/runtime/graph:graph",
            "//executorch/backends/native/runtime/graph:ids",
            "//executorch/backends/native/runtime/graph:node",
            "//executorch/backends/native/runtime/graph:memory_planning",
            "//executorch/backends/native/runtime/graph:scalar",
            "//executorch/backends/native/runtime/graph:scalar_type",
            "//executorch/backends/native/runtime/graph:tensor_meta",
            "//executorch/backends/native/runtime/graph:value",
            "//executorch/backends/native/runtime/graph:graph_utils",
            "//executorch/runtime/core:core",
            ":vulkan_passes",
            # ET-VK: ComputeGraph + operator registry + shaders (link_whole, so
            # the static op registrations survive), and the Vulkan API layer.
            ":xplat_vulkan_runtime",
        ],
        visibility = ["PUBLIC"],
    )
