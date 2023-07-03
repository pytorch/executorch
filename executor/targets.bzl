load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def _program_preprocessor_flags():
    """Returns the preprocessor_flags to use when building Program.cpp"""

    # The code for flatbuffer verification can add ~30k of .text to the binary.
    # It's a valuable feature, but make it optional for space-constrained
    # systems.
    enable_verification = native.read_config(
        "executorch",
        "enable_program_verification",
        # Default value
        "true",
    )
    if enable_verification == "false":
        return ["-DET_ENABLE_PROGRAM_VERIFICATION=0"]
    elif enable_verification == "true":
        # Enabled by default.
        return []
    else:
        fail("executorch.enable_program_verification must be one of 'true' or 'false'; saw '" +
             enable_verification + "'")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "memory_manager",
        exported_headers = [
            "HierarchicalAllocator.h",
            "MemoryAllocator.h",
            "MemoryManager.h",
        ],
        exported_deps = [
            "//executorch/core:core",
            "//executorch/profiler:profiler",
        ],
        visibility = [
            "//executorch/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.cxx_library(
        name = "program",
        srcs = ["Program.cpp"],
        exported_headers = ["Program.h"],
        deps = [
            "//executorch/compiler:compiler",
            "//executorch/core:core",
            "//executorch/schema:extended_header",
            "//executorch/schema:schema",
            "//executorch/profiler:profiler",
        ],
        preprocessor_flags = _program_preprocessor_flags(),
        exported_deps = ["//executorch/core:data_loader", "//executorch/core:freeable_buffer"],
        visibility = ["//executorch/executor/...", "@EXECUTORCH_CLIENTS"],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "executor" + aten_suffix,
            srcs = [
                "Executor.cpp",
                "tensor_parser{}.cpp".format(aten_suffix),
            ],
            deps = [
                "//executorch/backends:backend",
                "//executorch/core/prim_ops:prim_ops_registry" + aten_suffix,
                "//executorch/kernels:kernel_runtime_context" + aten_suffix,
                "//executorch/profiler:profiler",
                "//executorch/schema:schema",
            ],
            exported_deps = [
                "//executorch/compiler:compiler",
                "//executorch/core:core",
                "//executorch/core:operator_registry",
                "//executorch/core/kernel_types/util:tensor_util" + aten_suffix,
                "//executorch/core/kernel_types/util:dim_order_util",
                "//executorch/core/kernel_types/util:scalar_type_util",
                "//executorch/core/values:executor_values",
                "//executorch/executor:memory_manager",
                "//executorch/core/kernel_types:kernel_types" + aten_suffix,
                ":program",
            ],
            exported_headers = [
                "Executor.h",
            ],
            headers = [
                "tensor_parser.h",
            ],
            visibility = [
                "//executorch/backends/test/...",
                "//executorch/executor/test/...",
                "//executorch/pybindings/...",
                "//executorch/test/...",
                "//executorch/util/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )
