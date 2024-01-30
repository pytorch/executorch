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
            "memory_manager.h",
        ],
        exported_deps = [
            "//executorch/runtime/core:memory_allocator",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "program" + aten_suffix,
            srcs = [
                "method.cpp",
                "method_meta.cpp",
                "program.cpp",
                "tensor_parser_exec_aten.cpp",
                "tensor_parser{}.cpp".format(aten_suffix if aten_mode else "_portable"),
            ],
            headers = [
                "tensor_parser.h",
            ],
            exported_headers = [
                "method.h",
                "method_meta.h",
                "program.h",
            ],
            deps = [
                "//executorch/kernels/prim_ops:prim_ops_registry" + aten_suffix,
                "//executorch/runtime/backend:interface",
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
                "//executorch/runtime/core:core",
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
                "//executorch/runtime/kernel:operator_registry",
                "//executorch/runtime/platform:platform",
                "//executorch/schema:extended_header",
                "//executorch/schema:program",
                ":memory_manager",
            ],
            preprocessor_flags = _program_preprocessor_flags(),
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core:core",
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/platform:platform",
                "//executorch/runtime/core:event_tracer" + aten_suffix,
                ":memory_manager",
            ],
            visibility = [
                "//executorch/runtime/executor/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )
