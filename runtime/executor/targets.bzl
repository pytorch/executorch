load("@fbsource//xplat/executorch/build:build_variables.bzl", "PROGRAM_NO_PRIM_OPS_SRCS")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

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


    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""

        runtime.cxx_library(
            name = "pte_data_map" + aten_suffix,
            srcs = [
                "pte_data_map.cpp",
            ],
            exported_headers = [
                "pte_data_map.h",
            ],
            visibility = [
                "//executorch/runtime/executor/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                "//executorch/runtime/core:core",
                "//executorch/runtime/core:named_data_map" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
            ],
            deps = [
                "//executorch/schema:program",
            ],
            exported_preprocessor_flags = [] if runtime.is_oss else ["-DEXECUTORCH_INTERNAL_FLATBUFFERS=1"],
        )

        runtime.cxx_library(
            name = "merged_data_map" + aten_suffix,
            exported_headers = [
                "merged_data_map.h",
            ],
            exported_deps = [
                "//executorch/runtime/core:named_data_map" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "program" + aten_suffix,
            exported_deps = [
                ":program_no_prim_ops" + aten_suffix,
                "//executorch/kernels/prim_ops:prim_ops_registry" + aten_suffix,
            ],
            visibility = [
                "//executorch/runtime/executor/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )

        runtime.cxx_library(
            name = "program_no_prim_ops" + aten_suffix,
            srcs = PROGRAM_NO_PRIM_OPS_SRCS + [
                "tensor_parser{}.cpp".format(aten_suffix if aten_mode else "_portable"),
            ],
            headers = [
                "platform_memory_allocator.h",
            ],
            exported_headers = [
                "method.h",
                "method_meta.h",
                "program.h",
                "tensor_parser.h",
            ],
            compiler_flags = select({
                "ovr_config//os:windows": [],
                "DEFAULT" :["-Wno-error=deprecated-declarations"]
            }),
            preprocessor_flags = _program_preprocessor_flags(),
            exported_deps = [
                ":memory_manager",
                ":pte_data_map" + aten_suffix,
                ":merged_data_map" + aten_suffix,
                "//executorch/runtime/backend:interface" + aten_suffix,
                "//executorch/runtime/core:core",
                "//executorch/runtime/core:named_data_map" + aten_suffix,
                "//executorch/runtime/core:evalue" + aten_suffix,
                "//executorch/runtime/core:event_tracer" + aten_suffix,
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
                "//executorch/runtime/kernel:kernel_runtime_context" + aten_suffix,
                "//executorch/runtime/kernel:operator_registry" + aten_suffix,
                "//executorch/runtime/platform:platform",
                "//executorch/schema:extended_header",
            ],
            deps = [
                "//executorch/schema:program",
                "//executorch/runtime/core/exec_aten/util:tensor_dimension_limit"
            ],
            visibility = [
                "//executorch/runtime/executor/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )
