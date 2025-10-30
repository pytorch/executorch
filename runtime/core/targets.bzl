load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def event_tracer_enabled():
    return native.read_config("executorch", "event_tracer_enabled", "false") == "true"

def get_event_tracer_flags():
    event_tracer_flags = []
    if event_tracer_enabled():
        event_tracer_flags += ["-DET_EVENT_TRACER_ENABLED"]
    elif not runtime.is_oss:
        event_tracer_flags += select ({
            "DEFAULT": [],
            "fbsource//xplat/executorch/tools/buck/constraints:event-tracer-enabled" : ["-DET_EVENT_TRACER_ENABLED"]
        })
    return event_tracer_flags

def build_sdk():
    return native.read_config("executorch", "build_sdk", "false") == "true"

def get_sdk_flags():
    sdk_flags = []
    if build_sdk():
        sdk_flags += ["-DEXECUTORCH_BUILD_DEVTOOLS"]
    return sdk_flags

def enable_enum_strings():
    return native.read_config("executorch", "enable_enum_strings", "true") == "true"

def get_core_flags():
    core_flags = []
    core_flags += ["-DET_ENABLE_ENUM_STRINGS=" + ("1" if enable_enum_strings() else "0")]
    return core_flags

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "core",
        exported_headers = [
            "array_ref.h",  # TODO(T157717874): Migrate all users to span and then move this to portable_type
            "data_loader.h",
            "defines.h",
            "error.h",
            "freeable_buffer.h",
            "function_ref.h",
            "result.h",
            "span.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_preprocessor_flags = get_core_flags(),
        exported_deps = [
            "//executorch/runtime/core/portable_type/c10/c10:c10",
            "//executorch/runtime/platform:platform",
        ],
    )

    runtime.cxx_library(
        name = "tensor_shape_dynamism",
        exported_headers = [
            "tensor_shape_dynamism.h",
        ],
        visibility = [
            "//executorch/runtime/core/exec_aten/...",
            "//executorch/runtime/core/portable_type/...",
        ],
    )

    runtime.cxx_library(
        name = "memory_allocator",
        exported_headers = [
            "hierarchical_allocator.h",
            "memory_allocator.h",
        ],
        exported_deps = [
            ":core",
            "//executorch/runtime/core/portable_type/c10/c10:c10",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "evalue" + aten_suffix,
            exported_headers = [
                "evalue.h",
            ],
            srcs = ["evalue.cpp"],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":core",
                ":tag",
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "event_tracer" + aten_suffix,
            exported_headers = [
                "event_tracer.h",
                "event_tracer_hooks.h",
                "event_tracer_hooks_delegate.h",
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_preprocessor_flags = get_event_tracer_flags() + get_sdk_flags(),
            exported_deps = [
                "//executorch/runtime/platform:platform",
                "//executorch/runtime/core:evalue" + aten_suffix,
            ],
        )

        runtime.cxx_library(
            name = "named_data_map" + aten_suffix,
            exported_headers = [
                "named_data_map.h",
            ],
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            exported_deps = [
                ":tensor_layout" + aten_suffix,
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
            ],
        )


        runtime.cxx_library(
            name = "tensor_layout" + aten_suffix,
            srcs = ["tensor_layout.cpp"],
            exported_headers = ["tensor_layout.h"],
            deps = [
                "//executorch/runtime/core/portable_type/c10/c10:c10",
            ],
            exported_deps = [
                ":core",
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/core/exec_aten/util:scalar_type_util" + aten_suffix,
            ],
            visibility = ["//executorch/..."],
        )

    runtime.cxx_library(
        name = "tag",
        srcs = ["tag.cpp"],
        exported_headers = [
            "tag.h",
        ],
        exported_deps = [
            ":core",
            "//executorch/runtime/platform:compiler",
        ],
        visibility = [
            "//executorch/...",
        ],
    )
