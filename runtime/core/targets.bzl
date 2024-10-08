load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def event_tracer_enabled():
    return native.read_config("executorch", "event_tracer_enabled", "false") == "true"

def get_event_tracer_flags():
    event_tracer_flags = []
    if event_tracer_enabled():
        event_tracer_flags += ["-DET_EVENT_TRACER_ENABLED"]
    return event_tracer_flags

def build_sdk():
    return native.read_config("executorch", "build_sdk", "false") == "true"

def get_sdk_flags():
    sdk_flags = []
    if build_sdk():
        sdk_flags += ["-DEXECUTORCH_BUILD_DEVTOOLS"]
    return sdk_flags

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
            "error.h",
            "freeable_buffer.h",
            "result.h",
            "span.h",
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = [
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
        ],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    for aten_mode in (True, False):
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
                "//executorch/runtime/core:core",
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                ":tag",
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
        name = "tag",
        exported_headers = [
            "tag.h",
        ],
        visibility = [
            "//executorch/...",
        ],
    )
