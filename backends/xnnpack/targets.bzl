load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.genrule(
        name = "gen_xnnpack_schema",
        srcs = [
            "serialization/schema.fbs",
        ],
        # We're only generating a single file, so it seems like we could use
        # `out`, but `flatc` takes a directory as a parameter, not a single
        # file. Use `outs` so that `${OUT}` is expanded as the containing
        # directory instead of the file itself.
        outs = {
            "xnnpack_schema_generated.h": ["schema_generated.h"],
        },
        cmd = " ".join([
            "$(exe fbsource//third-party/flatbuffers/fbsource_namespace:flatc)",
            "--cpp",
            "--cpp-std c++11",
            "--scoped-enums",
            "-o ${OUT}",
            "${SRCS}",
        ]),
        default_outs = ["."],
    )

    runtime.cxx_library(
        name = "xnnpack_schema",
        srcs = [],
        exported_headers = {
            "xnnpack_schema_generated.h": ":gen_xnnpack_schema[xnnpack_schema_generated.h]",
        },
        exported_deps = [
            "fbsource//third-party/flatbuffers/fbsource_namespace:flatbuffers-api",
        ],
    )

    runtime.cxx_library(
        name = "xnnpack_backend",
        srcs = native.glob([
            "runtime/*.cpp",
        ]),
        headers = native.glob([
            "runtime/*.h",
        ]),
        visibility = [
            "//executorch/backends:backend_lib",
            "//executorch/backends/test/...",
            "//executorch/backends/xnnpack/test/...",
            "//executorch/extension/pybindings/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//xplat/third-party/XNNPACK:XNNPACK",
            ":xnnpack_schema",
            "//executorch/runtime/backend:backend_registry",
            "//executorch/backends/qnnpack:qnnpack_utils",  # TODO Use (1) portable for choose_qparams(), (2) xnnpack for quantize_per_tensor()
            "//executorch/extension/fb/threadpool:threadpool",
            "//executorch/util:memory_utils",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
        # XnnpackBackend.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
    )
