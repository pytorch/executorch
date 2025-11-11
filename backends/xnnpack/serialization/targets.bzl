load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.genrule(
        name = "gen_xnnpack_schema",
        srcs = [
            "runtime_schema.fbs",
        ],
        # We're only generating a single file, so it seems like we could use
        # `out`, but `flatc` takes a directory as a parameter, not a single
        # file. Use `outs` so that `${OUT}` is expanded as the containing
        # directory instead of the file itself.
        outs = {
            "schema_generated.h": ["runtime_schema_generated.h"],
        },
        cmd = " ".join([
            "$(exe {})".format(runtime.external_dep_location("flatc")),
            "--cpp",
            "--cpp-std c++11",
            "--scoped-enums",
            "-o ${OUT}",
            "${SRCS}",
        ]),
        default_outs = ["."],
    )

    runtime.cxx_library(
        name = "xnnpack_flatbuffer_header",
        srcs = [],
        visibility = [
            "//executorch/backends/xnnpack/...",
        ],
        exported_headers = {
            "schema_generated.h": ":gen_xnnpack_schema[schema_generated.h]",
        },
        exported_external_deps = ["flatbuffers-api"],
    )
