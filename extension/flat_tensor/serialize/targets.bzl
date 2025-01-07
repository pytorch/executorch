load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.genrule(
        name = "gen_schema",
        srcs = [
            "flat_tensor.fbs",
            "scalar_type.fbs",
        ],
        outs = {
            "schema_generated.h": ["flat_tensor_generated.h"],
            "scalar_type_generated.h": ["scalar_type_generated.h"]
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
        name = "generated_headers",
        srcs = [],
        visibility = [
            "//executorch/...",
        ],
        exported_headers = {
            "schema_generated.h": ":gen_schema[schema_generated.h]",
            "scalar_type_generated.h": ":gen_schema[scalar_type_generated.h]",
        },
        exported_external_deps = ["flatbuffers-api"],
    )
