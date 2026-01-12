load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.genrule(
        name = "gen_schema",
        srcs = [
            "flat_tensor.fbs",
            "scalar_type.fbs",
        ],
        outs = {
            "flat_tensor_generated.h": ["flat_tensor_generated.h"],
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
            "flat_tensor_generated.h": ":gen_schema[flat_tensor_generated.h]",
            "scalar_type_generated.h": ":gen_schema[scalar_type_generated.h]",
        },
        exported_external_deps = ["flatbuffers-api"],
    )

    runtime.cxx_library(
        name = "flat_tensor_header",
        srcs = ["flat_tensor_header.cpp"],
        exported_headers = ["flat_tensor_header.h"],
        visibility = [
            "//executorch/...",
        ],
        exported_deps = ["//executorch/runtime/core:core"],
    )

    runtime.cxx_library(
        name = "serialize_cpp",
        srcs = ["serialize.cpp"],
        deps = [
            ":flat_tensor_header",
            ":generated_headers",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        exported_headers = ["serialize.h"],
        visibility = ["PUBLIC"],
        exported_external_deps = ["flatbuffers-api"],
    )
