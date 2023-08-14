load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "APPLE",
    "CXX",
)
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_xplat")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.genrule(
        name = "gen_qnnpack_schema",
        srcs = [
            "serialization/schema.fbs",
        ],
        # We're only generating a single file, so it seems like we could use
        # `out`, but `flatc` takes a directory as a parameter, not a single
        # file. Use `outs` so that `${OUT}` is expanded as the containing
        # directory instead of the file itself.
        outs = {
            "qnnpack_schema_generated.h": ["schema_generated.h"],
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
        name = "qnnpack_schema",
        srcs = [],
        exported_headers = {
            "qnnpack_schema_generated.h": ":gen_qnnpack_schema[qnnpack_schema_generated.h]",
        },
        exported_deps = [
            "fbsource//third-party/flatbuffers/fbsource_namespace:flatbuffers-api",
        ],
    )

    for aten_mode in (True, False):
        aten_suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "qnnpack_utils" + aten_suffix,
            srcs = [
                "utils/utils.cpp",
            ],
            exported_headers = ["utils/utils.h"],
            deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/backend:backend_registry",
            ],
            visibility = [
                "//executorch/backends/qnnpack/test/...",
                "//executorch/backends/xnnpack/...",
                "@EXECUTORCH_CLIENTS",
            ],
        )

    runtime.cxx_library(
        name = "qnnpack_backend",
        srcs = [
            "QNNPackBackend.cpp",
        ],
        headers = [
            "executor/QNNExecutor.h",
        ],
        resources = [
            "serialization/schema.fbs",
        ],
        visibility = [
            "//executorch/exir/backend:backend_lib",
            "//executorch/backends/qnnpack/test/...",
            "//executorch/exir/backend/test/...",
            "//executorch/extension/pybindings/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
            "//executorch/runtime/backend:backend_registry",
            "//executorch/backends/xnnpack/threadpool:threadpool",
            "//executorch/util:memory_utils",
            "//{prefix}caffe2/aten/src/ATen/native/quantized/cpu/qnnpack:pytorch_qnnpack".format(
                prefix = (
                    "xplat/" if is_xplat() else ""
                ),
            ),
            ":qnnpack_schema",
            ":qnnpack_utils",
        ],
        platforms = [
            ANDROID,
            APPLE,
            CXX,
        ],
        # XnnpackBackend.cpp needs to compile with executor as whole
        # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
        link_whole = True,
    )
