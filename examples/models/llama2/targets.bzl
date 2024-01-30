load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    for aten in (True, False):
        aten_suffix = "_aten" if aten else ""

        runtime.cxx_binary(
            name = "main" + aten_suffix,
            srcs = [
                "main.cpp",
            ],
            preprocessor_flags = [
                "-DUSE_ATEN_LIB",
            ] if aten else [],
            deps = [
                "//executorch/examples/models/llama2/runner:runner" + aten_suffix,
            ],
            external_deps = [
                "gflags",
            ],
        )
