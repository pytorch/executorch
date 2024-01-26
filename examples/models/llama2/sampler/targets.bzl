load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    for aten in (True, False):
        aten_postfix = "_aten" if aten else ""
        runtime.cxx_library(
            name = "sampler" + aten_postfix,
            exported_headers = [
                "sampler.h",
            ],
            preprocessor_flags = [
                "-DUSE_ATEN_LIB",
            ] if aten else [],
            srcs = [
                "sampler.cpp",
            ],
            visibility = [
                "//executorch/...",
            ],
            external_deps = [
                "libtorch",
            ] if aten else [],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib",
            ],
        )
