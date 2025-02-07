load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    for aten in (True, False):
        aten_suffix = "_aten" if aten else ""

        runtime.cxx_library(
            name = "sampler" + aten_suffix,
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
                "@EXECUTORCH_CLIENTS",
            ],
            external_deps = [
                "libtorch",
            ] if aten else [],
            exported_deps = [
                "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
                "//executorch/runtime/platform:compiler",
            ],
        )
