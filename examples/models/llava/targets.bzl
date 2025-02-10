load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    runtime.cxx_binary(
        name = "main",
        srcs = [
            "main.cpp",
        ],
        compiler_flags = ["-Wno-global-constructors"],
        preprocessor_flags = [
            "-DET_USE_THREADPOOL",
        ],
        deps = [
            "//executorch/examples/models/llava/runner:runner",
            "//executorch/extension/evalue_util:print_evalue",
            "//executorch/extension/threadpool:cpuinfo_utils",
            "//executorch/extension/threadpool:threadpool",
        ],
        external_deps = [
            "gflags",
            "torch-core-cpp",
        ],
        **get_oss_build_kwargs()
    )
