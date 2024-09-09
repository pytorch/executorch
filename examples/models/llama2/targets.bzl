load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "runtime")

def define_common_targets():
    for aten in (True, False):
        if not runtime.is_oss or not aten:
            aten_suffix = "_aten" if aten else ""

            runtime.cxx_binary(
                name = "main" + aten_suffix,
                srcs = [
                    "main.cpp",
                ],
                compiler_flags = ["-Wno-global-constructors"],
                preprocessor_flags = [
                    "-DUSE_ATEN_LIB",
                ] if aten else [],
                deps = [
                    "//executorch/examples/models/llama2/runner:runner" + aten_suffix,
                    "//executorch/extension/evalue_util:print_evalue",
                    "//executorch/extension/threadpool:threadpool",
                    "//executorch/extension/threadpool:cpuinfo_utils",
                ],
                external_deps = [
                    "gflags",
                ],
                **get_oss_build_kwargs()
            )
