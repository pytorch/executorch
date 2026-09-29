load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if not is_fbcode:
        return

    runtime.python_library(
        name = "lib",
        srcs = [
            "gen.py",
            "utils.py",
        ],
        deps = [
            "fbsource//third-party/pypi/ruamel-yaml:ruamel-yaml",
            "//caffe2:torch",
            "//executorch/exir/dialects/edge/arg:lib",
            "//executorch/exir/dialects/edge/dtype:lib",
            "//executorch/exir/dialects/edge/op:lib",
        ],
    )

    python_binary(
        name = "gen",
        srcs = [],
        main_function = "executorch.exir.dialects.edge.spec.gen.main",
        deps = [
            "fbsource//third-party/pypi/expecttest:expecttest",  # @manual
            ":lib",
        ],
    )
