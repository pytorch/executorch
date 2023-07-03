load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.

    See README.md for instructions on selective build.
    """
    runtime.python_binary(
        name = "gen_oplist",
        main_module = "executorch.codegen.tools.gen_oplist",
        deps = [
            ":gen_oplist_lib",
        ],
        package_style = "inplace",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    runtime.python_library(
        name = "yaml_util",
        base_module = "executorch.codegen.tools",
        srcs = ["yaml_util.py"],
    )

    runtime.python_library(
        name = "merge_yaml_lib",
        srcs = ["merge_yaml.py"],
        base_module = "executorch.codegen.tools",
        deps = [
            ":yaml_util",
        ],
        fbcode_deps = [
            "//caffe2/torchgen:torchgen",
        ],
        xplat_deps = [
            "//xplat/caffe2/torchgen:torchgen",
        ],
    )

    runtime.python_binary(
        name = "merge_yaml",
        main_module = "executorch.codegen.tools.merge_yaml",
        deps = [
            ":merge_yaml_lib",
        ],
        package_style = "inplace",
        _is_external_target = True,
        visibility = ["PUBLIC"],
    )

    runtime.python_test(
        name = "test_gen_oplist",
        base_module = "",
        srcs = [
            "test/test_gen_oplist.py",
        ],
        deps = [
            ":gen_oplist_lib",
        ],
        package_style = "inplace",
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )
