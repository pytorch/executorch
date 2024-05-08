load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.

    See README.md for instructions on selective build.
    """
    runtime.python_library(
        name = "gen_oplist_lib",
        srcs = ["gen_oplist.py"],
        base_module = "executorch.codegen.tools",
        visibility = [
            "//executorch/...",
        ],
        external_deps = ["torchgen"],
        deps = select({
            "DEFAULT": [],
            "ovr_config//os:linux": [] if runtime.is_oss else ["//executorch/codegen/tools/fb:selective_build"],  # TODO(larryliu0820) :selective_build doesn't build in OSS yet
        }),
    )

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
        external_deps = ["torchgen"],
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

    runtime.python_library(
        name = "gen_oplist_copy_from_core",
        srcs = [
            "gen_oplist_copy_from_core.py",
        ],
        base_module = "tools_copy.code_analyzer",
        external_deps = ["torchgen"],
    )

    runtime.python_library(
        name = "gen_all_oplist_lib",
        srcs = ["gen_all_oplist.py"],
        base_module = "executorch.codegen.tools",
        visibility = [
            "//executorch/...",
        ],
        deps = [":gen_oplist_copy_from_core"],
    )

    runtime.python_binary(
        name = "gen_all_oplist",
        main_module = "executorch.codegen.tools.gen_all_oplist",
        package_style = "inplace",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":gen_all_oplist_lib",
        ],
        _is_external_target = True,
    )

    runtime.python_test(
        name = "test_gen_all_oplist",
        srcs = [
            "test/test_gen_all_oplist.py",
        ],
        package_style = "inplace",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":gen_all_oplist_lib",
        ],
        _is_external_target = True,
    )

    runtime.python_library(
        name = "gen_selected_op_variants_lib",
        srcs = ["gen_selected_op_variants.py"],
        base_module = "executorch.codegen.tools",
        visibility = ["//executorch/..."],
        deps = [":gen_oplist_copy_from_core"],
    )

    runtime.python_binary(
        name = "gen_selected_op_variants",
        main_module = "executorch.codegen.tools.gen_selected_op_variants",
        package_style = "inplace",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":gen_selected_op_variants_lib",
        ],
        _is_external_target = True,
    )

    runtime.python_test(
        name = "test_gen_selected_op_variants",
        srcs = [
            "test/test_gen_selected_op_variants.py",
        ],
        package_style = "inplace",
        visibility = [
            "PUBLIC",
        ],
        deps = [
            ":gen_selected_op_variants_lib",
            "fbsource//third-party/pypi/expecttest:expecttest",
        ],
        _is_external_target = True,
    )

    # TODO(larryliu0820): This is a hack to only run these two on fbcode. These targets depends on exir which is only available in fbcode.
    if not runtime.is_oss and is_fbcode:
        runtime.python_binary(
            name = "gen_functions_yaml",
            srcs = ["gen_ops_def.py"],
            main_module = "executorch.codegen.tools.gen_ops_def",
            package_style = "inplace",
            visibility = [
                "//executorch/...",
                "@EXECUTORCH_CLIENTS",
            ],
            deps = [
                "fbsource//third-party/pypi/pyyaml:pyyaml",
                ":yaml_util",
                "//caffe2:torch",
                "//executorch/exir:schema",
                "//executorch/exir/_serialize:lib",
            ],
        )

        runtime.python_test(
            name = "test_gen_oplist_real_model",
            srcs = ["test/test_gen_oplist_real_model.py"],
            base_module = "",
            resources = {
                "//executorch/test/models:exported_programs[ModuleLinear.pte]": "test/ModuleLinear.pte",
            },
            visibility = [
                "//executorch/...",
            ],
            deps = [
                ":gen_oplist_lib",
                "//libfb/py:parutil",
            ],
        )
