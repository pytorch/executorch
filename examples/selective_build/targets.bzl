load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "is_xplat", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Select all ops: register all the ops in portable/functions.yaml
    et_operator_library(
        name = "select_all_ops",
        include_all_operators = True,
    )

    executorch_generated_lib(
        name = "select_all_lib",
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        deps = [
            ":select_all_ops",
        ],
    )

    # Select a list of operators: defined in `ops`
    et_operator_library(
        name = "select_ops_in_list",
        ops = [
            "aten::add.out",
            "aten::mm.out",
        ],
    )

    executorch_generated_lib(
        name = "select_ops_in_list_lib",
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        deps = [
            ":select_ops_in_list",
        ],
    )

    # Select a dictionary of ops with kernel metadata
    et_operator_library(
        name = "select_ops_in_dict",
        ops_dict = {
            "aten::add.out": ["v1/3;0,1", "v1/6;0,1"],  # int, float
            "aten::mm.out": [],  # all dtypes
        },
    )

    executorch_generated_lib(
        name = "select_ops_in_dict_lib",
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        deps = [
            ":select_ops_in_dict",
        ],
        dtype_selective_build = True,
        visibility = ["//executorch/..."],
    )

    # Select all ops from a yaml file
    et_operator_library(
        name = "select_ops_from_yaml",
        ops_schema_yaml_target = "//executorch/examples/portable/custom_ops:custom_ops.yaml",
    )

    executorch_generated_lib(
        name = "select_ops_from_yaml_lib",
        custom_ops_yaml_target = "//executorch/examples/portable/custom_ops:custom_ops.yaml",
        kernel_deps = [
            "//executorch/examples/portable/custom_ops:custom_ops_1",
            "//executorch/examples/portable/custom_ops:custom_ops_2",
        ],
        deps = [
            ":select_ops_from_yaml",
        ],
    )

    # Select all ops from a given model
    # TODO(larryliu0820): Add this

    if not runtime.is_oss and not is_xplat():
        runtime.genrule(
            name = "add_mul_model",
            outs = {"add_mul": ["add_mul.pte"]},
            cmd = "$(exe fbcode//executorch/examples/portable/scripts:export) --model_name add_mul --output_dir $OUT",
            macros_only = False,
            visibility = ["//executorch/..."],
        )

        et_operator_library(
            name = "select_ops_from_model",
            model = ":add_mul_model[add_mul]",
        )

        executorch_generated_lib(
            name = "select_ops_from_model_lib",
            functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
            kernel_deps = ["//executorch/kernels/portable:operators"],
            deps = [":select_ops_from_model"],
            visibility = ["//executorch/kernels/..."],
        )

    # ~~~ Test binary for selective build ~~~
    select_ops = native.read_config("executorch", "select_ops", None)
    lib = []
    if select_ops == "all":
        lib.append(":select_all_lib")
    elif select_ops == "list":
        lib.append(":select_ops_in_list_lib")
    elif select_ops == "dict":
        lib.append(":select_ops_in_dict_lib")
    elif select_ops == "yaml":
        lib.append(":select_ops_from_yaml_lib")
    elif select_ops == "model":
        lib.append(":select_ops_from_model_lib")
    runtime.cxx_binary(
        name = "selective_build_test",
        srcs = [],
        deps = [
            "//executorch/examples/portable/executor_runner:executor_runner_lib",
        ] + lib,
        define_static_target = True,
        **get_oss_build_kwargs()
    )
