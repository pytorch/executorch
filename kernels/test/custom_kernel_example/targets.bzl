load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")
load("@fbsource//xplat/executorch/kernels/portable:op_registration_util.bzl", "define_op_target", "op_target")

MY_ATEN_COMPLIANT_OPS = (
    op_target(
        name = "op_relu",
        deps = [
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/runtime/core/exec_aten/util:tensor_util",
        ],
    ),
)

def define_common_targets():
    for op in MY_ATEN_COMPLIANT_OPS:
        define_op_target(is_aten_op = True, **op)

    all_op_targets = [":{}".format(op["name"]) for op in MY_ATEN_COMPLIANT_OPS]

    runtime.export_file(
        name = "my_functions.yaml",
        visibility = ["//executorch/..."],
    )

    runtime.cxx_library(
        name = "my_operators",
        srcs = [],
        visibility = [
            "//executorch/...",
            "@EXECUTORCH_CLIENTS",
        ],
        exported_deps = all_op_targets,
    )

    et_operator_library(
        name = "my_ops_list",
        _is_external_target = True,
        ops_schema_yaml_target = ":my_functions.yaml",
    )

    executorch_generated_lib(
        name = "generated_lib",
        deps = [
            ":my_ops_list",
            ":my_operators",
        ],
        functions_yaml_target = ":my_functions.yaml",
        define_static_targets = True,
    )
