load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_oss_build_kwargs", "is_xplat", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib", "ScalarType")

def define_selective_build_prim_ops_example():
    """
    Example showing how selected_prim_operators_genrule works to combine
    prim ops headers from multiple dependencies.
    """

    # Define several operator libraries with automatic prim ops extraction
    et_operator_library(
        name = "model_a_ops",
        ops = [
            "aten::add.out",
            "aten::mul.out",
            "executorch_prim::et_view.default",    # Auto-extracted to prim ops
            "aten::sym_size.int",                  # Auto-extracted to prim ops
        ],
        visibility = ["//executorch/..."],
    )
    # This creates: "model_a_ops" + "model_a_ops_selected_prim_ops"

    et_operator_library(
        name = "model_b_ops",
        ops = [
            "aten::sub.out",
            "aten::div.out",
            "executorch_prim::add.Scalar",         # Auto-extracted to prim ops
            "aten::sym_numel.int",                 # Auto-extracted to prim ops
        ],
        visibility = ["//executorch/..."],
    )
    # This creates: "model_b_ops" + "model_b_ops_selected_prim_ops"

    # Define a manual prim ops target as well
    et_operator_library(
        name = "extra_prim_ops",
        ops = [
            "executorch_prim::mul.Scalar",
            "executorch_prim::sym_max.Scalar",
        ],
        visibility = ["//executorch/..."],
    )
    # Use the combined header in an executorch_generated_lib
    executorch_generated_lib(
        name = "library_with_combined_prim_ops",
        deps = [
            ":model_a_ops",
            ":model_b_ops",
            ":extra_prim_ops",
        ],
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        aten_mode = False,
        visibility = ["PUBLIC"],
        include_all_prim_ops = False,
    )

    # Prim ops selected separately
    et_operator_library(
        name = "model_b_ops_no_prim_ops",
        ops = [
            "aten::sub.out",
            "aten::div.out",
        ],
        visibility = ["//executorch/..."],
    )

    # Use the combined header in an executorch_generated_lib
    executorch_generated_lib(
        name = "library_with_combined_prim_ops_1",
        deps = [
            ":model_b_ops_no_prim_ops",
            ":extra_prim_ops",
        ],
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        aten_mode = False,
        visibility = ["PUBLIC"],
        include_all_prim_ops = False,
    )

    # No prim ops selected. So include all prim ops.
    executorch_generated_lib(
        name = "library_with_combined_prim_ops_2",
        deps = [
            ":model_b_ops_no_prim_ops",
        ],
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        aten_mode = False,
        visibility = ["PUBLIC"],
        include_all_prim_ops = False,
    )

    # default to selecting all prim ops
    executorch_generated_lib(
        name = "library_with_all_prim_ops",
        deps = [
            ":model_b_ops",
        ],
        kernel_deps = [
            "//executorch/kernels/portable:operators",
        ],
        functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
        aten_mode = False,
        visibility = ["PUBLIC"],
    )

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

    if runtime.is_oss or is_xplat():
        executorch_generated_lib(
            name = "select_all_dtype_selective_lib",
            functions_yaml_target = "//executorch/kernels/portable:functions.yaml",
            kernel_deps = [
                "//executorch/kernels/portable:operators",
            ],
            # Setting dtype_selective_build without using list or dict selection isn't a
            # typical use case; we just do it here so that we can test that our mechanism
            # for getting buck deps right for dtype_selective_build is working.
            dtype_selective_build = True,
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
            # 1. Use kernel key, generated with a model, or
            # 2. Specify the dtype, from executorch/codegen/codegen.bzl
            "aten::add.out": ["v1/3;0,1", ScalarType("Float")],  # int, float
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
        dtype_selective_build = is_xplat(),
        visibility = ["//executorch/..."],
    )

    executorch_generated_lib(
        name = "select_ops_in_dict_lib_optimized",
        functions_yaml_target = "//executorch/kernels/optimized:optimized.yaml",
        kernel_deps = [
            "//executorch/kernels/optimized:optimized_operators",
        ],
        deps = [
            ":select_ops_in_dict",
        ],
        dtype_selective_build = is_xplat(),
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
    elif select_ops == "dict_optimized":
        lib.append(":select_ops_in_dict_lib_optimized")
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

    define_selective_build_prim_ops_example()
