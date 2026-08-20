load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")
load("@fbsource//xplat/executorch/codegen:codegen.bzl", "et_operator_library", "executorch_generated_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "operators",
        srcs = [],
        visibility = ["PUBLIC"],
        exported_deps = [
            "//executorch/kernels/portable/cpu:cpu",
        ],
    )

    if True in get_aten_mode_options():
        runtime.cxx_library(
            name = "operators_aten",
            srcs = [],
            visibility = ["PUBLIC"],
            exported_deps = [
                "//executorch/kernels/portable/cpu:cpu_aten",
            ],
        )

    runtime.export_file(
        name = "functions.yaml",
        visibility = ["PUBLIC"],
    )

    runtime.export_file(
        name = "custom_ops.yaml",
        visibility = ["PUBLIC"],
    )

    runtime.export_file(
        name = "et_copy_ops.yaml",
        visibility = ["PUBLIC"],
    )

    et_operator_library(
        name = "executorch_all_ops",
        include_all_operators = True,
        define_static_targets = True,
        visibility = ["PUBLIC"],
    )

    et_operator_library(
        name = "executorch_aten_ops",
        ops_schema_yaml_target = "//executorch/kernels/portable:functions.yaml",
        define_static_targets = True,
        visibility = ["PUBLIC"],
    )

    et_operator_library(
        name = "executorch_custom_ops",
        ops_schema_yaml_target = "//executorch/kernels/portable:custom_ops.yaml",
        define_static_targets = True,
        visibility = ["PUBLIC"],
    )

    generated_lib_common_args = {
        "custom_ops_yaml_target": "//executorch/kernels/portable:custom_ops.yaml",
        # size_test expects _static targets to be available for these libraries.
        "define_static_targets": True,
        "functions_yaml_target": "//executorch/kernels/portable:functions.yaml",
        "visibility": ["PUBLIC"],
    }

    for support_exceptions in [True, False]:
        exception_suffix = "_no_exceptions" if not support_exceptions else ""

        executorch_generated_lib(
            name = "generated_lib" + exception_suffix,
            deps = [
                ":executorch_aten_ops",
                ":executorch_custom_ops",
            ],
            kernel_deps = ["//executorch/kernels/portable:operators"],
            support_exceptions = support_exceptions,
            **generated_lib_common_args
        )

    if True in get_aten_mode_options():
        executorch_generated_lib(
            name = "generated_lib_aten",
            deps = [
                ":executorch_aten_ops",
                ":executorch_custom_ops",
                "//executorch/kernels/portable:operators_aten",
            ],
            custom_ops_aten_kernel_deps = [
                "//executorch/kernels/portable:operators_aten",
            ],
            custom_ops_yaml_target = "//executorch/kernels/portable:custom_ops.yaml",
            aten_mode = True,
            visibility = ["PUBLIC"],
            define_static_targets = True,
        )

        et_operator_library(
            name = "et_copy_ops",
            ops_schema_yaml_target = "//executorch/kernels/portable:et_copy_ops.yaml",
            define_static_targets = True,
            visibility = ["PUBLIC"],
        )

        # ATen-mode registration for the device-copy ops
        # (`et_copy::_h2d_copy.out` / `_d2h_copy.out`) that the ExecuTorch
        # device-placement pass inserts at CPU<->accelerator boundaries (e.g.
        # around CUDA-delegated subgraphs). These ops live in `functions.yaml`,
        # so `generated_lib` already registers them for the portable runtime.
        # ATen-mode codegen, however, reads ATen's `native_functions.yaml` for
        # `functions_yaml_target` entries and cannot register non-ATen ops that
        # way, so `generated_lib_aten` does not cover them. This dedicated lib
        # registers them for ATen-mode runtimes via the custom-ops path, reusing
        # the same `op__device_copy` kernel (its `_aten` variant). `et_copy` is
        # intentionally in its own `et_copy_ops.yaml` (not `custom_ops.yaml`) so
        # it is not double-registered in the portable `generated_lib`.
        #
        # Note: because `op__device_copy` is in the shared CUSTOM_OPS list, its
        # `_aten` kernel is also compiled into `operators_aten` and thus links
        # (as unregistered code) into the general `generated_lib_aten`. This lib
        # is what actually *registers* the ops for ATen-mode consumers; it does
        # not isolate the kernel object from other ATen consumers.
        executorch_generated_lib(
            name = "device_copy_ops_aten_lib",
            deps = [
                ":et_copy_ops",
            ],
            kernel_deps = [
                "//executorch/kernels/portable/cpu:op__device_copy_aten",
            ],
            custom_ops_yaml_target = "//executorch/kernels/portable:et_copy_ops.yaml",
            custom_ops_requires_aot_registration = False,
            aten_mode = True,
            visibility = ["PUBLIC"],
            define_static_targets = True,
        )
