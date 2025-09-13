load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    runtime.cxx_test(
        name = "dynamic_quant_utils_test",
        srcs = ["runtime/test_runtime_utils.cpp"],
        fbcode_deps = [
            "//caffe2:ATen-cpu",
        ],
        xplat_deps = [
            "//caffe2:aten_cpu",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/extension/aten_util:aten_bridge",
            "//executorch/backends/xnnpack:dynamic_quant_utils",
        ],
    )

    runtime.cxx_test(
        name = "xnnexecutor_test",
        srcs = ["runtime/test_xnnexecutor.cpp"],
        deps = [
            third_party_dep("XNNPACK"),
            third_party_dep("FP16"),
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/core/exec_aten/util:scalar_type_util",
            "//executorch/backends/xnnpack:xnnpack_backend",
        ],
    )

    runtime.cxx_test(
        name = "test_xnn_weights_cache",
        srcs = ["runtime/test_xnn_weights_cache.cpp"],
        deps = [
            third_party_dep("XNNPACK"),
            "//executorch/backends/xnnpack:xnnpack_backend",
            "//executorch/runtime/executor:pte_data_map",
            "//executorch/extension/data_loader:file_data_loader",
            "//executorch/extension/testing_util:temp_file",
            "//executorch/schema:program",
        ],
    )

    runtime.cxx_test(
        name = "test_xnn_data_separation",
        srcs = ["runtime/test_xnn_data_separation.cpp"],
        deps = [
                "//executorch/runtime/executor/test:managed_memory_manager",
                "//executorch/runtime/executor:program",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/backends/xnnpack:xnnpack_backend",
                "//executorch/extension/flat_tensor:flat_tensor_data_map",
            ],
            env = {
                # The tests use these vars to find the program files to load.
                # Uses an fbcode target path because the authoring/export tools
                # intentionally don't work in xplat (since they're host-only
                # tools).
                "ET_MODULE_LINEAR_XNN_PROGRAM_PATH": "$(location fbcode//executorch/test/models:exported_xnnpack_program_and_data[ModuleLinear-e.pte])",
                "ET_MODULE_LINEAR_XNN_DATA_PATH": "$(location fbcode//executorch/test/models:exported_xnnpack_program_and_data[ModuleLinear.ptd])",
            },
    )

    runtime.cxx_test(
        name = "test_workspace_sharing",
        srcs = ["runtime/test_workspace_sharing.cpp"],
        deps = [
                "//executorch/extension/module:module",
                "//executorch/extension/tensor:tensor",
                "//executorch/backends/xnnpack:xnnpack_backend",
            ],
            env = {
                "ET_XNNPACK_GENERATED_ADD_LARGE_PTE_PATH": "$(location fbcode//executorch/test/models:exported_xnnp_delegated_programs[ModuleAddLarge.pte])",
                "ET_XNNPACK_GENERATED_SUB_LARGE_PTE_PATH": "$(location fbcode//executorch/test/models:exported_xnnp_delegated_programs[ModuleSubLarge.pte])",
            },
    )

    runtime.cxx_test(
        name = "test_workspace_manager",
        srcs = ["runtime/test_workspace_manager.cpp"],
        deps = [
                third_party_dep("XNNPACK"),
                "//executorch/backends/xnnpack:xnnpack_backend",
            ],
    )
