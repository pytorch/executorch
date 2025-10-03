load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode=False):
    if not runtime.is_oss and is_fbcode:
        modules_env = {
            # The tests use this var to find the program file to load. This uses
            # an fbcode target path because the authoring/export tools
            # intentionally don't work in xplat (since they're host-only tools).
            "ET_MODULE_ADD_MUL_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleAddMul.ptd])",
            "ET_MODULE_LINEAR_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleLinear.ptd])",
            "ET_MODULE_SIMPLE_TRAIN_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleSimpleTrain.ptd])",
        }

        runtime.cxx_test(
            name = "merged_data_map_test",
            srcs = [
                "merged_data_map_test.cpp",
            ],
            deps = [
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/flat_tensor:flat_tensor_data_map",
                "//executorch/extension/named_data_map:merged_data_map",
                "//executorch/runtime/core:named_data_map",
                "//executorch/runtime/core/exec_aten:lib",
            ],
            env = modules_env,
        )
