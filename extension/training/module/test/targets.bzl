load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # TODO(dbort): Find a way to make these run for ANDROID/APPLE in xplat. The
    # android and ios test determinators don't like the reference to the model
    # file in fbcode. See https://fburl.com/9esapdmd
    if not runtime.is_oss and is_fbcode:
        modules_env = {
            # The tests use this var to find the program file to load. This uses
            # an fbcode target path because the authoring/export tools
            # intentionally don't work in xplat (since they're host-only tools).
            "ET_MODULE_ADD_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleAdd.pte])",
            "ET_MODULE_ADD_MUL_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleAddMul.ptd])",
            "ET_MODULE_ADD_MUL_PROGRAM_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleAddMul.pte])",
            "ET_MODULE_TRAIN_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleSimpleTrain.ptd])",
            "ET_MODULE_TRAIN_PROGRAM_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleSimpleTrainProgram.pte])",
            "ET_MODULE_SIMPLE_TRAIN_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleSimpleTrain.pte])",
        }

        runtime.cxx_test(
            name = "training_module_test",
            srcs = [
                "training_module_test.cpp",
            ],
            deps = [
                "//executorch/extension/training/module:training_module",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/flat_tensor:flat_tensor_data_map",
                "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
                "//executorch/kernels/portable:generated_lib",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "state_dict_util_test",
            srcs = [
                "state_dict_util_test.cpp",
            ],
            deps = [
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/flat_tensor:flat_tensor_data_map",
                "//executorch/extension/training/module:state_dict_util",
                "//executorch/runtime/core/exec_aten:lib",
            ],
            env = modules_env,
        )
