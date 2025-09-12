load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode=False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_test(
        name = "flat_tensor_header_test",
        srcs = [
            "flat_tensor_header_test.cpp",
        ],
        deps = [
            "//executorch/extension/flat_tensor/serialize:flat_tensor_header",
        ],
    )

    runtime.cxx_test(
        name = "serialize_cpp_test",
        srcs = [
            "test_serialize.cpp",
        ],
        deps = [
            "//executorch/extension/flat_tensor/serialize:serialize_cpp",
            "//executorch/extension/flat_tensor/serialize:generated_headers",
            "//executorch/extension/flat_tensor/serialize:flat_tensor_header",
            "//executorch/extension/tensor:tensor",
        ],
    )

    if not runtime.is_oss and is_fbcode:
        modules_env = {
            # The tests use this var to find the program file to load. This uses
            # an fbcode target path because the authoring/export tools
            # intentionally don't work in xplat (since they're host-only tools).
            "ET_MODULE_ADD_MUL_PROGRAM_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleAddMul.pte])",
            "ET_MODULE_ADD_MUL_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleAddMul.ptd])",
        }

        runtime.cxx_test(
            name = "flat_tensor_data_map_test",
            srcs = [
                "flat_tensor_data_map_test.cpp",
            ],
            deps = [
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/extension/flat_tensor:flat_tensor_data_map",
                "//executorch/runtime/core:named_data_map",
                "//executorch/runtime/core/exec_aten:lib",
            ],
            env = modules_env,
        )
