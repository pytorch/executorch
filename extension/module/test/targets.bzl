load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def define_common_targets(is_fbcode=False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    if not runtime.is_oss and is_fbcode:
        modules_env = {
            # The tests use this var to find the program file to load. This uses
            # an fbcode target path because the authoring/export tools
            # intentionally don't work in xplat (since they're host-only tools).
            "ET_MODULE_ADD_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleAdd.pte])",
            "ET_MODULE_ADD_MUL_PROGRAM_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleAddMul.pte])",
            "ET_MODULE_ADD_MUL_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleAddMul.ptd])",
            "ET_MODULE_LINEAR_PROGRAM_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleLinear.pte])",
            "ET_MODULE_LINEAR_DATA_PATH": "$(location fbcode//executorch/test/models:exported_program_and_data[ModuleLinear.ptd])",
            "ET_MODULE_SHARED_STATE": "$(location fbcode//executorch/test/models:exported_programs[ModuleSharedState.pte])",
        }

        for aten_mode in get_aten_mode_options():
            aten_suffix = ("_aten" if aten_mode else "")

            runtime.cxx_test(
                name = "test" + aten_suffix,
                srcs = [
                    "module_test.cpp",
                ],
                deps = [
                    "//executorch/kernels/portable:generated_lib" + aten_suffix,
                    "//executorch/extension/data_loader:file_data_loader",
                    "//executorch/extension/module:module" + aten_suffix,
                    "//executorch/extension/tensor:tensor" + aten_suffix,
                    "//executorch/runtime/core/exec_aten/testing_util:tensor_util" + aten_suffix,
                ],
                env = modules_env,
                platforms = [CXX, ANDROID],  # Cannot bundle resources on Apple platform.
                compiler_flags = [
                    "-Wno-error=deprecated-declarations",
                ],
            )

            runtime.cxx_test(
                name = "bundled_test" + aten_suffix,
                srcs = [
                    "bundled_module_test.cpp",
                ],
                deps = [
                    "//executorch/kernels/portable:generated_lib" + aten_suffix,
                    "//executorch/extension/module:bundled_module" + aten_suffix,
                    "//executorch/extension/tensor:tensor" + aten_suffix,
                ],
                env = {
                    "RESOURCES_PATH": "$(location :resources)/resources",
                    "ET_MODULE_PTE_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleAdd.pte])",
                },
                platforms = [CXX, ANDROID],  # Cannot bundle resources on Apple platform.
                compiler_flags = [
                    "-Wno-error=deprecated-declarations",
                ],
            )

    runtime.filegroup(
        name = "resources",
        srcs = native.glob([
            "resources/**",
        ]),
    )
