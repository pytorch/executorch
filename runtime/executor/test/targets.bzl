load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")
        runtime.cxx_library(
            name = "test_backend_compiler_lib" + aten_suffix,
            srcs = [
                "test_backend_compiler_lib.cpp",
            ],
            visibility = [
                "//executorch/exir/backend/test/...",
                "//executorch/runtime/backend/...",
                "//executorch/extension/pybindings/...",
                "//executorch/sdk/runners/...",
                "//executorch/test/...",
                "//executorch/examples/...",
            ],
            # registration of backends is done through a static global
            compiler_flags = ["-Wno-global-constructors"],
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            exported_deps = [
                "//executorch/runtime/backend:backend_registry" + aten_suffix,
            ],
            # TestBackendCompilerLib.cpp needs to compile with executor as whole
            # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
            link_whole = True,
        )

        runtime.cxx_library(
            name = "test_backend_with_delegate_mapping" + aten_suffix,
            srcs = [
                "test_backend_with_delegate_mapping.cpp",
            ],
            visibility = [
                "//executorch/exir/backend/test/...",
                "//executorch/runtime/backend/...",
                "//executorch/extension/pybindings/...",
                "//executorch/sdk/runners/...",
                "//executorch/test/...",
                "//executorch/examples/...",
            ],
            # registration of backends is done through a static global
            compiler_flags = ["-Wno-global-constructors"],
            preprocessor_flags = ["-DUSE_ATEN_LIB"] if aten_mode else [],
            exported_deps = [
                "//executorch/runtime/backend:backend_registry" + aten_suffix,
            ],
            # TestBackendCompilerLib.cpp needs to compile with executor as whole
            # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
            link_whole = True,
        )

    runtime.cxx_test(
        name = "executor_test",
        srcs = [
            "executor_test.cpp",
        ],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core:evalue",
            "//executorch/runtime/core:core",
            "//executorch/runtime/platform:platform",
            "//executorch/runtime/kernel:operator_registry",
            "//executorch/runtime/executor:executor",
            "//executorch/kernels/portable:generated_lib",
            "//executorch/runtime/kernel:kernel_runtime_context",
            "//executorch/extension/pytree:pytree",
            "//executorch/test/utils:utils",
            "//executorch/util:test_memory_config",
        ],
    )

    runtime.cxx_library(
        name = "managed_memory_manager",
        srcs = [],
        exported_headers = [
            "managed_memory_manager.h",
        ],
        visibility = [
            "//executorch/runtime/executor/test/...",
            "//executorch/test/...",
            "@EXECUTORCH_CLIENTS",
        ],
        deps = [
            "//executorch/runtime/core:memory_allocator",
            "//executorch/runtime/executor:memory_manager",
        ],
    )

    # TODO(dbort): Find a way to make these run for ANDROID/APPLE in xplat. The
    # android and ios test determinators don't like the reference to the model
    # file in fbcode. See https://fburl.com/9esapdmd
    if not runtime.is_oss and is_fbcode:
        modules_env = {
            # The tests use this var to find the program file to load. This uses
            # an fbcode target path because the authoring/export tools
            # intentionally don't work in xplat (since they're host-only tools).
            "ET_MODULE_ADD_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleAdd.pte])",
            "ET_MODULE_INDEX_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleIndex.pte])",
            "ET_MODULE_MULTI_ENTRY_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleMultipleEntry.pte])",
        }

        runtime.cxx_test(
            name = "allocation_failure_stress_test",
            srcs = [
                "allocation_failure_stress_test.cpp",
            ],
            deps = [
                ":managed_memory_manager",
                "//executorch/runtime/executor:program",
                "//executorch/kernels/portable:generated_lib",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/util:util",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "execution_plan_test",
            srcs = [
                "execution_plan_test.cpp",
            ],
            deps = [
                ":managed_memory_manager",
                "//executorch/runtime/executor:executor",
                "//executorch/util:util",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/kernels/portable:generated_lib",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "method_test",
            srcs = [
                "method_test.cpp",
            ],
            deps = [
                ":managed_memory_manager",
                "//executorch/runtime/executor:program",
                "//executorch/util:util",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/kernels/portable:generated_lib",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "method_meta_test",
            srcs = [
                "method_meta_test.cpp",
            ],
            deps = [
                "//executorch/runtime/executor:program",
                "//executorch/util:util",
                "//executorch/extension/data_loader:file_data_loader",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "program_test",
            srcs = [
                "program_test.cpp",
            ],
            deps = [
                "//executorch/runtime/executor:program",
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/extension/data_loader:file_data_loader",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "kernel_resolution_test",
            srcs = [
                "kernel_resolution_test.cpp",
            ],
            deps = [
                ":managed_memory_manager",
                "//executorch/runtime/executor:executor",
                "//executorch/runtime/kernel:operator_registry",
                "//executorch/util:util",
                "//executorch/extension/data_loader:file_data_loader",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "kernel_integration_test",
            srcs = [
                "kernel_integration_test.cpp",
            ],
            deps = [
                ":managed_memory_manager",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/runtime/core:core",
                "//executorch/runtime/executor:program",
                "//executorch/runtime/kernel:kernel_runtime_context",
                "//executorch/runtime/kernel:operator_registry",
                "//executorch/runtime/platform:platform",
                "//executorch/util:util",
            ],
            env = modules_env,
        )

        runtime.cxx_test(
            name = "backend_integration_test",
            srcs = [
                "backend_integration_test.cpp",
            ],
            deps = [
                ":managed_memory_manager",
                "//executorch/runtime/backend:backend_registry",
                "//executorch/runtime/executor:program",
                "//executorch/extension/data_loader:buffer_data_loader",
                "//executorch/extension/data_loader:file_data_loader",
                "//executorch/util:util",
            ],
            env = {
                # The tests use these vars to find the program files to load.
                # Uses an fbcode target path because the authoring/export tools
                # intentionally don't work in xplat (since they're host-only
                # tools).
                "ET_MODULE_ADD_MUL_NOSEGMENTS_DA1024_PATH": "$(location fbcode//executorch/test/models:exported_delegated_programs[ModuleAddMul-nosegments-da1024.pte])",
                "ET_MODULE_ADD_MUL_NOSEGMENTS_PATH": "$(location fbcode//executorch/test/models:exported_delegated_programs[ModuleAddMul-nosegments.pte])",
                "ET_MODULE_ADD_MUL_PATH": "$(location fbcode//executorch/test/models:exported_delegated_programs[ModuleAddMul.pte])",
            },
        )
