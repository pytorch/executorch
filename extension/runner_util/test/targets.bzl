load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    for aten_mode in (True, False):
        aten_suffix = ("_aten" if aten_mode else "")

        # TODO(dbort): Find a way to make these run for ANDROID/APPLE in xplat. The
        # android and ios test determinators don't like the reference to the model
        # file in fbcode. See https://fburl.com/9esapdmd
        if not runtime.is_oss and is_fbcode:
            runtime.cxx_test(
                name = "inputs_test" + aten_suffix,
                srcs = [
                    "inputs_test.cpp",
                ],
                deps = [
                    "//executorch/extension/runner_util:inputs",
                    "//executorch/runtime/executor/test:managed_memory_manager",
                    "//executorch/runtime/executor:program",
                    "//executorch/kernels/portable:generated_lib",
                    "//executorch/extension/data_loader:file_data_loader",
                ],
                env = {
                    "ET_MODULE_ADD_PATH": "$(location fbcode//executorch/test/models:exported_programs[ModuleAdd.pte])",
                },
            )
