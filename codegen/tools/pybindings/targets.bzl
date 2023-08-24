load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    module_name = "selective_build"
    runtime.cxx_python_extension(
        name = module_name,
        srcs = [
            "selective_build.cpp",
        ],
        types = ["{}.pyi".format(module_name)],
        base_module = "executorch.codegen.tools.pybindings",
        preprocessor_flags = [
            "-DEXECUTORCH_PYTHON_MODULE_NAME={}".format(module_name),
        ],
        deps = [
            "//executorch/schema:program",
            "//executorch/util:read_file",
        ],
        external_deps = [
            "pybind11",
        ],
        use_static_deps = True,
        visibility = ["//executorch/codegen/..."],
    )
