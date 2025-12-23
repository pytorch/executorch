load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/qnn_version.bzl", "get_qnn_library_version")

PYTHON_MODULE_NAME = "PyQnnManagerAdaptor"

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.
    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    
    runtime.cxx_python_extension(
        name = "PyQnnManagerAdaptor",
        srcs = [
            "PyQnnManagerAdaptor.cpp",
        ],
        headers = [
            "PyQnnManagerAdaptor.h",
        ],
        base_module = "executorch.backends.qualcomm.python",
        preprocessor_flags = [
            "-DEXECUTORCH_PYTHON_MODULE_NAME={}".format(PYTHON_MODULE_NAME),
        ],
        deps = [
            "//executorch/runtime/core:core",
            "//executorch/backends/qualcomm/aot/python:python_lib",
            "//executorch/backends/qualcomm/aot/wrappers:wrappers",
            "//executorch/backends/qualcomm/runtime:logging",
            "//executorch/backends/qualcomm:schema",
            "//executorch/backends/qualcomm/runtime:runtime",
            "fbsource//third-party/pybind11:pybind11",
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:api".format(get_qnn_library_version()),
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:app_sources".format(get_qnn_library_version()),
        ],
        external_deps = [
            "libtorch_python",
        ],
        use_static_deps = True,
        visibility = [
            "//executorch/backends/qualcomm/...",
        ],
    )


    runtime.cxx_library(
        name = "python_lib",
        srcs = glob([
            "*.cpp",
        ]),
        exported_headers = glob([
            "*.h",
        ]),
        platforms = (CXX),
        visibility = ["@EXECUTORCH_CLIENTS"],
        deps = [
            "//executorch/backends/qualcomm/aot/wrappers:wrappers",
            "//executorch/backends/qualcomm/runtime:logging",
            "//executorch/backends/qualcomm:schema",
            "//executorch/backends/qualcomm/runtime:runtime",
            "fbsource//third-party/pybind11:pybind11",
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:api".format(get_qnn_library_version()),
            "fbsource//third-party/qualcomm/qnn/qnn-{0}:app_sources".format(get_qnn_library_version()),
        ],
    )
