load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/qnn_version.bzl", "get_qnn_library_version")

# Dictionary mapping third-party library name to [internal_dep, oss_dep].
# Internal deps use fbsource//third-party/qualcomm/qnn/qnn-{version}:target.
# OSS deps use //backends/qualcomm/third-party:target (prebuilt from QNN_SDK_ROOT).
_QNN_THIRD_PARTY_LIBS = {
    "api": [
        "fbsource//third-party/qualcomm/qnn/qnn-{0}:api".format(get_qnn_library_version()),
        "//backends/qualcomm/third-party:qnn_api",
    ],
    "app_sources": [
        "fbsource//third-party/qualcomm/qnn/qnn-{0}:app_sources".format(get_qnn_library_version()),
        "//backends/qualcomm/third-party:qnn_app_sources",
    ],
    "qnn_offline_compile_libs": [
        "fbsource//third-party/qualcomm/qnn/qnn-{0}:qnn_offline_compile_libs".format(get_qnn_library_version()),
        "//backends/qualcomm/third-party:qnn_offline_compile_libs",
    ],
    "log": [
        "fbsource//third-party/toolchains:log",
        "//backends/qualcomm/third-party:log",
    ],
    "pybind11": [
        "fbsource//third-party/pybind11:pybind11",
        "//third-party:pybind11",
    ],
}

def qnn_third_party_dep(name):
    if name not in _QNN_THIRD_PARTY_LIBS:
        fail("Cannot find QNN third party library " + name + ", please register it in _QNN_THIRD_PARTY_LIBS first!")
    return _QNN_THIRD_PARTY_LIBS[name][1] if runtime.is_oss else _QNN_THIRD_PARTY_LIBS[name][0]
