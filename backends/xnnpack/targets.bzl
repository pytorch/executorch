load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:build_variables.bzl", "XNNPACK_BACKEND_BUCK_SRCS")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "get_aten_mode_options", "runtime")

def _get_preprocessor_flags():
    """
    Disable if someone explictly specified a config option,
    else Enable otherwise
    """
    preprocessor_flags = []
    if native.read_config("executorch", "xnnpack_workspace_sharing", "0") != "0":
        preprocessor_flags.append("-DENABLE_XNNPACK_SHARED_WORKSPACE")

    if native.read_config("executorch", "xnnpack_weights_cache", "0") != "0":
        preprocessor_flags.append("-DENABLE_XNNPACK_WEIGHTS_CACHE")

    # Enable if not disabled through config
    return preprocessor_flags

def define_common_targets():
    runtime.cxx_library(
        name = "dynamic_quant_utils",
        srcs = [
            "runtime/utils/utils.cpp",
        ],
        exported_headers = ["runtime/utils/utils.h"],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/backend:interface",
        ],
        visibility = [
            "//executorch/backends/xnnpack/...",
            "@EXECUTORCH_CLIENTS",
        ],
    )

    for aten_mode in get_aten_mode_options():
        aten_suffix = "_aten" if aten_mode else ""
        runtime.cxx_library(
            name = "xnnpack_backend" + aten_suffix,
            srcs = XNNPACK_BACKEND_BUCK_SRCS,
            headers = native.glob([
                "runtime/*.h",
                "runtime/profiling/*.h",
            ]),
            visibility = [
                "//executorch/exir/backend:backend_lib",
                "//executorch/exir/backend/test/...",
                "//executorch/backends/xnnpack/test/...",
                "//executorch/extension/pybindings/...",
                "@EXECUTORCH_CLIENTS",
            ],
            preprocessor_flags = [
                # Uncomment to enable per operator timings
                # "-DENABLE_XNNPACK_PROFILING",
                # Uncomment to enable using KleidiAI Kernels
                # "-DENABLE_XNNPACK_KLEIDI"
            ] + _get_preprocessor_flags(),
            exported_deps = [
                "//executorch/runtime/backend:interface" + aten_suffix,
            ],
            deps = [
                third_party_dep("XNNPACK"),
                "//executorch/backends/xnnpack/serialization:xnnpack_flatbuffer_header",
                "//executorch/extension/threadpool:threadpool",
                "//executorch/runtime/core/exec_aten/util:tensor_util" + aten_suffix,
                "//executorch/runtime/executor:pte_data_map" + aten_suffix,
            ],
            # XnnpackBackend.cpp needs to compile with executor as whole
            # @lint-ignore BUCKLINT: Avoid `link_whole=True` (https://fburl.com/avoid-link-whole)
            link_whole = True,
        )
