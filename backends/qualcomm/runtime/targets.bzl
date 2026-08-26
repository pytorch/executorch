load(
    "@fbsource//tools/build_defs:default_platform_defs.bzl",
    "ANDROID",
    "CXX",
)
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/qualcomm/third-party:third_party_libs.bzl", "qnn_third_party_dep")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    runtime.cxx_library(
        name = "logging",
        srcs = [
            "Logging.cpp",
        ],
        exported_headers = [
            "Logging.h",
        ],
        define_static_target = True,
        platforms = [ANDROID, CXX],
        visibility = ["PUBLIC"],
        deps = [
            qnn_third_party_dep("api"),
            qnn_third_party_dep("app_sources"),
            "//executorch/runtime/backend:interface",
        ],
        exported_deps = [
            qnn_third_party_dep("log"),
            "//executorch/backends/qualcomm:schema",
            "//executorch/runtime/core:core",
        ],
    )

    # Platform Abstraction Layer. The headers are included as <pal/...> (matching
    # the CMake build's `include_directories(runtime/pal/include)`). They are
    # exposed through a header map (dict `exported_headers` with an empty
    # namespace) instead of an `-I` flag, so the short <pal/...> include resolves
    # identically under both the fbcode (`cpp_library`) and xplat
    # (`fb_xplat_cxx_library`) rules, which do not share an include-dir attribute.
    # Kept in their own library so the mapping does not disturb the runtime
    # target's namespaced <executorch/...> exported headers.
    runtime.cxx_library(
        name = "pal",
        srcs = glob([
            "pal/src/linux/*.cpp",
        ]),
        exported_headers = {
            "pal/DynamicLoading.h": "pal/include/pal/DynamicLoading.h",
            "pal/Path.h": "pal/include/pal/Path.h",
        },
        header_namespace = "",
        define_static_target = True,
        # Match `:logging` and `:runtime` (both [ANDROID, CXX]) -- `:pal` is an
        # exported dep of `:runtime`, so its host (CXX) variant must exist for the
        # `:runtime` CXX build to resolve on Linux (OSS `//backends/qualcomm/...`
        # buck build, and the internal x86 simulator runner). Sources are
        # `pal/src/linux/*.cpp`, which build fine on any Linux host.
        platforms = [ANDROID, CXX],
        visibility = ["PUBLIC"],
    )

    # "runtime" target is used for offline compile, can be renamed to runtime_aot_build as a BE.
    for include_aot_qnn_lib in (True, False):
        qnn_build_suffix = ("" if include_aot_qnn_lib else "_android_build")
        runtime.cxx_library(
            name = "runtime" + qnn_build_suffix,
            srcs = glob(
                [
                    "*.cpp",
                    "backends/*.cpp",
                    "backends/gpu/*.cpp",
                    "backends/htp/*.cpp",
                    "backends/ir/*.cpp",
                    "backends/lpai/*.cpp",
                ] + (["backends/gpu/host/*.cpp"] if include_aot_qnn_lib else ["backends/gpu/target/*.cpp"]) + (
                    ["backends/htp/host/*.cpp"] if include_aot_qnn_lib else ["backends/htp/target/*.cpp"]) + (
                    ["backends/ir/host/*.cpp"] if include_aot_qnn_lib else ["backends/ir/target/*.cpp"]) + (
                    ["backends/lpai/host/*.cpp"] if include_aot_qnn_lib else ["backends/lpai/target/*.cpp"]
                ),
                exclude = ["Logging.cpp"],
            ),
            exported_headers = glob(
                [
                    "*.h",
                    "backends/*.h",
                    "backends/gpu/*.h",
                    "backends/htp/*.h",
                    "backends/ir/*.h",
                    "backends/lpai/*.h",
                ],
                exclude = ["Logging.h"],
            ),
            define_static_target = True,
            link_whole = True,  # needed for executorch/examples/models/llama:main to register QnnBackend
            platforms = [ANDROID, CXX],
            visibility = ["PUBLIC"],
            resources = ({
                "qnn_lib": qnn_third_party_dep("qnn_offline_compile_libs"),
                } if include_aot_qnn_lib else {
            }),
            deps = [
                qnn_third_party_dep("api"),
                qnn_third_party_dep("app_sources"),
                ":logging",
                "//executorch/backends/qualcomm:schema",
                "//executorch/backends/qualcomm/aot/wrappers:wrappers",
                "//executorch/runtime/core:core",
                "//executorch/extension/tensor:tensor",
            ],
            exported_deps = [
                ":pal",
                "//executorch/runtime/backend:interface",
                "//executorch/runtime/core/exec_aten/util:scalar_type_util",
                "//executorch/runtime/core:event_tracer",
            ],
        )
