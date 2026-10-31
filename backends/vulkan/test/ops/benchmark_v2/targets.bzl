load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load("@fbsource//xplat/executorch/backends/vulkan:targets.bzl", "get_platforms", "vulkan_spv_shader_lib")

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        return

    runtime.filegroup(
        name = "migration_shaders",
        srcs = native.glob(["operators/gemm/linear/qw/q4gsw/glsl/**/*"]),
        visibility = ["PUBLIC"],
    )

    vulkan_spv_shader_lib(
        name = "migration_shaderlib",
        spv_filegroups = {":migration_shaders": "operators"},
        is_fbcode = is_fbcode,
    )

    runtime.cxx_library(
        name = "config",
        srcs = [
            "framework/Config.cpp",
            "framework/DataInitializer.cpp",
            "framework/Sweep.cpp",
        ],
        exported_headers = [
            "framework/Config.h",
            "framework/DataInitializer.h",
            "framework/Sweep.h",
        ],
        platforms = get_platforms(),
        exported_external_deps = ["nlohmann_json"],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "throughput",
        srcs = [
            "framework/ThroughputRegistry.cpp",
            "operators/binary/add/AddThroughput.cpp",
            "operators/binary/q8ta/Q8taBinaryThroughput.cpp",
            "operators/clone/q8ta/Q8taCloneThroughput.cpp",
            "operators/gemm/linear/q8ta/Q8taLinearThroughput.cpp",
            "operators/pixel_shuffle/q8ta/Q8taPixelShuffleThroughput.cpp",
            "operators/quantize/q8ta/Q8taQdqThroughput.cpp",
            "operators/unary/q8ta/Q8taUnaryThroughput.cpp",
            "operators/attention/sdpa/SdpaThroughput.cpp",
            "operators/gemm/matmul/MatmulThroughput.cpp",
            "operators/quantization/choose_qparams_per_row/ChooseQparamsThroughput.cpp",
            "operators/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswThroughput.cpp",
            "operators/binary/BinaryThroughput.cpp",
            "operators/gemm/GemmThroughputCommon.cpp",
            "operators/gemm/linear/LinearThroughput.cpp",
            "operators/gemm/linear/qw/q4gsw/Q4gswLinearThroughput.cpp",
            "operators/convolution/Conv2dThroughputCommon.cpp",
            "operators/convolution/Conv2dThroughput.cpp",
            "operators/convolution/q8ta/Q8taConv2dThroughput.cpp",
        ],
        exported_headers = [
            "framework/ThroughputRegistry.h",
            "operators/convolution/Conv2dThroughputCommon.h",
            "operators/gemm/GemmThroughputCommon.h",
        ],
        platforms = get_platforms(),
        exported_deps = [":config"],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "throughput_registry_test",
        srcs = ["framework/test/throughput_registry_test.cpp"],
        platforms = get_platforms(),
        deps = [":throughput", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_test(
        name = "throughput_report_test",
        srcs = ["framework/test/throughput_report_test.cpp"],
        platforms = get_platforms(),
        deps = [":report", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_library(
        name = "report",
        srcs = ["framework/Report.cpp"],
        exported_headers = ["framework/Report.h"],
        platforms = get_platforms(),
        exported_deps = [":config", ":throughput"],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "report_test",
        srcs = ["framework/test/report_test.cpp"],
        platforms = get_platforms(),
        deps = [
            ":report",
            "//third-party/googletest:gtest_main",
        ],
    )

    runtime.cxx_library(
        name = "reference",
        srcs = [
            "framework/Reference.cpp",
            "framework/ReferenceRegistry.cpp",
            "operators/binary/add/AddReference.cpp",
            "operators/convolution/conv1d/Conv1dReference.cpp",
            "operators/gemm/matmul/MatmulReference.cpp",
            "operators/embedding/qw/q4gsw/EmbeddingReference.cpp",
            "operators/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswReference.cpp",
            "operators/attention/sdpa/SdpaReference.cpp",
            "operators/quantization/choose_qparams_per_row/ChooseQparamsReference.cpp",
            "operators/binary/q8ta/Q8taBinaryReference.cpp",
            "operators/clone/q8ta/Q8taCloneReference.cpp",
            "operators/convolution/q8ta/transposed/Q8taTransposedReference.cpp",
            "operators/convolution/qw/q8csw/Q8cswConv2dReference.cpp",
            "operators/gemm/linear/q8ta/Q8taLinearReference.cpp",
            "operators/gemm/linear/qw/q8csw/Q8cswLinearReference.cpp",
            "operators/pixel_shuffle/q8ta/Q8taPixelShuffleReference.cpp",
            "operators/quantize/q8ta/Q8taQdqReference.cpp",
            "operators/unary/q8ta/Q8taUnaryReference.cpp",
            "operators/binary/BinaryReference.cpp",
            "operators/convolution/Conv2dReferenceCommon.cpp",
            "operators/convolution/Conv2dReference.cpp",
            "operators/convolution/q8ta/Q8taConv2dReference.cpp",
            "operators/gemm/GemmReference.cpp",
            "operators/gemm/linear/LinearReference.cpp",
            "operators/gemm/linear/qw/q4gsw/Q4gswLinearReference.cpp",
        ],
        exported_headers = [
            "framework/Reference.h",
            "framework/ReferenceRegistry.h",
            "operators/binary/add/AddReference.h",
            "operators/gemm/matmul/MatmulReference.h",
            "operators/embedding/qw/q4gsw/EmbeddingReference.h",
            "operators/gemm/linear/qw/dq8ca_q4gsw/Dq8caQ4gswReference.h",
            "operators/attention/sdpa/SdpaReference.h",
            "operators/convolution/conv1d/Conv1dReference.h",
            "operators/quantize/q8ta/Q8taReferenceCommon.h",
            "operators/quantization/choose_qparams_per_row/ChooseQparamsReference.h",
            "operators/binary/BinaryReference.h",
            "operators/convolution/Conv2dReferenceCommon.h",
            "operators/convolution/Conv2dReference.h",
            "operators/convolution/q8ta/Q8taConv2dReference.h",
            "operators/gemm/GemmReference.h",
            "operators/gemm/linear/LinearReference.h",
            "operators/gemm/linear/qw/q4gsw/Q4gswLinearReference.h",
        ],
        compiler_flags = ["-O3"],
        # Static reference registrations must survive archive dead stripping.
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        platforms = get_platforms(),
        exported_deps = [
            ":report",
            "//executorch/runtime/core/exec_aten:lib",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "reference_test",
        srcs = ["framework/test/reference_test.cpp"],
        platforms = get_platforms(),
        deps = [
            ":reference",
            "//third-party/googletest:gtest_main",
        ],
    )

    runtime.cxx_test(
        name = "reference_registry_test",
        srcs = ["framework/test/reference_registry_test.cpp"],
        platforms = get_platforms(),
        deps = [
            ":reference",
            "//third-party/googletest:gtest_main",
        ],
    )

    runtime.cxx_library(
        name = "routes",
        srcs = [
            "framework/RouteRegistry.cpp",
            "operators/convolution/Conv2dRouteCommon.cpp",
            "operators/convolution/Conv2dRoutes.cpp",
            "operators/convolution/q8ta/Q8taConv2dRoutes.cpp",
            "operators/gemm/matmul/MatmulRoutes.cpp",
            "operators/convolution/conv2d/DepthwiseRoutes.cpp",
            "operators/gemm/linear/qw/q4gsw/Q4gswRoutes.cpp",
            "operators/attention/sdpa/SdpaRoutes.cpp",
            "operators/gemm/linear/q8ta/Q8taLinearRoutes.cpp",
            "operators/quantize/q8ta/Q8taQdqRoutes.cpp",
        ],
        exported_headers = [
            "framework/RouteRegistry.h",
            "operators/convolution/Conv2dRouteCommon.h",
            "operators/convolution/Conv2dRoutes.h",
            "operators/convolution/q8ta/Q8taConv2dRoutes.h",
        ],
        platforms = get_platforms(),
        exported_deps = [":config", ":migration_shaderlib", "//executorch/backends/vulkan:vulkan_graph_runtime"],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "route_registry_test",
        srcs = ["framework/test/route_registry_test.cpp"],
        platforms = get_platforms(),
        deps = [":routes", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_test(
        name = "operator_route_test",
        srcs = ["framework/test/operator_route_test.cpp"],
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan/test/api_abstractions:test_utils",
            ":routes",
            "//executorch/runtime/core/exec_aten:lib",
            "//third-party/googletest:gtest_main",
        ],
    )

    runtime.cxx_library(
        name = "gpu_clock",
        srcs = ["framework/GpuClock.cpp"],
        exported_headers = ["framework/GpuClock.h"],
        platforms = get_platforms(),
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "gpu_clock_test",
        srcs = ["framework/test/gpu_clock_test.cpp"],
        platforms = get_platforms(),
        deps = [":gpu_clock", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_library(
        name = "framework",
        srcs = [
            "framework/CaseCheckRegistry.cpp",
            "framework/OperatorBenchmark.cpp",
            "framework/ShaderBenchmark.cpp",
            "framework/TensorUtils.cpp",
            "framework/Timestamps.cpp",
            "framework/Timing.cpp",
            "framework/Validation.cpp",
            "framework/WorkgroupUtils.cpp",
        ],
        exported_headers = [
            "framework/CaseCheckRegistry.h",
            "framework/OperatorBenchmark.h",
            "framework/ShaderBenchmark.h",
            "framework/TensorUtils.h",
            "framework/Timestamps.h",
            "framework/Timing.h",
            "framework/Validation.h",
            "framework/WorkgroupUtils.h",
        ],
        platforms = get_platforms(),
        exported_deps = [
            ":config",
            ":gpu_clock",
            ":report",
            ":reference",
            ":routes",
            "//executorch/backends/vulkan:vulkan_graph_runtime",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "data_initializer_test",
        srcs = ["framework/test/data_initializer_test.cpp"],
        platforms = get_platforms(),
        deps = [":config", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_test(
        name = "config_test",
        srcs = ["framework/test/config_test.cpp"],
        platforms = get_platforms(),
        deps = [
            ":config",
            "//third-party/googletest:gtest_main",
        ],
    )

    runtime.cxx_test(
        name = "validation_test",
        srcs = ["framework/test/validation_test.cpp"],
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan/test/api_abstractions:test_utils",
            ":framework",
            "//third-party/googletest:gtest_main",
        ],
    )

    runtime.cxx_library(
        name = "case_checks",
        srcs = ["operators/convolution/q8ta/Q8taValidationChecks.cpp"],
        exported_headers = ["operators/convolution/q8ta/Q8taValidationChecks.h"],
        platforms = get_platforms(),
        exported_deps = [":framework"],
        # @lint-ignore BUCKLINT: Avoid `link_whole=True`
        link_whole = True,
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "case_check_registry_test",
        srcs = ["framework/test/case_check_registry_test.cpp"],
        platforms = get_platforms(),
        deps = [
            "//executorch/backends/vulkan/test/api_abstractions:test_utils",":framework", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_test(
        name = "migration_reference_test",
        srcs = ["operators/MigrationReferenceTest.cpp"],
        platforms = get_platforms(),
        deps = [":reference", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_test(
        name = "q8_benchmark_reference_test",
        srcs = ["operators/quantize/q8ta/test/q8_benchmark_reference_test.cpp"],
        platforms = get_platforms(),
        deps = [":reference", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_test(
        name = "q8ta_native_reference_test",
        srcs = ["operators/convolution/q8ta/test/q8ta_native_reference_test.cpp"],
        platforms = get_platforms(),
        deps = [":reference", "//third-party/googletest:gtest_main"],
    )

    runtime.cxx_binary(
        name = "etvk_ops_bench",
        srcs = ["etvk_ops_bench.cpp"],
        platforms = get_platforms(),
        define_static_target = False,
        deps = [
            ":framework",
            ":case_checks",
        ],
        external_deps = ["gflags"],
    )
