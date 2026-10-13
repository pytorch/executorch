load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(":cpu_backend.bzl", "cpu_backend", "cpu_provider")

def define_common_targets():
    runtime.cxx_library(
        name = "cpu_runtime",
        srcs = [
            "runtime/KernelProvider.cpp",
            "runtime/CPUPlan.cpp",
            "runtime/op_helpers/Convolution.cpp",
        ],
        exported_headers = [
            "runtime/KernelProvider.h",
            "runtime/CPUPlan.h",
            "runtime/op_helpers/Convolution.h",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime:runtime",
            "//executorch/runtime/backend:interface",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "cpu_engine",
        srcs = ["runtime/CPUBackend.cpp"],
        exported_deps = [":cpu_runtime"],
        deps = ["//executorch/extension/threadpool:threadpool"],
        visibility = ["PUBLIC"],
        # @lint-ignore BUCKLINT: Backend registration runs at static initialization.
        link_whole = True,
    )

    runtime.cxx_library(
        name = "cpu_et",
        srcs = ["runtime/providers/executorch/ETProvider.cpp"],
        exported_headers = ["runtime/providers/executorch/ETProvider.h"],
        exported_deps = [":cpu_runtime"],
        deps = [
            "//executorch/runtime/kernel:kernel_runtime_context",
            "//executorch/runtime/kernel:operator_registry",
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "cpu_xnnpack",
        srcs = ["runtime/providers/xnnpack/XNNPACKProvider.cpp"],
        exported_headers = ["runtime/providers/xnnpack/XNNPACKProvider.h"],
        exported_deps = [":cpu_runtime"],
        deps = [
            "//executorch/extension/threadpool:threadpool",
            third_party_dep("XNNPACK"),
        ],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_test(
        name = "xnnpack_scalar_test",
        srcs = ["test/XNNPACKScalarTest.cpp"],
        deps = [
            ":cpu_xnnpack",
            "//executorch/runtime/platform:platform",
        ],
    )

    cpu_backend(
        name = "cpu_backend",
        providers = [
            cpu_provider(":cpu_xnnpack", "executorch/backends/cpu/runtime/providers/xnnpack/XNNPACKProvider.h", "create_xnnpack_provider"),
            cpu_provider(":cpu_et", "executorch/backends/cpu/runtime/providers/executorch/ETProvider.h", "create_et_provider"),
        ],
        kernel_libraries = ["//executorch/configurations:optimized_native_cpu_ops"],
    )

    runtime.cxx_binary(
        name = "runner",
        srcs = ["test/runner.cpp"],
        deps = [
            ":cpu_backend",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
            "//executorch/extension/threadpool:threadpool",
        ],
    )

    runtime.cxx_library(
        name = "buffer_test_util",
        exported_headers = ["runtime/test/BufferTestUtil.h"],
        exported_deps = [":cpu_runtime"],
        visibility = [":provider_test"],
    )

    runtime.cxx_test(
        name = "plan_test",
        srcs = ["runtime/test/CPUPlanTest.cpp"],
        deps = [
            ":cpu_runtime",
            ":cpu_et",
            ":cpu_xnnpack",
            "//executorch/configurations:optimized_native_cpu_ops",
        ],
    )

    runtime.cxx_test(
        name = "provider_test",
        srcs = ["runtime/test/ProviderTest.cpp"],
        deps = [
            ":buffer_test_util",
            ":cpu_et",
            ":cpu_xnnpack",
            "//executorch/configurations:optimized_native_cpu_ops",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )
