load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/build:build_variables.bzl", "CPU_RUNTIME_BUCK_SRCS")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(":cpu_backend.bzl", "cpu_backend", "cpu_provider")

_EVENT_TRACER = "fbsource//xplat/executorch/tools/buck/constraints:event-tracer-enabled"
_OPERATOR_SHAPES = "fbsource//xplat/executorch/tools/buck/constraints:cpu-xnnpack-operator-shapes-enabled"

def _if_operator_shapes(values):
    """Returns values in builds that attach XNNPACK operator shapes to traces."""
    if native.read_config("executorch", "cpu_xnnpack_operator_shapes", "false") == "true":
        return values
    if runtime.is_oss:
        return []
    return select({
        "DEFAULT": [],
        _OPERATOR_SHAPES: values,
    })

def define_common_targets():
    runtime.cxx_library(
        name = "cpu_runtime",
        srcs = CPU_RUNTIME_BUCK_SRCS,
        exported_headers = [
            "runtime/KernelProvider.h",
            "runtime/CPUPlan.h",
            "runtime/op_helpers/Convolution.h",
        ],
        exported_deps = [
            "//executorch/backends/native/runtime:runtime",
            "//executorch/runtime/backend:interface",
        ],
        deps = ["//executorch/extension/threadpool:threadpool"],
        visibility = ["PUBLIC"],
    )

    runtime.cxx_library(
        name = "cpu_engine",
        srcs = ["runtime/CPUBackend.cpp"],
        exported_deps = [":cpu_runtime"],
        visibility = ["PUBLIC"],
        # @lint-ignore BUCKLINT: Backend registration runs at static initialization.
        link_whole = True,
    )

    runtime.cxx_library(
        name = "cpu_reference",
        srcs = ["runtime/providers/reference/ReferenceProvider.cpp"],
        exported_headers = ["runtime/providers/reference/ReferenceProvider.h"],
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
        preprocessor_flags = _if_operator_shapes(["-DENABLE_CPU_XNNPACK_OPERATOR_SHAPES"]),
        deps = [
            "//executorch/backends/xnnpack:xnnpack_profiler",
            "//executorch/extension/memory_allocator:malloc_memory_allocator",
            "//executorch/extension/threadpool:threadpool",
            third_party_dep("XNNPACK"),
        ] + _if_operator_shapes([third_party_dep("XNNPACK_internal_headers")]),
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
            cpu_provider(":cpu_reference", "executorch/backends/cpu/runtime/providers/reference/ReferenceProvider.h", "create_reference_provider"),
        ],
        kernel_libraries = ["//executorch/configurations:optimized_native_cpu_ops"],
    )

    runtime.cxx_binary(
        name = "runner",
        srcs = ["test/runner.cpp"],
        deps = [
            ":cpu_backend",
            "//executorch/devtools/etdump:etdump_flatcc",
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

    plan_test_deps = [
        ":cpu_runtime",
        ":cpu_reference",
        ":cpu_xnnpack",
        "//executorch/devtools/etdump:etdump_flatcc",
        "//executorch/configurations:optimized_native_cpu_ops",
    ]
    runtime.cxx_test(
        name = "plan_test",
        srcs = ["runtime/test/CPUPlanTest.cpp"],
        preprocessor_flags = _if_operator_shapes(["-DENABLE_CPU_XNNPACK_OPERATOR_SHAPES"]),
        deps = plan_test_deps,
    )

    if not runtime.is_oss:
        for suffix, modifiers in [
            ("_event_tracer", [_EVENT_TRACER]),
            ("_operator_shapes", [_EVENT_TRACER, _OPERATOR_SHAPES]),
        ]:
            runtime.cxx_test(
                name = "plan_test" + suffix,
                srcs = ["runtime/test/CPUPlanTest.cpp"],
                preprocessor_flags = _if_operator_shapes(["-DENABLE_CPU_XNNPACK_OPERATOR_SHAPES"]),
                deps = plan_test_deps,
                modifiers = modifiers,
            )

    runtime.cxx_test(
        name = "provider_test",
        srcs = ["runtime/test/ProviderTest.cpp"],
        deps = [
            ":buffer_test_util",
            ":cpu_reference",
            ":cpu_xnnpack",
            "//executorch/configurations:optimized_native_cpu_ops",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
        ],
    )
