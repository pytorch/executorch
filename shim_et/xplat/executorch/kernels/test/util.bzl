load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "is_xplat", "runtime")

def op_test(name, deps = [], kernel_name = "portable", use_kernel_prefix = False):
    """Defines a cxx_test() for an "op_*_test.cpp" file.

    Args:
        name: "op_<operator-group-name>_test"; e.g., "op_add_test". Must match
            the non-extension part of the test source file (e.g.,
            "op_add_test.cpp"). This name must also agree with the target names
            under //kernels/<kernel>/...; e.g., "op_add_test" will depend on
            "//kernels/portable/cpu:op_add".
        deps: Optional extra deps to add to the cxx_test().
        kernel_name: The name string as in //executorch/kernels/<kernel_name>.
        use_kernel_prefix: If True, the target name is
            <kernel>_op_<operator-group-name>_test. Used by common kernel testing.
    """
    if not (name.startswith("op_") and name.endswith("_test")):
        fail("'{}' must match the pattern 'op_*_test'")
    op_root = name[:-len("_test")]  # E.g., "op_add" if name is "op_add_test".

    if kernel_name == "aten":
        generated_lib_and_op_deps = [
            "//executorch/kernels/aten:generated_lib",
            #TODO(T187390274): consolidate all aten ops into one target
            "//executorch/kernels/aten/cpu:op__to_dim_order_copy_aten",
            "//executorch/kernels/aten:generated_lib_headers",
            "//executorch/kernels/test:supported_features_aten",
        ]
    else:
        generated_lib_and_op_deps = [
            "//executorch/kernels/{}/cpu:{}".format(kernel_name, op_root),
            "//executorch/kernels/{}:generated_lib_headers".format(kernel_name),
            "//executorch/kernels/{}/test:supported_features".format(kernel_name),
        ]

    name_prefix = ""
    aten_suffix = ""
    if kernel_name == "aten":
        # For aten kernel, we need to use aten specific utils and types
        name_prefix = "aten_"
        aten_suffix = "_aten"
    elif use_kernel_prefix:
        name_prefix = kernel_name + "_"
    runtime.cxx_test(
        name = name_prefix + name,
        srcs = [
            "{}.cpp".format(name),
        ],
        visibility = ["//executorch/kernels/..."],
        deps = [
            "//executorch/runtime/core/exec_aten:lib" + aten_suffix,
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util" + aten_suffix,
            "//executorch/runtime/kernel:kernel_includes" + aten_suffix,
            "//executorch/kernels/test:test_util" + aten_suffix,
        ] + generated_lib_and_op_deps + deps,
    )

def generated_op_test(name, op_impl_target, generated_lib_headers_target, supported_features_target, function_header_wrapper_target, deps = []):
    """
    Build rule for testing an aten compliant op from an external kernel
    (outside of executorch/) and re-use test cases here, so we can compare
    between the external kernel and portable.

    Args:
        name: "op_<operator-group-name>_test"; e.g., "op_add_test".
        mandatory dependency targets:
              - op_impl_target (e.g. executorch/kernels/portable/cpu:op_add)
                required for testing the kernel impl
              - generated_lib_headers_target (e.g. executorch/kernels/portable:generated_lib_headers)
                required for dispatching op to the specific kernel
              - supported_features_target (e.g. executorch/kernels/portable/test:supported_features)
                required so we know which features that kernel support, and bypass unsupported tests
              - function_header_wrapper_target (e.g. executorch/kernels/portable/test:function_header_wrapper_portable)
                required so we can include a header wrapper for Functions.h. Use codegen_function_header_wrapper() to generate.
        deps: additional deps
    """
    runtime.cxx_test(
        name = name,
        srcs = [
            "fbsource//xplat/executorch/kernels/test:test_srcs_gen[{}.cpp]".format(name),
        ] if is_xplat() else [
            "//executorch/kernels/test:test_srcs_gen[{}.cpp]".format(name),
        ],
        deps = [
            "//executorch/runtime/core/exec_aten:lib",
            "//executorch/runtime/core/exec_aten/testing_util:tensor_util",
            "//executorch/runtime/kernel:kernel_includes",
            "//executorch/kernels/test:test_util",
            op_impl_target,
            generated_lib_headers_target,
            supported_features_target,
            function_header_wrapper_target,
        ] + deps,
    )

def define_supported_features_lib():
    runtime.genrule(
        name = "supported_feature_gen",
        cmd = "$(exe //executorch/kernels/test:gen_supported_features) ${SRCS} > $OUT/supported_features.cpp",
        srcs = ["supported_features_def.yaml"],
        outs = {"supported_features.cpp": ["supported_features.cpp"]},
        default_outs = ["."],
    )

    runtime.cxx_library(
        name = "supported_features",
        srcs = [":supported_feature_gen[supported_features.cpp]"],
        visibility = [
            "//executorch/kernels/...",
        ],
        exported_deps = [
            "//executorch/kernels/test:supported_features_header",
        ],
    )

def codegen_function_header_wrapper(kernel_path, kernel_name):
    """Produces a file (FunctionHeaderWrapper.h) which simply includes the real
    Functions.h for the specified kernel.

    Generate the wrapper for each kernel (except aten where we can use portable).
    Use target "function_header_wrapper_<kernel_name>" in tests.

    For ATen kernel, use portable as we use its functions.yaml
    """
    header = "\"#include <{}/Functions.h>\"".format(kernel_path)

    runtime.genrule(
        name = "gen_function_header_wrapper_{}".format(kernel_name),
        cmd = "echo " + header + " > $OUT/FunctionHeaderWrapper.h",
        outs = {"FunctionHeaderWrapper.h": ["FunctionHeaderWrapper.h"]},
        default_outs = ["."],
    )

    runtime.cxx_library(
        name = "function_header_wrapper_{}".format(kernel_name),
        exported_headers = {
            "FunctionHeaderWrapper.h": ":gen_function_header_wrapper_{}[FunctionHeaderWrapper.h]".format(kernel_name),
        },
        # TODO(T149423767): So far we have to expose this to users. Ideally this part can also be codegen.
        _is_external_target = True,
        visibility = ["//executorch/...", "//pye/..."],
    )
