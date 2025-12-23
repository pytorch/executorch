# load("//caffe2/test/fb:defs.bzl", "define_tests")
load("@fbsource//tools/build_defs:fbsource_utils.bzl", "is_fbcode")
load("@fbcode_macros//build_defs:python_pytest.bzl", "python_pytest")
load("@bazel_skylib//lib:paths.bzl", "paths")

def define_arm_tests():
    # TODO [fbonly] Add more tests
    test_files = []

    # Passes
    test_files += native.glob(["passes/test_*.py"])
    # https://github.com/pytorch/executorch/issues/8606
    test_files.remove("passes/test_ioquantization_pass.py")

    # Operators
    test_files += [
        "ops/test_add.py",
        "ops/test_addmm.py",
        "ops/test_avg_pool2d.py",
        "ops/test_cat.py",
        "ops/test_conv2d.py",
        "ops/test_linear.py", 
        "ops/test_mul.py",
        "ops/test_permute.py",
        "ops/test_rsqrt.py",
        "ops/test_slice.py",
        "ops/test_sigmoid.py",
        "ops/test_sub.py",
        "ops/test_tanh.py",
        "ops/test_view.py",
        "ops/test_cos.py",
        "ops/test_to_copy.py",
    ]

    # Quantization
    test_files += [
        "quantizer/test_generic_annotater.py",
    ]

    # Misc tests
    test_files += [
        "misc/test_compile_spec.py",
        "misc/test_tosa_spec.py",
        "misc/test_bn_relu_folding_qat.py",
        "misc/test_custom_partition.py",
        "misc/test_debug_hook.py",
        # "misc/test_dim_order.py", (TODO - T238390249)
        "misc/test_outputs_order.py",
    ]

    TESTS = {}

    for test_file in test_files:
        test_file_name = paths.basename(test_file)
        test_name = test_file_name.replace("test_", "").replace(".py", "")

        python_pytest(
            name = test_name,
            srcs = [test_file],
            pytest_config = "pytest.ini",
            resources = ["conftest.py"],
            compile = "with-source",
            typing = False,
            preload_deps = [
                "//executorch/kernels/quantized:custom_ops_generated_lib",
            ],
            deps = [
                "//executorch/backends/arm/test/tester/fb:arm_tester_fb" if is_fbcode else "//executorch/backends/arm/test:arm_tester",
                "//executorch/backends/arm/test:conftest",
                "//executorch/backends/arm/test/misc:dw_convs_shared_weights_module",
                "//executorch/backends/arm:ethosu",
                "//executorch/backends/arm/tosa:compile_spec",
                "//executorch/backends/arm/tosa:partitioner",
                "//executorch/backends/arm:vgf",
                "//executorch/exir:lib",
                "fbsource//third-party/pypi/pytest:pytest",
                "fbsource//third-party/pypi/parameterized:parameterized",
            ],
        )
