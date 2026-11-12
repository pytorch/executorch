load("@fbsource//tools/build_defs:platform_defs.bzl", "ANDROID", "CXX")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets(is_fbcode = False):
    if is_fbcode:
        runtime.python_binary(
            name = "export_models",
            srcs = ["export_models.py"],
            main_module = "executorch.backends.vulkan.test.external_io.export_models",
            deps = [
                "//caffe2:torch",
                "//executorch/backends/vulkan:vulkan_preprocess",
                "//executorch/backends/vulkan/partitioner:vulkan_partitioner",
                "//executorch/backends/vulkan/serialization:lib",
                "//executorch/exir:lib",
            ],
        )

        runtime.genrule(
            name = "exported_models",
            cmd = "$(exe :export_models) --outdir $OUT",
            out = "models",
            labels = ["reads_configerator", "justknobs"],
        )

        runtime.cxx_test(
            name = "external_io_test",
            srcs = ["external_io_test.cpp"],
            args = ["--models_dir=$(location :exported_models)"],
            env = {
                # Vulkan objects that live as long as the process, such as the
                # command buffers of the global context, are reported as leaks.
                "ASAN_OPTIONS": "detect_leaks=0",
            },
            deps = [
                "//executorch/backends/vulkan:vulkan_backend_lib",
                "//executorch/extension/module:module",
                "//executorch/extension/tensor:tensor",
                "//executorch/runtime/backend:backend_options_map",
            ],
            external_deps = ["gflags"],
        )
        return

    # For devices that the test above does not run on, e.g. Android, where it
    # also covers AHardwareBuffer resources. Pass --models_dir, a directory of
    # programs written by :export_models.
    runtime.cxx_binary(
        name = "external_io_test_bin",
        srcs = ["external_io_test.cpp"],
        platforms = [ANDROID, CXX],
        deps = [
            "//third-party/googletest:gtest",
            "//executorch/backends/vulkan:vulkan_backend_lib",
            "//executorch/extension/module:module",
            "//executorch/extension/tensor:tensor",
            "//executorch/runtime/backend:backend_options_map",
        ],
        external_deps = ["gflags"],
    )
