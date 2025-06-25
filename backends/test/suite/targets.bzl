load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_tests_for_backend(name, deps):
    runtime.python_test(
        name = "suite_" + name,
        srcs = glob([
            "operators/*.py",
        ]) + [
            "__init__.py",
        ],
        deps = [
            "//executorch/backends/xnnpack/test/tester:tester",
            "//executorch/exir:lib",
            "fbsource//third-party/pypi/parameterized:parameterized",
        ] + deps,
        external_deps = [
            "libtorch",
        ],
        supports_static_listing = False,
        labels = [
            "exclude_from_coverage",
        ],
        env = {
            "ET_TEST_BACKENDS": name,
        },
    )


def define_common_targets(is_fbcode):
    if is_fbcode:
        define_tests_for_backend("xnnpack", [
            "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
        ])

        define_tests_for_backend("coreml", [
            "//executorch/backends/apple/coreml:tester",
        ])
