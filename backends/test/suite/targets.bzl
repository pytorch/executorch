load("@fbcode_macros//build_defs:python_pytest.bzl", "python_pytest")
load("@fbsource//tools/target_determinator/macros:ci.bzl", "ci")
load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_tests_for_backend(name, deps):
    runtime.python_library(
        name = "suite_" + name + "_lib",
        srcs = [
            "flows/__init__.py",
            "flows/portable.py",
            "flows/" + name + ".py",
            "__init__.py",
            "context.py",
            "discovery.py",
            "flow.py",
            "reporting.py",
            "runner.py",
        ],
        deps = [
            "//executorch/backends/xnnpack/test/tester:tester",
            "//executorch/devtools/etrecord:etrecord",
            "//executorch/exir:lib",
            "fbsource//third-party/pypi/parameterized:parameterized",
            "fbsource//third-party/pypi/pytest:pytest",
        ] + deps,
        external_deps = [
            "libtorch",
        ],
    )

    python_pytest(
        name = "suite_" + name,
        srcs = glob([
            "operators/test_*.py",
        ]) + [
            "conftest.py",
            "operators/__init__.py",
        ],
        typing = False,
        deps = [
            ":suite_" + name + "_lib",
            "fbsource//third-party/pypi/pytest:pytest",
        ],
        supports_static_listing = False,
        labels = [
            "exclude_from_coverage",
        ] + ci.labels(ci.map(ci.skip_test())), # Manual only
        env = {
            "ET_TEST_BACKENDS": name,
        },
    )


def define_common_targets(is_fbcode):
    if is_fbcode:
        define_tests_for_backend("xnnpack", [
            "//executorch/backends/xnnpack/partition:xnnpack_partitioner",
        ])
