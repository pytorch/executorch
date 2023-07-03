load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")
load(
    "@fbsource//xplat/executorch/kernels/optimized:lib_defs.bzl",
    "get_vec_android_preprocessor_flags",
    "get_vec_cxx_preprocessor_flags",
)
load("@fbsource//xplat/executorch/kernels/test:util.bzl", "define_supported_features_lib")

def _lib_test_bin(name, extra_deps = [], in_cpu = False):
    """Defines a cxx_binary() for a single test file.
    """
    if not (name.endswith("_test_bin")):
        fail("'{}' must match the pattern '*_vec_test_bin'")

    src_root = name[:-len("_bin")]
    lib_root = name[:-len("_test_bin")]

    cpu_path = "/cpu" if in_cpu else ""

    runtime.cxx_binary(
        name = name,
        srcs = [
            "{}.cpp".format(src_root),
        ],
        deps = [
            "//executorch/test/utils:utils",
            "//executorch/kernels/optimized{}:{}".format(cpu_path, lib_root),
        ] + extra_deps,
        cxx_platform_preprocessor_flags = get_vec_cxx_preprocessor_flags(),
        fbandroid_platform_preprocessor_flags = get_vec_android_preprocessor_flags(),
    )

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    define_supported_features_lib()

    _lib_test_bin("libvec_test_bin")
    _lib_test_bin("moments_utils_test_bin", in_cpu = True)
    _lib_test_bin("libblas_test_bin")
