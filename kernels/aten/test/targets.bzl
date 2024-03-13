load("@fbsource//xplat/executorch/kernels/test:util.bzl", "define_supported_features_lib")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """
    define_supported_features_lib()
