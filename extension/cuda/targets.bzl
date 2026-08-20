load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

def define_common_targets():
    """Defines targets that should be shared between fbcode and xplat.

    The directory containing this targets.bzl file should also contain both
    TARGETS and BUCK files that call this function.
    """

    # Backend-neutral: both the CUDA and TensorRT delegates can depend on it to
    # share a caller's stream. The caller-stream thread-local must be one
    # instance per process, so the main target stays shareable: OSS cxx_library
    # defaults force_static=True, which would duplicate the thread-local into
    # every dependent shared object (see export.h). The :caller_stream_static
    # variant stays available for fully-static consumers.
    runtime.cxx_library(
        name = "caller_stream",
        srcs = [
            "caller_stream.cpp",
        ],
        exported_headers = [
            "caller_stream.h",
            "export.h",
        ],
        # Opt out of the OSS force_static default so consumers *can* link one
        # shared instance and keep the thread-local unique (see above); the
        # wrapper pins preferred_linkage="any", so this allows shared linkage
        # rather than forcing it.
        force_static = False,
        # dllexport branch of export.h when building this lib; inert off Windows.
        preprocessor_flags = [
            "-DEXECUTORCH_EXTENSION_CUDA_BUILDING",
        ],
        visibility = ["PUBLIC"],
        external_deps = [
            ("cuda", None, "cuda-lazy"),
        ],
    )
