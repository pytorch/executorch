load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Dictionary mappping third-party library name to the correct OSS/Internal target dependencies.
# the values of the dictionary are lists where the first element is the internal dep and the
# second element is the OSS dep
_THIRD_PARTY_LIBS = {
    "FP16": ["fbsource//xplat/third-party/FP16:FP16Fbcode", "//backends/xnnpack/third-party:FP16"],
    "FXdiv": ["fbsource//xplat/third-party/FXdiv:FXdiv", "//backends/xnnpack/third-party:FXdiv"],
    "XNNPACK": ["fbsource//xplat/third-party/XNNPACK:XNNPACK", "//backends/xnnpack/third-party:XNNPACK"],
    "clog": ["fbsource//xplat/third-party/clog:clog", "//backends/xnnpack/third-party:clog"],
    "cpuinfo": ["fbsource//third-party/cpuinfo:cpuinfo", "//backends/xnnpack/third-party:cpuinfo"],
    "pthreadpool": ["fbsource//xplat/third-party/pthreadpool:pthreadpool", "//backends/xnnpack/third-party:pthreadpool"],
    "pthreadpool_header": ["fbsource//xplat/third-party/pthreadpool:pthreadpool_header", "//backends/xnnpack/third-party:pthreadpool_header"],
}

def third_party_dep(name):
    if name not in _THIRD_PARTY_LIBS:
        fail("Cannot find third party library " + name + ", please register it in THIRD_PARTY_LIBS first!")

    return _THIRD_PARTY_LIBS[name][1] if runtime.is_oss else _THIRD_PARTY_LIBS[name][0]
