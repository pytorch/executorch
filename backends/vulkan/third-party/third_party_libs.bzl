load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

# Dictionary mappping third-party library name to the correct OSS/Internal target dependencies.
# the values of the dictionary are lists where the first element is the internal dep and the
# second element is the OSS dep
_THIRD_PARTY_LIBS = {
    "torch_vulkan_api": ["//caffe2:torch_vulkan_api", "//third-party:torch_vulkan_api"],
    "torch_vulkan_ops": ["//caffe2:torch_vulkan_ops", "//third-party:torch_vulkan_ops"],
}

def third_party_dep(name):
    if name not in _THIRD_PARTY_LIBS:
        fail("Cannot find third party library " + name + ", please register it in THIRD_PARTY_LIBS first!")

    return _THIRD_PARTY_LIBS[name][1] if runtime.is_oss else _THIRD_PARTY_LIBS[name][0]
