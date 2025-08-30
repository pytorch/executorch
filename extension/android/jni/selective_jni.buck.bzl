load("@fbcode_macros//build_defs:build_file_migration.bzl", "non_fbcode_target")
load("@fbsource//tools/build_defs/android:fb_android_cxx_library.bzl", "fb_android_cxx_library")
load("@fbsource//xplat/executorch/backends/xnnpack/third-party:third_party_libs.bzl", "third_party_dep")
load("@fbsource//xplat/executorch/extension/android/jni:build_defs.bzl", "ET_JNI_COMPILER_FLAGS")

def selective_jni_target(name, deps, srcs = [], soname = "libexecutorch.$(ext)"):
    non_fbcode_target(
        _kind = fb_android_cxx_library,
        name = name,
        srcs = [
            "//xplat/executorch/extension/android/jni:jni_layer.cpp",
            "//xplat/executorch/extension/android/jni:jni_layer_runtime.cpp",
            "//xplat/executorch/extension/android/jni:jni_helper.cpp",
        ] + srcs,
        allow_jni_merging = False,
        compiler_flags = ET_JNI_COMPILER_FLAGS,
        soname = soname,
        visibility = ["PUBLIC"],
        deps = [
            "//fbandroid/libraries/fbjni:fbjni",
            "//fbandroid/native/fb:fb",
            "//third-party/glog:glog",
            "//xplat/executorch/extension/android/jni:jni_headers",
            "//xplat/executorch/extension/android/jni:log_provider_static",
            "//xplat/executorch/extension/module:module_static",
            "//xplat/executorch/extension/runner_util:inputs_static",
            "//xplat/executorch/extension/tensor:tensor_static",
            "//xplat/executorch/extension/threadpool:threadpool_static",
            third_party_dep("cpuinfo"),
        ] + deps,
    )
