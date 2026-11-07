load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

ET_JNI_COMPILER_FLAGS = [
    "-frtti",
    "-fexceptions",
    "-Wno-unused-variable",
] + (
    ["-DEXECUTORCH_HAS_THREADPOOL_USE_N_THREADS_GUARD"] if not runtime.is_oss else []
)

def get_generic_jni_compiler_flags():
    if native.read_config("executorch", "use_generic_jni", "false") == "true":
        return ["-DEXECUTORCH_USE_GENERIC_JNI=1"]
    return []
