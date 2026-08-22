load("@fbsource//xplat/executorch/build:runtime_wrapper.bzl", "runtime")

ET_JNI_COMPILER_FLAGS = [
    "-frtti",
    "-fexceptions",
    "-Wno-unused-variable",
] + (
    ["-DEXECUTORCH_HAS_THREADPOOL_USE_N_THREADS_GUARD"] if not runtime.is_oss else []
)
